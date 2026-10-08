"""The SIM/mock lot book — its model, its migration re-key, and its ledger mixin.

The per-scope conid book (``ledger``) is keyed by conid because IBKR names lots
by conid. The sim/mock path has no conid: its broker mints a synthetic
``position_id`` (``SYM_{ts}_{seq}``) and the mock fixture may hold lots the
strategy never opened. ``live_sim_lot`` is therefore the sim path's OWN durable
book — the lot plus the fill detail that opened it — so the sim reads as much
like a real execution record as the conid-keyed book does.

Ownership (an OPEN row) also scopes a sim close: only a lot the strategy OPENED
is closable, so an exogenous fixture lot is left alone. Keyed by ``scope`` — a
STABLE identity — so a config edit does not orphan every previously opened lot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import peewee
import pandas as pd

from src.live.ledger_base import (
    _Base,
    LedgerReadError,
    _SqliteOps,
    _is_missing_table,
    _ms,
    _ts,
    _table_columns,
    _table_exists,
)

from peewee import FloatField, IntegerField, TextField


@dataclass(frozen=True)
class SimLot:
    """One sim lot: the minted ``position_id``, its entry, and its exit (when closed).

    The detail is optional because a row may predate it (an ownership-only
    record): ``symbol``/``side``/``qty``/``entry_price`` are ``None`` then, so a
    caller can always tell "we own it" from "we know what it is" rather than
    reading an invented size as a real one. ``exit_price``/``exit_commission``
    are set only by a close, which is what makes the row a completed round trip.
    """

    position_id: str
    symbol: str | None = None
    side: str | None = None
    qty: float | None = None
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    tag: str | None = None
    opened_at: pd.Timestamp | None = None
    entry_commission: float | None = None
    exit_price: float | None = None
    exit_commission: float | None = None
    closed_at: pd.Timestamp | None = None

    @property
    def has_detail(self) -> bool:
        """Whether this row carries the lot's own size and entry."""
        return bool(self.symbol) and self.qty is not None

    @property
    def is_open(self) -> bool:
        """Whether the lot is still held (no exit recorded)."""
        return self.closed_at is None


class LiveSimLot(_Base):
    scope = TextField()
    position_id = TextField()
    #: Lot detail (NULL on an ownership-only row — see ``SimLot``).
    symbol = TextField(null=True)
    side = TextField(null=True)
    qty = FloatField(null=True)
    entry_price = FloatField(null=True)
    stop_loss = FloatField(null=True)
    take_profit = FloatField(null=True)
    tag = TextField(null=True)
    opened_at = IntegerField(null=True)
    entry_commission = FloatField(null=True)
    #: The exit leg: set by a close, which is what closes the round trip.
    exit_price = FloatField(null=True)
    exit_commission = FloatField(null=True)
    closed_at = IntegerField(null=True)

    class Meta:
        table_name = "live_sim_lot"
        primary_key = peewee.CompositeKey("scope", "position_id")


def _rekey_sim_lots(db: peewee.SqliteDatabase) -> None:
    """Re-key legacy ``live_sim_lot`` rows from the config hash to the scope.

    Pre-4.1 sim ownership was keyed by ``strategy_id`` (the config hash), so a
    parameter edit orphaned every opened lot; ownership is the scope. The column
    is renamed in place (SQLite rewrites the composite PK) and backfilled through
    the ``live_strategy`` audit link (``strategy_id`` -> ``scope``). A row with
    no link keeps an empty scope — PRESERVED, never dropped.
    """
    if not _table_exists(db, "live_sim_lot"):
        return
    columns = _table_columns(db, "live_sim_lot")
    if "scope" in columns or "strategy_id" not in columns:
        return
    db.execute_sql("ALTER TABLE live_sim_lot RENAME COLUMN strategy_id TO scope")
    if _table_exists(db, "live_strategy"):
        db.execute_sql(
            "UPDATE live_sim_lot SET scope = (SELECT s.scope FROM live_strategy s "
            "WHERE s.strategy_id = live_sim_lot.scope) "
            "WHERE scope IN (SELECT strategy_id FROM live_strategy)"
        )


class SimLotBook(_SqliteOps):
    """Ledger mixin: the sim/mock lot book (``live_sim_lot``)."""

    def record_sim_lot(self, scope: str, lot: SimLot) -> None:
        """Record a sim lot the strategy just opened (resurrects a closed one)."""
        with self._write():
            LiveSimLot.insert(
                scope=scope,
                position_id=lot.position_id,
                symbol=lot.symbol,
                side=lot.side,
                qty=lot.qty,
                entry_price=lot.entry_price,
                stop_loss=lot.stop_loss,
                take_profit=lot.take_profit,
                tag=lot.tag,
                opened_at=None if lot.opened_at is None else _ms(lot.opened_at),
                entry_commission=lot.entry_commission,
                exit_price=lot.exit_price,
                exit_commission=lot.exit_commission,
                closed_at=None if lot.closed_at is None else _ms(lot.closed_at),
            ).on_conflict("REPLACE").execute()

    def record_sim_open(self, scope: str, position_id: str) -> None:
        """Record an ownership-only sim lot (no fill detail), never clobbering one.

        A row that already carries detail is only re-opened, so a caller that
        knows less than the store never erases what it holds.
        """
        with self._write():
            reopened = (
                LiveSimLot.update(closed_at=None)
                .where(
                    (LiveSimLot.scope == scope)
                    & (LiveSimLot.position_id == position_id)
                )
                .execute()
            )
            if not reopened:
                LiveSimLot.insert(
                    scope=scope, position_id=position_id, closed_at=None
                ).execute()

    def mark_sim_closed(
        self,
        scope: str,
        position_id: str,
        closed_at: pd.Timestamp,
        exit_price: float | None = None,
        commission: float | None = None,
    ) -> None:
        """Stamp a sim lot closed, with the exit leg when the edge reported one.

        An unknown id is a no-op. Re-marking an already-closed lot is
        IDEMPOTENT: a close that does not say what it exited at never erases an
        exit already recorded.
        """
        with self._write():
            fields: dict[str, object] = {"closed_at": _ms(closed_at)}
            if exit_price is not None:
                fields["exit_price"] = exit_price
            if commission is not None:
                fields["exit_commission"] = commission
            LiveSimLot.update(**fields).where(
                (LiveSimLot.scope == scope) & (LiveSimLot.position_id == position_id)
            ).execute()

    def sim_lots(self, scope: str) -> tuple[SimLot, ...]:
        """Every sim lot this scope ever opened, oldest first (empty if unwritten)."""
        try:
            with self._database.bind_ctx((LiveSimLot,)):
                rows = (
                    LiveSimLot.select()
                    .where(LiveSimLot.scope == scope)
                    .order_by(LiveSimLot.opened_at, LiveSimLot.position_id)
                    .execute()
                )
        except peewee.OperationalError as exc:
            if not _is_missing_table(exc):
                raise LedgerReadError(str(exc)) from exc
            return ()
        return tuple(_sim_lot(row) for row in rows)

    def sim_open_lots(self, scope: str) -> tuple[SimLot, ...]:
        """The OPEN sim lots this scope owns, oldest first (empty if unwritten)."""
        return tuple(lot for lot in self.sim_lots(scope) if lot.is_open)

    def sim_open_ids(self, scope: str) -> frozenset[str]:
        """The sim lot ids this scope currently owns (empty if unwritten)."""
        return frozenset(lot.position_id for lot in self.sim_open_lots(scope))


def _sim_lot(row: LiveSimLot) -> SimLot:
    """Project one stored sim lot row (NULL detail stays ``None``)."""
    return SimLot(
        position_id=str(row.position_id),
        symbol=cast("str | None", row.symbol),
        side=cast("str | None", row.side),
        qty=cast("float | None", row.qty),
        entry_price=cast("float | None", row.entry_price),
        stop_loss=cast("float | None", row.stop_loss),
        take_profit=cast("float | None", row.take_profit),
        tag=cast("str | None", row.tag),
        opened_at=_ts(cast("int | None", row.opened_at)),
        entry_commission=cast("float | None", row.entry_commission),
        exit_price=cast("float | None", row.exit_price),
        exit_commission=cast("float | None", row.exit_commission),
        closed_at=_ts(cast("int | None", row.closed_at)),
    )
