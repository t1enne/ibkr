"""The SIM/mock lot book — its model, its migration re-key, and its ledger mixin.

The per-scope conid book (``ledger``) is keyed by conid because IBKR names lots
by conid. The sim/mock path has no conid: its broker mints a synthetic
``position_id`` (``SYM_{ts}_{seq}``) and the mock fixture may hold lots the
strategy never opened. Ownership is recorded here so a sim close can only target
a lot the strategy OPENED (``owned`` in ``reconcile``); a fixture lot never opened
is left alone. Keyed by ``scope`` — a STABLE identity — so a config edit does not
orphan every previously opened lot.
"""

from __future__ import annotations

import peewee
import pandas as pd

from src.live.ledger_base import (
    _Base,
    LedgerReadError,
    _SqliteOps,
    _is_missing_table,
    _ms,
    _table_columns,
    _table_exists,
)

from peewee import IntegerField, TextField


class LiveSimLot(_Base):
    scope = TextField()
    position_id = TextField()
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
    """Ledger mixin: the sim/mock lot ownership store (``live_sim_lot``)."""

    def record_sim_open(self, scope: str, position_id: str) -> None:
        """Record a sim lot the strategy just opened (resurrects a closed one)."""
        with self._write():
            LiveSimLot.insert(
                scope=scope, position_id=position_id, closed_at=None
            ).on_conflict("REPLACE").execute()

    def mark_sim_closed(
        self, scope: str, position_id: str, closed_at: pd.Timestamp
    ) -> None:
        """Stamp a sim lot closed; an unknown id is a no-op."""
        with self._write():
            LiveSimLot.update(closed_at=_ms(closed_at)).where(
                (LiveSimLot.scope == scope) & (LiveSimLot.position_id == position_id)
            ).execute()

    def sim_open_ids(self, scope: str) -> frozenset[str]:
        """The sim lot ids this scope currently owns (empty if unwritten)."""
        try:
            with self._database.bind_ctx((LiveSimLot,)):
                rows = (
                    LiveSimLot.select(LiveSimLot.position_id)
                    .where((LiveSimLot.scope == scope) & LiveSimLot.closed_at.is_null())
                    .execute()
                )
        except peewee.OperationalError as exc:
            if not _is_missing_table(exc):
                raise LedgerReadError(str(exc)) from exc
            return frozenset()
        return frozenset(str(r.position_id) for r in rows)
