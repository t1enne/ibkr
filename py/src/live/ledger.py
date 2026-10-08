"""Per-scope sqlite book — the durable, per-strategy position store (plan §3).

The 7-day ``/iserver/account/trades`` window is a *confirmation channel*: it
advances the book. The book itself lives here, ONE table (``live_position``), one
row per ``(scope, position_id)``, plus one row per applied ``execution_id`` so
re-applying the window is a no-op. The table carries both book ROLES, told apart
by ``source``: ``executions`` rows are the immutable fold of our own fills (the
IBKR reconcile book, keyed ``str(conid)``), ``account`` rows are the
human/broker-editable exposure surface (the sim lot book).
Live-only: the backtest persists nothing. Tables live in the SAME candle DB
(``src.data.db``), created idempotently on first WRITE — a read (and therefore a
``--dry-run``) writes no DDL.

``strategy_id`` (the config hash) survives on ``live_strategy`` as an AUDIT
column only — never an ownership filter. Ownership is the ``scope`` (the cOID
prefix, §4). ``live_cash`` carries each scope's ``initial_capital``; per-scope
cash is derived from the scope's own executions, never read from the account
summary (N strategies share one account).

Storage is peewee ORM over the same SQLite file. The model DDL mirrors the
previous raw ``CREATE TABLE`` SQL column-for-column, so an existing phase-3.5
book rows read back unchanged (``create_tables`` is ``IF NOT EXISTS`` and skips
already-present matching tables). DDL stays lazy: ``create_tables`` runs only
inside a WRITE path, never at import, never for a read/``--dry-run``.

``SqliteLedger`` is the seam implementing the live Protocols; its per-domain
methods live in their own mixins (``SimLotStore`` and the stores below) so each
class stays small. Shared plumbing (the template binding, the base model, the
epoch codec) is in ``ledger_base``; the lossless migrations are in
``ledger_migration``. Every public method keeps its name and signature.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pandas as pd
import peewee
from peewee import SqliteDatabase, fn

from src.bt.state import ActionType
from src.db.migrations.runner import run_pending
from src.db.migrations.versions import LIVE_MIGRATIONS
from src.db.path import resolve_live_db_path
from src.exec.refs import scope_tag
from src.exec.types import OrderSide
from src.live.adapters.ibkr.trades import (
    BookRow,
    Execution,
    StrategyBook,
    is_ours,
)
from src.live.pure import OrderResult
from src.live.identity import (
    OPEN_STATES,
    IntentKey,
    IntentRecord,
    IntentState,
    order_ref,
)
from src.live.lease import file_lease
from src.live.ledger_base import (
    _is_missing_table,
    _ms,
    _SqliteOps,
    _ts,
    LedgerReadError,
)
from src.live.models import (
    LIVE_MODELS,
    SOURCE_ACCOUNT,
    SOURCE_EXECUTIONS,
    LiveCash,
    LiveExecution,
    LiveOrderIntent,
    LivePosition,
    LiveStrategy,
)

logger = logging.getLogger(__name__)

#: The ``bind_ctx`` group: every live model, rebindable per ledger instance.
#: Rebound here (not globally) because peewee binds at CLASS level and two ledgers
#: on two paths share these classes — see :mod:`src.live.models`.
_MODELS = LIVE_MODELS


def _alias_scope(db: peewee.SqliteDatabase, scope: str) -> str:
    """Follow a migration scope alias: a re-keyed legacy scope reads as the new one.

    ``migrate`` re-keys a bare legacy scope ``X`` to ``ibkr_X_legacy`` in every
    live table. Without this lookup the operator's still-config'd ``X`` would
    address an EMPTY book, so the strategy would read itself flat and re-open on
    top of live positions. An absent table (fresh db) or no row for *scope* means
    "not a legacy scope", so the scope is returned unchanged.
    """
    try:
        row = db.execute_sql(
            "SELECT new_scope FROM live_scope_alias WHERE legacy_scope=?", (scope,)
        ).fetchone()
    except peewee.OperationalError:
        return scope
    return scope if row is None else str(row[0])


@dataclass(frozen=True)
class StrategyAudit:
    """One ``live_strategy`` audit row: which config revision wrote this scope.

    ``strategy_id`` is the config HASH (an audit key, never an ownership filter);
    ``created_at``/``last_cycle_at`` are the epoch-ms columns read back as UTC
    timestamps (``None`` when unwritten).
    """

    strategy_id: str
    scope: str
    name: str
    mode: str
    created_at: pd.Timestamp | None
    last_cycle_at: pd.Timestamp | None


@dataclass(frozen=True)
class ExecutionRecord:
    """One stored fill (``live_execution``) projected for the report.

    ``live_execution`` stores ``position_id`` but no symbol, so ``symbol`` is
    resolved from the scope's book rows (the lot's symbol, ``""`` when no row
    remains). ``ts`` is the fill's UTC instant.
    """

    scope: str
    execution_id: str
    #: ``None`` for a sim fill (a sim lot has no conid). IBKR rows derive it from
    #: the stored ``position_id``.
    conid: int | None
    symbol: str
    side: str
    qty: float
    price: float
    commission: float
    cash_delta: float
    ts: pd.Timestamp | None
    #: The lot this fill belongs to, in the BOOK's id space: ``str(conid)`` for an
    #: IBKR row, the sim lot's own minted id for a sim row. This is the fold key
    #: ``book_from_executions`` groups on; ``conid`` cannot serve (it is ``None``
    #: for every sim fill, which would merge a symbol's whole sim book into one lot).
    #: Last and defaulted so an existing caller that only knows the conid still
    #: constructs.
    position_id: str = ""


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


#: The terminal (closed) intent states ``prune`` is allowed to delete.
_CLOSED_INTENTS = [
    IntentState.FILLED.value,
    IntentState.UNFILLED.value,
    IntentState.REJECTED.value,
]


class MetadataStore(_SqliteOps):
    """Ledger mixin: the audit/metadata tables (``live_strategy``/``live_cash``)."""

    def scopes_of_store(self) -> tuple[str, ...]:
        """Every scope the store knows about, sorted (empty when unwritten).

        The union of the tables that carry a ``scope`` column, so a scope whose
        strategy row was never written but which holds lots, cash or intents is
        still reported. A genuinely missing table contributes nothing; any other
        read failure raises ``LedgerReadError``.
        """
        found: set[str] = set()
        with self._database.bind_ctx(_MODELS):
            for model in (
                LiveStrategy,
                LiveCash,
                LivePosition,
                LiveExecution,
                LiveOrderIntent,
            ):
                try:
                    rows = model.select(model.scope).distinct().execute()
                except peewee.OperationalError as exc:
                    if not _is_missing_table(exc):
                        raise LedgerReadError(str(exc)) from exc
                    continue
                found.update(str(row.scope) for row in rows)
        return tuple(sorted(found))

    def ensure_strategy(
        self, strategy_id: str, scope: str, name: str, mode: str
    ) -> None:
        """Record the strategy audit row; warn when a run reuses another's scope.

        Ownership is the ``scope`` (one book, one cash seed, one cOID prefix), but
        ``strategy_id`` is the config HASH — it changes on ANY parameter edit, so a
        differing hash on the same scope is NORMAL (a tuned config). It is a real
        danger only when a DIFFERENT config file reuses a scope: two strategies
        then share one book, cash seed and cOID prefix (identical cOIDs within one
        second). The two are indistinguishable here (no path is stored, and an edit
        changes the hash), so this WARNS rather than refuses — a hard refuse would
        wedge every config edit. Give the other config its own ``scope``.
        """
        with self._write():
            scope = _alias_scope(self._database, scope)
            others = (
                LiveStrategy.select(LiveStrategy.strategy_id)
                .where(
                    (LiveStrategy.scope == scope)
                    & (LiveStrategy.strategy_id != strategy_id)
                )
                .execute()
            )
            for row in others:
                logger.warning(
                    "scope %r already belongs to strategy %s; strategy %s now shares "
                    "its book, cash seed and cOID prefix — if these are two DIFFERENT "
                    "configs, give this one its own scope",
                    scope,
                    row.strategy_id,
                    strategy_id,
                )
            LiveStrategy.insert(
                strategy_id=strategy_id,
                scope=scope,
                name=name,
                mode=mode,
                created_at=_ms(pd.Timestamp.now()),
            ).on_conflict("IGNORE").execute()

    def ensure_cash(self, scope: str, initial_capital: float) -> None:
        """Record the scope's ``initial_capital`` once; a later cycle never resets it."""
        with self._write():
            scope = _alias_scope(self._database, scope)
            LiveCash.insert(
                scope=scope,
                initial_capital=initial_capital,
                updated_at=_ms(pd.Timestamp.now()),
            ).on_conflict("IGNORE").execute()

    def touch_cycle(self, strategy_id: str, at: pd.Timestamp) -> None:
        with self._write():
            LiveStrategy.update(last_cycle_at=_ms(at)).where(
                LiveStrategy.strategy_id == strategy_id
            ).execute()

    def strategies_of(self, scope: str) -> tuple[StrategyAudit, ...]:
        """Every audit row sharing *scope*, oldest first (empty if the table is absent).

        Read-only: this is the ``ibkr live pf`` view of which config revisions have
        written the scope. A missing table (the never-written case) reads as empty,
        like the book reads; any OTHER ``OperationalError`` is a genuine read
        failure and raises :class:`LedgerReadError` rather than reporting nothing.
        """
        scope = _alias_scope(self._database, scope)
        try:
            with self._database.bind_ctx(_MODELS):
                rows = (
                    LiveStrategy.select()
                    .where(LiveStrategy.scope == scope)
                    .order_by(LiveStrategy.created_at, LiveStrategy.strategy_id)
                    .execute()
                )
        except peewee.OperationalError as exc:
            if not _is_missing_table(exc):
                raise LedgerReadError(str(exc)) from exc
            return ()
        return tuple(_audit_of(row) for row in rows)


class BookStore(_SqliteOps):
    """Ledger mixin: the IBKR reconcile book, its executions and its cash seed.

    This mixin owns the ``source='executions'`` ROLE of ``live_position`` — the
    fold of our own fills that ``trades.reconcile`` advances. The sim's
    ``source='account'`` role is :class:`SimLotStore`; keeping the roles on
    distinct ``source`` values is what stops one adapter's rows from
    double-counting as the other's (the two never share a scope either).
    """

    def sim_lots(self, scope: str) -> tuple[SimLot, ...]:
        """Every sim lot for *scope*: provided by :class:`SimLotStore`.

        Declared here so this mixin's execution/cash reads can compose the sim
        book with the fill fold; the concrete ledger inherits ``SimLotStore``
        FIRST, so this stub is never the one that runs.
        """
        raise NotImplementedError

    def load_book(self, scope: str) -> StrategyBook:
        """The durable rows + applied execution ids for *scope* (empty if unaware)."""
        scope = _alias_scope(self._database, scope)
        try:
            with self._database.bind_ctx(_MODELS):
                rows = (
                    LivePosition.select()
                    .where(
                        (LivePosition.scope == scope)
                        & (LivePosition.source == SOURCE_EXECUTIONS)
                    )
                    .order_by(LivePosition.position_id)
                    .execute()
                )
                applied = {
                    e.execution_id
                    for e in LiveExecution.select(LiveExecution.execution_id).where(
                        LiveExecution.scope == scope
                    )
                }
        except peewee.OperationalError as exc:
            if not _is_missing_table(exc):
                raise LedgerReadError(str(exc)) from exc
            return StrategyBook()
        return StrategyBook(
            rows=tuple(_model_to_book(row) for row in rows),
            applied=frozenset(applied),
        )

    def save_book(
        self,
        scope: str,
        book: StrategyBook,
        executions: tuple[Execution, ...],
        initial_capital: float,
    ) -> None:
        """Persist the advanced book, the newly applied executions and the cash seed.

        Executions are inserted ``OR IGNORE`` keyed ``(scope, execution_id)``, so
        the cash ledger accumulates each fill exactly once even across re-runs.
        """
        with self._write():
            scope = _alias_scope(self._database, scope)
            LiveCash.insert(
                scope=scope,
                initial_capital=initial_capital,
                updated_at=_ms(pd.Timestamp.now()),
            ).on_conflict("IGNORE").execute()
            for row in book.rows:
                LivePosition.insert(**_book_fields(scope, row)).on_conflict(
                    "REPLACE"
                ).execute()
            for execution in executions:
                if execution.execution_id not in book.applied:
                    continue
                if not is_ours(scope, execution):
                    continue
                LiveExecution.insert(
                    scope=scope,
                    execution_id=execution.execution_id,
                    position_id=str(execution.conid),
                    side=execution.side.value,
                    qty=execution.qty,
                    price=execution.price,
                    commission=execution.commission,
                    cash_delta=_cash_delta(execution),
                    ts=_ms(execution.ts),
                ).on_conflict("IGNORE").execute()

    def executions_of(self, scope: str) -> tuple[ExecutionRecord, ...]:
        """Stored fill history for *scope*, oldest first (empty if unwritten).

        One :class:`ExecutionRecord` per applied fill, `ts` ascending, with
        `symbol` resolved from the lot's book row (`""` when none remains). A
        missing table (a never-written scope) reads as empty; a genuine read
        failure raises :class:`LedgerReadError`.
        """
        scope = _alias_scope(self._database, scope)
        try:
            with self._database.bind_ctx(_MODELS):
                symbols = {
                    cast("str", row.position_id): cast("str", row.symbol)
                    for row in LivePosition.select(
                        LivePosition.position_id, LivePosition.symbol
                    ).where(LivePosition.scope == scope)
                }
                rows = (
                    LiveExecution.select()
                    .where(LiveExecution.scope == scope)
                    .order_by(LiveExecution.ts)
                    .execute()
                )
        except peewee.OperationalError as exc:
            if not _is_missing_table(exc):
                raise LedgerReadError(str(exc)) from exc
            return ()
        return tuple(_model_to_execution(row, symbols) for row in rows)

    def sim_executions(self, scope: str) -> tuple[ExecutionRecord, ...]:
        """The sim path's fills for *scope* that no stored execution row already holds.

        The sim has no execution stream to replay, so its OWN lot book is the
        record: each lot contributes an entry fill and, once closed, an exit
        fill. This is what lets a sim scope read like a real one — the same
        ``_flows``/commission/realized math the IBKR fills feed — instead of a
        book with no fills at all (which reads an open lot's cost as profit).

        A fill :meth:`record_results` already persisted is SKIPPED (both paths
        mint the same ``<pid>:open``/``:close`` id), so a recorded fill counts ONCE
        — the derived view and the stored rows are two views of one book, never
        two books. ``conid`` is ``None``: a sim lot has no conid, so the symbol is
        carried directly.
        """
        stored = {record.execution_id for record in self.executions_of(scope)}
        return tuple(
            record
            for lot in self.sim_lots(scope)
            for record in _sim_fill_records(scope, lot)
            if record.execution_id not in stored
        )

    def sim_cash_delta(self, scope: str) -> float:
        """Signed cash the sim lots already moved (entry debits, exits credits)."""
        return sum(record.cash_delta for record in self.sim_executions(scope))

    def cash_of(self, scope: str, default_initial: float = 0.0) -> float:
        """Per-scope cash: ``initial_capital`` advanced by the scope's own fills.

        The stored ``initial_capital`` (first cycle's config value) wins; a scope
        with no stored row falls back to *default_initial*. Never reads the
        account summary: N strategies share one account's cash. Both books
        contribute: the stored fills AND the sim lot book, so a sim scope's cash
        tracks its own fills exactly as a real one's does.
        """
        scope = _alias_scope(self._database, scope)
        try:
            with self._database.bind_ctx(_MODELS):
                cash = LiveCash.get_or_none(LiveCash.scope == scope)
                initial = (
                    float(cash.initial_capital) if cash is not None else default_initial
                )
                sunk = (
                    LiveExecution.select(
                        fn.COALESCE(fn.SUM(LiveExecution.cash_delta), 0.0)
                    )
                    .where(LiveExecution.scope == scope)
                    .scalar()
                )
        except peewee.OperationalError as exc:
            if not _is_missing_table(exc):
                raise LedgerReadError(str(exc)) from exc
            return default_initial
        return initial + float(sunk) + self.sim_cash_delta(scope)

    def initial_capital_of(self, scope: str) -> float:
        scope = _alias_scope(self._database, scope)
        try:
            with self._database.bind_ctx(_MODELS):
                cash = LiveCash.get_or_none(LiveCash.scope == scope)
        except peewee.OperationalError as exc:
            if not _is_missing_table(exc):
                raise LedgerReadError(str(exc)) from exc
            return 0.0
        return float(cash.initial_capital) if cash is not None else 0.0

    def prune_closed(self, before: pd.Timestamp) -> int:
        """Delete closed rows older than *before*; return the deleted count.

        The only DELETE over book rows — open rows are never pruned.
        """
        with self._write():
            return (
                LivePosition.delete()
                .where(
                    LivePosition.closed_at.is_null(False)
                    & (LivePosition.closed_at < _ms(before))
                )
                .execute()
            )

    def net_exposure(self, conid: int) -> float:
        """Signed net quantity the book holds on *conid*, summed over ALL scopes.

        Long rows add, short rows subtract, so the result is the net a broker
        account would show if every booked fill were the whole story. Only the
        ``executions`` ROLE counts: the id space is a conid's, and an ``account``
        role row (a sim lot, keyed by a minted id) is not a conid at all —
        including it would inflate the net this guard reconciles against the
        broker, which is what gates a close refusal. A missing table reads as a
        flat (0.0) book — the dry-run case; a genuine read failure still raises
        (:class:`LedgerReadError`), never a silent zero.
        """
        try:
            with self._database.bind_ctx(_MODELS):
                rows = (
                    LivePosition.select(LivePosition.side, LivePosition.qty)
                    .where(
                        (LivePosition.position_id == str(conid))
                        & (LivePosition.source == SOURCE_EXECUTIONS)
                        & LivePosition.closed_at.is_null()
                    )
                    .execute()
                )
        except peewee.OperationalError as exc:
            if not _is_missing_table(exc):
                raise LedgerReadError(str(exc)) from exc
            return 0.0
        return sum(
            float(row.qty) if row.side == "long" else -float(row.qty) for row in rows
        )


class SimLotStore(_SqliteOps):
    """Ledger mixin: the sim/mock lot book, in ``live_position source='account'``.

    The sim/mock broker mints its own ``position_id`` (``SYM_{ts}_{seq}``) and the
    mock fixture may hold lots the strategy never opened, so the sim path needs
    its own durable record of what it holds. It shares the ONE book table with the
    IBKR reconcile book; ``source`` keeps the two roles apart (and the scope
    embeds the adapter, so they never share a scope either).

    Ownership (an OPEN row) also scopes a sim close: only a lot the strategy
    OPENED is closable, so an exogenous fixture lot is left alone. Keyed by
    ``scope`` — a STABLE identity — so a config edit does not orphan open lots.
    """

    def record_sim_lot(self, scope: str, lot: SimLot) -> None:
        """Record a sim lot the strategy just opened (resurrects a closed one)."""
        with self._write():
            scope = _alias_scope(self._database, scope)
            LivePosition.insert(
                scope=scope,
                position_id=lot.position_id,
                symbol=lot.symbol or "",
                side=lot.side or "",
                qty=0.0 if lot.qty is None else lot.qty,
                entry_price=0.0 if lot.entry_price is None else lot.entry_price,
                stop_loss=lot.stop_loss,
                take_profit=lot.take_profit,
                tag=lot.tag or "",
                opened_at=None if lot.opened_at is None else _ms(lot.opened_at),
                closed_at=None if lot.closed_at is None else _ms(lot.closed_at),
                entry_commission=lot.entry_commission,
                exit_price=lot.exit_price,
                exit_commission=lot.exit_commission,
                source=SOURCE_ACCOUNT,
            ).on_conflict("REPLACE").execute()

    def record_sim_open(self, scope: str, position_id: str) -> None:
        """Record an ownership-only sim lot (no fill detail), never clobbering one.

        A row that already carries detail is only re-opened, so a caller that
        knows less than the store never erases what it holds.
        """
        with self._write():
            scope = _alias_scope(self._database, scope)
            reopened = (
                LivePosition.update(closed_at=None)
                .where(
                    (LivePosition.scope == scope)
                    & (LivePosition.position_id == position_id)
                    & (LivePosition.source == SOURCE_ACCOUNT)
                )
                .execute()
            )
            if not reopened:
                LivePosition.insert(
                    scope=scope,
                    position_id=position_id,
                    symbol="",
                    side="",
                    qty=0.0,
                    entry_price=0.0,
                    tag="",
                    closed_at=None,
                    source=SOURCE_ACCOUNT,
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
            scope = _alias_scope(self._database, scope)
            fields: dict[str, object] = {"closed_at": _ms(closed_at)}
            if exit_price is not None:
                fields["exit_price"] = exit_price
            if commission is not None:
                fields["exit_commission"] = commission
            LivePosition.update(**fields).where(
                (LivePosition.scope == scope)
                & (LivePosition.position_id == position_id)
                & (LivePosition.source == SOURCE_ACCOUNT)
            ).execute()

    def sim_lots(self, scope: str) -> tuple[SimLot, ...]:
        """Every sim lot this scope ever opened, oldest first (empty if unwritten)."""
        scope = _alias_scope(self._database, scope)
        try:
            with self._database.bind_ctx(_MODELS):
                rows = (
                    LivePosition.select()
                    .where(
                        (LivePosition.scope == scope)
                        & (LivePosition.source == SOURCE_ACCOUNT)
                    )
                    .order_by(LivePosition.opened_at, LivePosition.position_id)
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

    def record_results(
        self,
        scope: str,
        results: tuple[OrderResult, ...],
        now: pd.Timestamp,
    ) -> None:
        """Record a cycle's placement results into the book — the ONE write point.

        The engine hands over what the adapter RETURNED (adapters hold no book),
        so this mirrors the pre-seam ``_record_owned`` semantics: a failed or
        rejected result records nothing, an OPEN upserts the account lot with the
        fill's detail (an unnamed open can never be targeted by a close, so it
        records nothing), and a CLOSE marks the targeted lot closed without
        erasing an exit already recorded.

        Each touched lot's implied fills are ALSO persisted into
        ``live_execution``, so the ledger holds the same fill rows a real broker
        would replay while :meth:`sim_executions` skips what is already stored.
        """
        for result in results:
            if not result.ok:
                continue
            intent = result.intent
            if intent.action is ActionType.close:
                if not intent.position_id:
                    continue
                self.mark_sim_closed(
                    scope,
                    intent.position_id,
                    now,
                    exit_price=result.fill.executed_price if result.fill else None,
                    commission=result.fill.commission if result.fill else None,
                )
                pid = intent.position_id
            else:
                if not result.position_id:
                    continue
                self.record_sim_lot(scope, _result_lot(result))
                pid = result.position_id
            for record in self._fills_of(scope, pid):
                self._insert_execution(record)

    def _fills_of(self, scope: str, position_id: str) -> tuple[ExecutionRecord, ...]:
        """The fills the stored lot with *position_id* now implies (empty if none)."""
        for lot in self.sim_lots(scope):
            if lot.position_id == position_id:
                return _sim_fill_records(scope, lot)
        return ()

    def _insert_execution(self, record: ExecutionRecord) -> None:
        """Persist one already-derived fill row, keyed so a re-run is a no-op."""
        with self._write():
            LiveExecution.insert(
                scope=record.scope,
                execution_id=record.execution_id,
                position_id=record.position_id,
                side=record.side,
                qty=record.qty,
                price=record.price,
                commission=record.commission,
                cash_delta=record.cash_delta,
                ts=0 if record.ts is None else _ms(record.ts),
            ).on_conflict("IGNORE").execute()


def _result_lot(result: OrderResult) -> SimLot:
    """The lot an OPEN fill created, from the fill and the intent behind it.

    A result with no fill still records the lot (ownership is never lost to
    missing detail): size and entry simply stay unknown, and the row reads as an
    ownership-only one rather than inventing a position.
    """
    fill = result.fill
    return SimLot(
        position_id=cast("str", result.position_id),
        symbol=result.intent.symbol,
        side=result.intent.action.value,
        qty=fill.filled_qty if fill is not None else None,
        entry_price=fill.executed_price if fill is not None else None,
        stop_loss=result.intent.stop_loss,
        take_profit=result.intent.take_profit,
        tag=result.intent.tag or None,
        opened_at=fill.timestamp if fill is not None else None,
        entry_commission=fill.commission if fill is not None else None,
    )


def _sim_lot(row: LivePosition) -> SimLot:
    """Project one stored lot row. An ownership-only row (no symbol) reads bare."""
    symbol = cast("str", row.symbol)
    detail = bool(symbol)
    return SimLot(
        position_id=cast("str", row.position_id),
        symbol=symbol if detail else None,
        side=cast("str", row.side) if detail else None,
        qty=cast("float", row.qty) if detail else None,
        entry_price=cast("float", row.entry_price) if detail else None,
        stop_loss=cast("float | None", row.stop_loss),
        take_profit=cast("float | None", row.take_profit),
        tag=cast("str | None", row.tag) or None,
        opened_at=_ts(cast("int | None", row.opened_at)),
        entry_commission=cast("float | None", row.entry_commission),
        exit_price=cast("float | None", row.exit_price),
        exit_commission=cast("float | None", row.exit_commission),
        closed_at=_ts(cast("int | None", row.closed_at)),
    )


class IntentStore(_SqliteOps):
    """Ledger mixin: the durable owner of OPEN order state (``PendingIntents``).

    Reads tolerate a missing table (empty), like ``load_book``; writes create it
    lazily. The identity columns (scope/symbol/action/position_id) are the primary
    key, so two intents whose crc32 tokens collide stay separately addressable by
    IDENTITY. NOTE: that is a ROW-level guarantee only — the broker still
    attributes a working order by the token-keyed ref prefix, so two colliding
    keys share a prefix there.
    """

    def load(self, key: IntentKey) -> IntentRecord | None:
        """The durable record for *key*, or ``None`` when unwritten."""
        key = self._key(key)
        try:
            with self._database.bind_ctx(_MODELS):
                row = LiveOrderIntent.get_or_none(self._intent_predicate(key))
        except peewee.OperationalError as exc:
            if not _is_missing_table(exc):
                raise LedgerReadError(str(exc)) from exc
            return None
        return None if row is None else _model_to_intent(row)

    def load_open(self, scope: str) -> tuple[IntentRecord, ...]:
        """Every OPEN record for *scope* (empty if unwritten)."""
        scope = _alias_scope(self._database, scope)
        states = [s.value for s in OPEN_STATES]
        try:
            with self._database.bind_ctx(_MODELS):
                rows = (
                    LiveOrderIntent.select()
                    .where(
                        (LiveOrderIntent.scope == scope)
                        & (LiveOrderIntent.state << states)
                    )
                    .execute()
                )
        except peewee.OperationalError as exc:
            if not _is_missing_table(exc):
                raise LedgerReadError(str(exc)) from exc
            return ()
        return tuple(_model_to_intent(row) for row in rows)

    def intents_of(self, scope: str) -> tuple[IntentRecord, ...]:
        """Every stored order intent for *scope*, newest first (empty if unwritten).

        Unlike :meth:`load_open`, not restricted to OPEN states: the full audit
        trail (filled / unfilled / rejected included) is what the report shows.
        A missing table reads as empty; a genuine read failure raises.
        """
        scope = _alias_scope(self._database, scope)
        try:
            with self._database.bind_ctx(_MODELS):
                rows = (
                    LiveOrderIntent.select()
                    .where(LiveOrderIntent.scope == scope)
                    .order_by(LiveOrderIntent.updated_at.desc())
                    .execute()
                )
        except peewee.OperationalError as exc:
            if not _is_missing_table(exc):
                raise LedgerReadError(str(exc)) from exc
            return ()
        return tuple(_model_to_intent(row) for row in rows)

    def save(self, record: IntentRecord) -> None:
        """Upsert *record* keyed on its identity columns.

        The identity columns are the primary key, so a crc32 token collision can
        never overwrite a foreign identity's row — two colliding keys simply
        coexist under distinct identities (D7).
        """
        with self._write():
            record = self._record(record)
            LiveOrderIntent.insert(**_intent_fields(record)).on_conflict(
                "REPLACE"
            ).execute()

    def open_attempt(
        self, key: IntentKey, decision_ts: pd.Timestamp | None, now: pd.Timestamp
    ) -> IntentRecord:
        """Mint the next attempt for *key* and persist it PENDING.

        The attempt bumps past any prior stored record, so a legitimate re-send
        (a close re-closed after a partial fill) gets a NEW cOID and IBKR's own
        dedupe cannot swallow it.
        """
        existing = self.load(key)
        attempt = existing.attempt + 1 if existing is not None else 0
        record = IntentRecord(
            key=key,
            state=IntentState.PENDING,
            attempt=attempt,
            order_ref=order_ref(key, attempt),
            order_id=None,
            decision_ts=decision_ts,
        )
        self.save(record)
        return record

    def close(
        self,
        key: IntentKey,
        state: IntentState,
        order_id: str | None,
        now: pd.Timestamp,
    ) -> None:
        """Stamp *state* (and *order_id* when known) on *key*'s record.

        The stored ``order_id`` is only overwritten when a new one is supplied,
        so an UNRESOLVED transition does not erase a known id. Keyed on the
        identity columns (never the token), so a token collision cannot touch a
        foreign row.
        """
        fields: dict[str, object] = {"state": state.value, "updated_at": _ms(now)}
        if order_id is not None:
            fields["order_id"] = order_id
        with self._write():
            key = self._key(key)
            LiveOrderIntent.update(**fields).where(
                self._intent_predicate(key)
            ).execute()

    def _key(self, key: IntentKey) -> IntentKey:
        """*key* re-pointed at its scope's alias-resolved name (migration re-key)."""
        scope = _alias_scope(self._database, key.scope)
        return key if scope == key.scope else replace(key, scope=scope)

    def _record(self, record: IntentRecord) -> IntentRecord:
        """*record* re-pointed at its scope's alias-resolved name."""
        key = self._key(record.key)
        return record if key == record.key else replace(record, key=key)

    @staticmethod
    def _intent_predicate(key: IntentKey):
        """The peewee WHERE identifying one intent row (identity columns)."""
        return (
            (LiveOrderIntent.scope == key.scope)
            & (LiveOrderIntent.symbol == key.symbol)
            & (LiveOrderIntent.action == key.action.value)
            & (LiveOrderIntent.position_id == (key.position_id or ""))
        )

    def prune(self, before: pd.Timestamp) -> int:
        """Delete CLOSED intent rows older than *before*; return the count."""
        with self._write():
            return (
                LiveOrderIntent.delete()
                .where(
                    (LiveOrderIntent.state << _CLOSED_INTENTS)
                    & (LiveOrderIntent.updated_at < _ms(before))
                )
                .execute()
            )


class SqliteLedger(MetadataStore, SimLotStore, BookStore, IntentStore):
    """peewee-backed per-scope book. One database per ledger, bound on init.

    Construction writes NOTHING (no DDL): a ``--dry-run`` that only reads must
    leave the schema untouched. The schema is created lazily on the first
    write; reads of a missing table return an empty book.

    The default path is the LIVE file (:func:`src.db.path.resolve_live_db_path`),
    NOT the candle file: the book is durable state no download can rebuild, so it
    no longer shares a file with bulk research data (D-Q7). The split was a
    one-off verified copy, and the stale source tables left behind in the candle
    file were then dropped by ``data_0002_drop_migrated_live_tables``; a ledger
    pointed at the wrong file reads the book as FLAT and re-opens every position.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = db_path
        self._schema_ready = False
        path = str(resolve_live_db_path(db_path))
        self._database = SqliteDatabase(path)

    @property
    def db_path(self) -> str:
        """The resolved sqlite path this ledger is bound to (read-only)."""
        return str(self._database.database)

    # -- lazy DDL / migration ---------------------------------------------

    def _ready_schema(self) -> None:
        """Run the LIVE migrations exactly once, on the first WRITE only.

        :func:`run_pending` applies every pending migration and its bookkeeping
        row inside ONE ``BEGIN IMMEDIATE`` transaction. That takes the SQLite
        write lock up front, so two processes racing the first migration (the
        overlap the cycle lease guards against) are SERIALIZED: the second blocks
        until the first commits, re-reads the bookkeeping and finds nothing
        pending. There is no check-then-act window, so neither a ``no such table``
        nor a double-re-key race is reachable.

        It is deliberately NOT wrapped in the cycle lease —
        ``ensure_strategy``/``ensure_cash`` write before ``run_cycle`` takes the
        lease, and re-taking it here would deadlock a cycle already holding it.
        What this does NOT do: it does not serialize those later idempotent writes
        against a live cycle; each is its own atomic write, and the lease remains
        the cross-process guard for placement.

        ONLY :data:`LIVE_MIGRATIONS` runs here. The live first write must never
        replay a bulk candle migration before it can place an order.
        """
        if self._schema_ready:
            return
        run_pending(self._database, LIVE_MIGRATIONS)
        self._schema_ready = True

    @contextmanager
    def _write(self) -> Iterator[None]:
        """A write: models bound to this ledger's db, lazy DDL once, then atomic."""
        with self._database.bind_ctx(_MODELS):
            self._ready_schema()
            with self._database.atomic():
                yield

    def cycle_lease(self, scope: str = "") -> AbstractContextManager[None]:
        """Exclusive cross-process lease for ONE scope's cycle on this DB.

        Held for the cycle's duration so a cron overlap or a racing human run
        REFUSES to start rather than both placing off the same pre-order book.
        The lock is PER SCOPE (``<db>.<scope_tag>.cycle.lock``), so two adapters
        — or two configs — run CONCURRENTLY without blocking each other, while
        two cycles on the SAME scope still serialize. The kernel drops the flock
        on process exit, so a crashed run never wedges live trading; there is no
        TTL. Single host / local filesystem only (the caveat ``lease`` documents).
        """
        path = self._db_path if self._db_path is not None else resolve_live_db_path()
        tag = scope_tag(scope) if scope else "global"
        return file_lease(str(path), tag)


def config_hash(config: Mapping[str, object]) -> str:
    """Pure: sha256 of canonical JSON (sorted keys, no whitespace) -> hex.

    An AUDIT key only (which config revision placed what) — never an ownership
    filter, because a parameter edit would then orphan a live lot (§4).
    """
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _cash_delta(execution: Execution) -> float:
    """Signed cash flow of one execution: a SELL credits, a BUY debits (net of fee)."""
    gross = execution.qty * execution.price
    if execution.side is OrderSide.SELL:
        return gross - execution.commission
    return -(gross + execution.commission)


def _sim_fill_records(scope: str, lot: SimLot) -> tuple[ExecutionRecord, ...]:
    """The fills one sim lot implies: its entry, plus its exit when it has one.

    A lot without detail (an ownership-only row) implies nothing — there is no
    size or price to book, and inventing one would put a phantom fill in the
    scope's history. An open long DEBITS cash on the entry and will CREDIT it on
    the exit; a short is the mirror, so the side of each leg follows the lot.
    """
    if not lot.has_detail:
        return ()
    qty = cast("float", lot.qty)
    entry = cast("float", lot.entry_price)
    short = lot.side == "short"
    records = [
        _sim_fill(
            scope,
            f"{lot.position_id}:open",
            lot,
            OrderSide.SELL if short else OrderSide.BUY,
            qty,
            entry,
            lot.entry_commission,
            lot.opened_at,
        )
    ]
    if lot.exit_price is not None:
        records.append(
            _sim_fill(
                scope,
                f"{lot.position_id}:close",
                lot,
                OrderSide.BUY if short else OrderSide.SELL,
                qty,
                lot.exit_price,
                lot.exit_commission,
                lot.closed_at,
            )
        )
    return tuple(records)


def _sim_fill(
    scope: str,
    execution_id: str,
    lot: SimLot,
    side: OrderSide,
    qty: float,
    price: float,
    commission: float | None,
    ts: pd.Timestamp | None,
) -> ExecutionRecord:
    """One sim leg as the same record shape a replayed IBKR fill produces."""
    fee = commission or 0.0
    gross = qty * price
    return ExecutionRecord(
        scope=scope,
        execution_id=execution_id,
        conid=None,
        position_id=lot.position_id,
        symbol=lot.symbol or "",
        side=side.value,
        qty=qty,
        price=price,
        commission=fee,
        cash_delta=(gross - fee) if side is OrderSide.SELL else -(gross + fee),
        ts=ts,
    )


#: Public name for the cash-flow helper (used by the IBKR portfolio source).
execution_cash_delta = _cash_delta


def _intent_fields(record: IntentRecord) -> dict[str, object]:
    now = pd.Timestamp.now()
    return {
        "scope": record.key.scope,
        "token": record.key.token(),
        "symbol": record.key.symbol,
        "action": record.key.action.value,
        #: ``''`` (never SQL NULL) is the ``position_id`` stored for an open, so
        #: the identity primary key stays total (a NULL would defeat its
        #: uniqueness enforcement on SQLite).
        "position_id": record.key.position_id or "",
        "state": record.state.value,
        "attempt": record.attempt,
        "order_ref": record.order_ref,
        "order_id": record.order_id,
        "decision_ts": (
            _ms(record.decision_ts) if record.decision_ts is not None else None
        ),
        "tif": record.tif,
        "stuck_cycles": record.stuck_cycles,
        "updated_at": _ms(now),
    }


def _audit_of(row: LiveStrategy) -> StrategyAudit:
    return StrategyAudit(
        strategy_id=cast("str", row.strategy_id),
        scope=cast("str", row.scope),
        name=cast("str", row.name),
        mode=cast("str", row.mode),
        created_at=_ts(cast("int | None", row.created_at)),
        last_cycle_at=_ts(cast("int | None", row.last_cycle_at)),
    )


def _model_to_intent(row: LiveOrderIntent) -> IntentRecord:
    return IntentRecord(
        key=IntentKey(
            scope=cast("str", row.scope),
            symbol=cast("str", row.symbol),
            action=ActionType(cast("str", row.action)),
            #: ``''`` round-trips to ``None``: an open's row stores ``''`` but the
            #: in-memory key keeps ``position_id=None`` for equality with callers.
            position_id=cast("str | None", row.position_id) or None,
        ),
        state=IntentState(cast("str", row.state)),
        attempt=int(cast("int", row.attempt)),
        order_ref=cast("str", row.order_ref),
        order_id=cast("str | None", row.order_id),
        decision_ts=_ts(cast("int | None", row.decision_ts)),
        tif=cast("str", row.tif),
        stuck_cycles=int(cast("int", row.stuck_cycles)),
    )


def _book_fields(scope: str, row: BookRow) -> dict[str, object]:
    """One reconciled ``BookRow`` as ``live_position`` fields (the executions role).

    ``scope`` comes from the caller, not ``row``: ``reconcile`` stamps the row's
    ``scope`` but the ledger's alias-resolved scope is the authority (a legacy
    scope re-keyed at migration must be written under its new name).
    """
    return {
        "scope": scope,
        "position_id": str(row.conid),
        "symbol": row.symbol,
        "side": row.side,
        "qty": row.qty,
        "entry_price": row.entry_price,
        "opened_at": _ms(row.opened_at) if row.opened_at is not None else None,
        "closed_at": _ms(row.closed_at) if row.closed_at is not None else None,
        "stop_loss": row.stop_loss,
        "take_profit": row.take_profit,
        "tag": row.tag,
        "order_ref": row.order_ref,
        "source": SOURCE_EXECUTIONS,
    }


def _model_to_execution(
    row: LiveExecution, symbols: Mapping[str, str]
) -> ExecutionRecord:
    """Project a stored execution, resolving its symbol from *symbols* (lot id)."""
    position_id = cast("str", row.position_id)
    return ExecutionRecord(
        scope=cast("str", row.scope),
        execution_id=cast("str", row.execution_id),
        conid=_conid_of(position_id),
        position_id=position_id,
        symbol=symbols.get(position_id, ""),
        side=cast("str", row.side),
        qty=float(cast("float", row.qty)),
        price=float(cast("float", row.price)),
        commission=float(cast("float", row.commission)),
        cash_delta=float(cast("float", row.cash_delta)),
        ts=_ts(cast("int | None", row.ts)),
    )


def _conid_of(position_id: str) -> int | None:
    """The conid a lot id names, or ``None`` when it is not an IBKR conid.

    ``live_position.position_id`` is TEXT and carries either ``str(conid)`` (the
    IBKR executions role) or a sim lot id, so a non-numeric id is a legitimate
    sim row rather than an error.
    """
    return int(position_id) if position_id.isdigit() else None


def _model_to_book(row: LivePosition) -> BookRow:
    conid = _conid_of(cast("str", row.position_id))
    if conid is None:
        raise LedgerReadError(
            f"live_position row {cast('str', row.position_id)!r} in scope "
            f"{cast('str', row.scope)!r} is not an IBKR conid; the executions "
            "book cannot read it"
        )
    return BookRow(
        scope=cast("str", row.scope),
        conid=conid,
        symbol=cast("str", row.symbol),
        side=cast("str", row.side),
        qty=float(cast("float", row.qty)),
        entry_price=float(cast("float", row.entry_price)),
        opened_at=_ts(cast("int | None", row.opened_at)),
        closed_at=_ts(cast("int | None", row.closed_at)),
        stop_loss=cast("float | None", row.stop_loss),
        take_profit=cast("float | None", row.take_profit),
        tag=cast("str", row.tag),
        order_ref=cast("str", row.order_ref),
    )
