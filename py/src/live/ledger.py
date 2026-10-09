"""Per-scope sqlite book — the durable, per-strategy position store (plan §3).

The 7-day ``/iserver/account/trades`` window is a *confirmation channel*: it
advances the book. The book itself lives here, ONE table (``live_position``), one
row per ``(scope, position_id)``, plus one row per applied ``execution_id`` so
re-applying the window is a no-op. The table carries ONE book ROLE: ``source='executions'``
rows are the immutable fold of our own fills (the IBKR reconcile book, keyed
``str(conid)``). The sim ACCOUNT book lives OUTSIDE sqlite — a JSON file
(``src.live.adapters.sim.store``) mimicking IBKR responses — so an operator can
hand-edit it to create a divergence; our OWN record stays this sqlite fill fold.
Live-only: the backtest persists nothing. Tables live in the LIVE file
(``src.db.path.resolve_live_db_path``), whose schema is owned by ``src.db`` and
applied by ``ibkr db migrate``. The ledger writes NO DDL: it never creates or
migrates tables, so a ledger on an unmigrated file fails loud (``no such
table``) rather than building a half-schema under a live order.

``strategy_id`` (the config hash) survives on ``live_strategy`` as an AUDIT
column only — never an ownership filter. Ownership is the ``scope`` (the cOID
prefix, §4). ``live_cash`` carries each scope's ``initial_capital``; per-scope
cash is derived from the scope's own executions, never read from the account
summary (N strategies share one account).

Storage is peewee ORM over the same SQLite file. The model DDL mirrors the
previous raw ``CREATE TABLE`` SQL column-for-column, so an existing phase-3.5
book rows read back unchanged. The ledger only ever reads and writes rows; the
schema (and its lossless migrations) belong to ``src.db``.

``SqliteLedger`` is the seam implementing the live Protocols; its per-domain
methods live in their own mixins (``MetadataStore``, ``BookStore``,
``IntentStore``) so each class stays small. Shared plumbing (the template
binding, the base model, the epoch codec) is in ``ledger_base``; the lossless
migrations are in ``ledger_migration``. Every public method keeps its name and
signature.

The value shapes (``ExecutionRecord``/``StrategyAudit``) live in
:mod:`src.live.types` and the money/decision code (the write planner
``plan_result_writes`` and the execution projections) lives in
:mod:`src.live.pure_plan`; this module re-exports them so every existing import
path keeps working. The ledger itself only reads rows and applies planned writes
in ONE transaction per cycle.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import TypeVar, cast

import pandas as pd
import peewee
from peewee import SqliteDatabase, fn

from src.bt.state import ActionType
from src.db.path import resolve_live_db_path
from src.exec.refs import scope_tag

from src.live.adapters.ibkr.trades import (
    BookRow,
    Execution,
    StrategyBook,
    is_ours,
)
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
    SOURCE_EXECUTIONS,
    LiveCash,
    LiveExecution,
    LiveOrderIntent,
    LivePosition,
    LiveStrategy,
)
from src.live.pure import OrderResult
from src.live.pure_plan import (
    execution_cash_delta,
    next_attempt,
    plan_result_writes,
)
from src.live.types import ExecutionRecord, StrategyAudit, synthetic_conid

__all__ = [
    "ExecutionRecord",
    "SqliteLedger",
    "StrategyAudit",
    "execution_cash_delta",
]

logger = logging.getLogger(__name__)

#: The ``bind_ctx`` group: every live model, rebindable per ledger instance.
#: Rebound here (not globally) because peewee binds at CLASS level and two ledgers
#: on two paths share these classes — see :mod:`src.live.models`.
_MODELS = LIVE_MODELS

_T = TypeVar("_T")

#: The terminal (closed) intent states ``prune`` is allowed to delete.
_CLOSED_INTENTS = [
    IntentState.FILLED.value,
    IntentState.UNFILLED.value,
    IntentState.REJECTED.value,
]


def _read(db: SqliteDatabase, query: Callable[[], _T], default: _T) -> _T:
    """Run *query* under the live model binding; an absent table yields *default*.

    An absent table is the never-written (or dry-run) store, whose reads are
    empty rather than an error. Any OTHER ``OperationalError`` — a lock, a
    corrupt file, a shape drift — raises :class:`LedgerReadError`, so an
    unreadable book can never read as a flat one.
    """
    try:
        with db.bind_ctx(_MODELS):
            return query()
    except peewee.OperationalError as exc:
        if not _is_missing_table(exc):
            raise LedgerReadError(str(exc)) from exc
        return default


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
        for model in (
            LiveStrategy,
            LiveCash,
            LivePosition,
            LiveExecution,
            LiveOrderIntent,
        ):
            rows = _read(
                self._database,
                lambda m=model: m.select(m.scope).distinct().execute(),
                (),
            )
            found.update(str(row.scope) for row in rows)
        return tuple(sorted(found))

    def ensure_strategy(self, strategy_id: str, scope: str, name: str) -> None:
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
                created_at=_ms(pd.Timestamp.now()),
            ).on_conflict("IGNORE").execute()

    def ensure_cash(self, scope: str, initial_capital: float) -> None:
        """Record the scope's ``initial_capital`` once; a later cycle never resets it."""
        with self._write():
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
        rows = _read(
            self._database,
            lambda: (
                LiveStrategy.select()
                .where(LiveStrategy.scope == scope)
                .order_by(LiveStrategy.created_at, LiveStrategy.strategy_id)
                .execute()
            ),
            (),
        )
        return tuple(_audit_of(row) for row in rows)


class BookStore(_SqliteOps):
    """Ledger mixin: the IBKR reconcile book, its executions and its cash seed.

    This mixin owns the ``source='executions'`` ROLE of ``live_position`` — the
    fold of our own fills that ``trades.reconcile`` advances. The sim ACCOUNT
    book lives OUTSIDE sqlite (the JSON ``src.live.adapters.sim.store``), so
    this is the single durable record of our own fills.
    """

    def load_book(self, scope: str) -> StrategyBook:
        """The durable rows + applied execution ids for *scope* (empty if unaware)."""

        def query() -> tuple[tuple[LivePosition, ...], frozenset[str]]:
            """The execution-role rows plus the ids of the executions already applied."""
            return (
                tuple(
                    LivePosition.select()
                    .where(
                        (LivePosition.scope == scope)
                        & (LivePosition.source == SOURCE_EXECUTIONS)
                    )
                    .order_by(LivePosition.position_id)
                    .execute()
                ),
                frozenset(
                    e.execution_id
                    for e in LiveExecution.select(LiveExecution.execution_id).where(
                        LiveExecution.scope == scope
                    )
                ),
            )

        rows, applied = _read(self._database, query, ((), frozenset()))
        return StrategyBook(
            rows=tuple(_model_to_book(row) for row in rows),
            applied=applied,
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
                    conid=execution.conid,
                    side=execution.side.value,
                    qty=execution.qty,
                    price=execution.price,
                    commission=execution.commission,
                    cash_delta=execution_cash_delta(execution),
                    ts=_ms(execution.ts),
                ).on_conflict("IGNORE").execute()

    def executions_of(self, scope: str) -> tuple[ExecutionRecord, ...]:
        """Stored fill history for *scope*, oldest first (empty if unwritten).

        One :class:`ExecutionRecord` per applied fill, `ts` ascending, with
        `symbol` resolved from the lot's book row (`""` when none remains). A
        missing table (a never-written scope) reads as empty; a genuine read
        failure raises :class:`LedgerReadError`.
        """

        def query() -> tuple[dict[str, str], list[LiveExecution]]:
            """The scope's symbol-per-position_id map and its execution rows."""
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
            return symbols, list(rows)

        symbols, rows = _read(self._database, query, ({}, []))
        return tuple(_model_to_execution(row, symbols) for row in rows)

    def owned_ids(self, scope: str) -> frozenset[str]:
        """The lot ids this scope currently owns (from the fill fold, empty if unwritten).

        The ids of ``book_from_executions(executions_of(scope))`` — every OPEN lot
        our fills account for. This scopes a close to lots the strategy actually
        opened, so a hand-edited JSON account book never makes a close target a
        lot we do not own.
        """
        #: Deferred import: ``divergence`` imports ``ExecutionRecord`` from this
        #: module, so a module-level import here would close a cycle.
        from src.live.divergence import book_from_executions

        book = book_from_executions(self.executions_of(scope))
        return frozenset(
            position.position_id
            for positions in book.positions.values()
            for position in positions
            if position.position_id
        )

    def record_results(
        self,
        scope: str,
        results: tuple[OrderResult, ...],
        now: pd.Timestamp,
    ) -> None:
        """Record a cycle's placement results into the book — the ONE write point.

        The engine hands over what the adapter RETURNED (adapters hold no book),
        so a failed or rejected result records nothing, and each ``ok`` result
        with a fill and a ``position_id`` mints its fill row(s). WHICH rows to
        write is decided by the pure :func:`src.live.pure_plan.plan_result_writes`;
        this method only applies them, all in ONE transaction, so a cycle's
        results land together or not at all (never a half-recorded cycle).
        """
        writes = plan_result_writes(scope, results, now)
        if not writes:
            return
        with self._write():
            for write in writes:
                _insert_execution(write.record)

    def cash_of(self, scope: str, default_initial: float = 0.0) -> float:
        """Per-scope cash: ``initial_capital`` advanced by the scope's own fills.

        The stored ``initial_capital`` (first cycle's config value) wins; a scope
        with no stored row falls back to *default_initial*. Never reads the
        account summary: N strategies share one account's cash. Cash is the fold
        of ``live_execution`` alone — the JSON account book's summary is the
        operator-editable side, never the authority here.
        """

        def query() -> tuple[float, float]:
            """The stored (or default) initial capital and the fills' summed cash."""
            cash = LiveCash.get_or_none(LiveCash.scope == scope)
            initial = (
                float(cash.initial_capital) if cash is not None else default_initial
            )
            sunk = (
                LiveExecution.select(fn.COALESCE(fn.SUM(LiveExecution.cash_delta), 0.0))
                .where(LiveExecution.scope == scope)
                .scalar()
            )
            return initial, float(sunk)

        initial, sunk = _read(self._database, query, (default_initial, 0.0))
        return initial + sunk

    def initial_capital_of(self, scope: str) -> float:
        cash = _read(
            self._database,
            lambda: LiveCash.get_or_none(LiveCash.scope == scope),
            None,
        )
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
        ``executions`` ROLE counts: the id space is a conid's. A missing table
        reads as a flat (0.0) book — the dry-run case; a genuine read failure
        still raises (:class:`LedgerReadError`), never a silent zero.
        """
        rows = _read(
            self._database,
            lambda: (
                LivePosition.select(LivePosition.side, LivePosition.qty)
                .where(
                    (LivePosition.position_id == str(conid))
                    & (LivePosition.source == SOURCE_EXECUTIONS)
                    & LivePosition.closed_at.is_null()
                )
                .execute()
            ),
            (),
        )
        return sum(
            float(row.qty) if row.side == "long" else -float(row.qty) for row in rows
        )


def _insert_execution(record: ExecutionRecord) -> None:
    """Persist one derived fill row and fold its ``source='executions'`` lot row.

    The caller holds the transaction. The ``live_execution`` row is keyed
    ``(scope, execution_id)`` so a re-run is a no-op; the matching
    ``live_position`` row (``source='executions'``) carries the lot's symbol/side/
    qty/entry so ``executions_of``'s symbol map and ``book_from_executions`` can
    rebuild the book. The lot row FOLDS every execution of the lot (each BUY leg
    adds, each SELL leg subtracts — the sim account book's ``_reduce_position``
    arithmetic), so a ``:close`` leaves the REMAINING signed qty and side on the
    row, never the close leg's own qty, and ``closed_at`` is stamped only when the
    net reaches zero. The row is never dropped: the fold must stay replayable, and
    recomputing from the IGNORE-once execution rows keeps a re-plan a no-op.
    """
    ts_ms = 0 if record.ts is None else _ms(record.ts)
    LiveExecution.insert(
        scope=record.scope,
        execution_id=record.execution_id,
        position_id=record.position_id,
        conid=_fill_conid(record),
        side=record.side,
        qty=record.qty,
        price=record.price,
        commission=record.commission,
        cash_delta=record.cash_delta,
        ts=ts_ms,
    ).on_conflict("IGNORE").execute()
    remaining = _folded_qty(record.scope, record.position_id)
    existing = LivePosition.get_or_none(
        (LivePosition.scope == record.scope)
        & (LivePosition.position_id == record.position_id)
    )
    fields = _lot_fields(record, existing, remaining, ts_ms)
    if existing is None:
        LivePosition.insert(**fields).execute()
    else:
        LivePosition.update(**fields).where(
            (LivePosition.scope == record.scope)
            & (LivePosition.position_id == record.position_id)
        ).execute()


def _lot_fields(
    record: ExecutionRecord,
    existing: LivePosition | None,
    remaining: float,
    ts_ms: int,
) -> dict[str, object]:
    """The folded ``live_position`` row for a lot after *record* folds in.

    ``remaining`` is the lot's signed net across ALL its fills, so the row's
    side/qty mirror the sim account book's ``_reduce_position`` (a partial close
    leaves the remainder, a zero net keeps the lot's REAL side and stamps
    ``closed_at`` only now — a prior stamp survives a replay untouched).
    """
    prior_closed = existing.closed_at if existing is not None else None
    closed_at = (
        prior_closed
        if prior_closed is not None
        else (ts_ms if remaining == 0.0 else None)
    )
    side = (
        "long"
        if remaining > 0
        else "short"
        if remaining < 0
        else cast("str", existing.side)
        if existing is not None
        else "long"
    )
    return {
        "scope": record.scope,
        "position_id": record.position_id,
        "symbol": record.symbol,
        "side": side,
        "qty": abs(remaining),
        "entry_price": (
            float(cast("float", existing.entry_price))
            if existing is not None
            else record.price
        ),
        "stop_loss": None,
        "take_profit": None,
        "tag": "",
        "order_ref": "",
        "opened_at": existing.opened_at if existing is not None else ts_ms,
        "closed_at": closed_at,
        "source": SOURCE_EXECUTIONS,
    }


def _folded_qty(scope: str, position_id: str) -> float:
    """The lot's signed size as the sum of ALL its fills: a BUY adds, a SELL subtracts."""
    return sum(
        float(leg.qty) if leg.side == "BUY" else -float(leg.qty)
        for leg in LiveExecution.select(LiveExecution.side, LiveExecution.qty).where(
            (LiveExecution.scope == scope) & (LiveExecution.position_id == position_id)
        )
    )


class IntentStore(_SqliteOps):
    """Ledger mixin: the durable owner of OPEN order state (``PendingIntents``).

    Reads tolerate a missing table (empty), like ``load_book``. A missing table on
    a WRITE is not tolerated: the ledger writes no DDL (``ibkr db migrate`` owns
    the schema), so an unmigrated file fails loud. The identity columns
    (scope/symbol/action/position_id) are the primary
    key, so two intents whose crc32 tokens collide stay separately addressable by
    IDENTITY. NOTE: that is a ROW-level guarantee only — the broker still
    attributes a working order by the token-keyed ref prefix, so two colliding
    keys share a prefix there.
    """

    def load(self, key: IntentKey) -> IntentRecord | None:
        """The durable record for *key*, or ``None`` when unwritten."""
        row = _read(
            self._database,
            lambda: LiveOrderIntent.get_or_none(self._intent_predicate(key)),
            None,
        )
        return None if row is None else _model_to_intent(row)

    def load_open(self, scope: str) -> tuple[IntentRecord, ...]:
        """Every OPEN record for *scope* (empty if unwritten)."""
        states = [s.value for s in OPEN_STATES]
        rows = _read(
            self._database,
            lambda: (
                LiveOrderIntent.select()
                .where(
                    (LiveOrderIntent.scope == scope) & (LiveOrderIntent.state << states)
                )
                .execute()
            ),
            (),
        )
        return tuple(_model_to_intent(row) for row in rows)

    def intents_of(self, scope: str) -> tuple[IntentRecord, ...]:
        """Every stored order intent for *scope*, newest first (empty if unwritten).

        Unlike :meth:`load_open`, not restricted to OPEN states: the full audit
        trail (filled / unfilled / rejected included) is what the report shows.
        A missing table reads as empty; a genuine read failure raises.
        """
        rows = _read(
            self._database,
            lambda: (
                LiveOrderIntent.select()
                .where(LiveOrderIntent.scope == scope)
                .order_by(LiveOrderIntent.updated_at.desc())
                .execute()
            ),
            (),
        )
        return tuple(_model_to_intent(row) for row in rows)

    def save(self, record: IntentRecord) -> None:
        """Upsert *record* keyed on its identity columns.

        The identity columns are the primary key, so a crc32 token collision can
        never overwrite a foreign identity's row — two colliding keys simply
        coexist under distinct identities (D7).
        """
        with self._write():
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
        attempt = next_attempt(existing)
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
            LiveOrderIntent.update(**fields).where(
                self._intent_predicate(key)
            ).execute()

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


class SqliteLedger(MetadataStore, BookStore, IntentStore):
    """peewee-backed per-scope book. One database per ledger, bound on init.

    The ledger writes NO DDL, ever: construction writes nothing (a ``--dry-run``
    that only reads must leave the file untouched) and neither does a write.
    Schema is ``src.db``'s job — run ``ibkr db migrate`` before the first cycle.
    An unmigrated file therefore fails LOUD on the first write; a read of a
    genuinely missing table still returns an empty book (see
    :class:`LedgerReadError` for the drifted-table case).

    The default path is the LIVE file (:func:`src.db.path.resolve_live_db_path`),
    NOT the candle file: the book is durable state no download can rebuild, so it
    no longer shares a file with bulk research data (D-Q7). The split was a
    one-off verified copy, and the stale source tables left behind in the candle
    file were then dropped by ``data_0002_drop_migrated_live_tables``; a ledger
    pointed at the wrong file reads the book as FLAT and re-opens every position.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = db_path
        path = str(resolve_live_db_path(db_path))
        self._database = SqliteDatabase(path)

    @property
    def db_path(self) -> str:
        """The resolved sqlite path this ledger is bound to (read-only)."""
        return str(self._database.database)

    @contextmanager
    def _write(self) -> Iterator[None]:
        """A write: models bound to this ledger's db, then atomic."""
        with self._database.bind_ctx(_MODELS):
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
    ``scope`` but the ledger's scope is the authority (a caller may be addressing a
    book it read under another name).
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


def _fill_conid(record: ExecutionRecord) -> int:
    """The ``conid`` value a fill row needs (provenance-only, but NOT NULL).

    An IBKR fill's lot id IS ``str(conid)``, so it parses back; a sim fill's lot
    id is a minted ``SYM_<ts>_<seq>`` handle with no conid, so it mints a stable
    synthetic one from the symbol. The migrated column is NOT NULL and nothing
    reads it back, but ``INSERT OR IGNORE`` would silently DROP a row that omits
    it — so it must always be supplied.
    """
    if record.position_id.isdigit():
        return int(record.position_id)
    return synthetic_conid(record.symbol)


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
