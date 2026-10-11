"""Per-scope sqlite book — the durable, per-strategy position store.

Our own fills are the ONE durable record. ``live_execution`` is the append-only
fold source, keyed ``(scope, execution_id)`` so re-applying a window is a no-op;
``live_strategy`` carries each scope's ownership, audit and ``initial_capital``;
``live_order_intent`` is the durable owner of OPEN order state. Per-scope cash is
derived from the scope's own executions (never read from the account summary — N
strategies share one account).

The simulator's ACCOUNT book lives OUTSIDE sqlite (a JSON file,
``src.live.adapters.sim.store``) so an operator can hand-edit it to create a
divergence; our own record stays this sqlite fill fold. Live-only: the backtest
persists nothing. Tables live in the LIVE file
(``src.db.path.resolve_live_db_path``), whose schema is owned by ``src.db`` and
applied by ``ibkr db migrate``. The ledger writes NO DDL: it never creates or
migrates tables, so a ledger on an unmigrated file fails loud (``no such table``)
rather than building a half-schema under a live order.

``strategy_id`` (the config hash) survives on ``live_strategy`` as an AUDIT
column only — never an ownership filter. Ownership is the ``scope`` (the cOID
prefix). ``SqliteLedger`` is the seam implementing the live Protocols; its
per-domain methods live in their own mixins (``MetadataStore``, ``BookStore``,
``IntentStore``) so each class stays small. Shared plumbing (the template
binding, the base model, the epoch codec) is in ``ledger_base``.

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

from src.live.types import Execution, StrategyBook
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
    LiveExecution,
    LiveOrderIntent,
    LiveStrategy,
)
from src.live.pure import OrderResult
from src.live.pure_plan import (
    commission_of,
    execution_cash_delta,
    next_attempt,
    plan_result_writes,
)
from src.live.types import ExecutionRecord, StrategyAudit

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
    """Ledger mixin: the scope's ownership, audit and cash seed (``live_strategy``)."""

    def scopes_of_store(self) -> tuple[str, ...]:
        """Every scope the store knows about, sorted (empty when unwritten).

        The union of the tables that carry a ``scope`` column, so a scope whose
        strategy row was never written but which holds fills or intents is still
        reported. A genuinely missing table contributes nothing; any other read
        failure raises ``LedgerReadError``.
        """
        found: set[str] = set()
        for model in (LiveStrategy, LiveExecution, LiveOrderIntent):
            rows = _read(
                self._database,
                lambda m=model: m.select(m.scope).distinct().execute(),
                (),
            )
            found.update(str(row.scope) for row in rows)
        return tuple(sorted(found))

    def ensure_strategy(
        self, strategy_id: str, scope: str, name: str, initial_capital: float
    ) -> None:
        """Upsert the scope's row: first write seeds, later writes only audit.

        Ownership is the ``scope`` (one book, one cash seed, one cOID prefix);
        ``strategy_id`` is the config HASH and changes on ANY parameter edit, so
        it is stored as the latest revision rather than a key. ``initial_capital``
        and ``created_at`` are set on the FIRST write and never reset — a later
        cycle with a different config value must not silently re-seed the book.

        A DIFFERENT config file reusing a scope shares its book, cash seed and cOID
        prefix (identical cOIDs within one second). The two are indistinguishable
        here (no path is stored, and an edit changes the hash), so this WARNS
        rather than refuses — a hard refuse would wedge every config edit. Give
        the other config its own ``scope``.
        """
        now = _ms(pd.Timestamp.now())
        with self._write():
            _warn_rival_configs(scope, strategy_id)
            existing = LiveStrategy.get_or_none(LiveStrategy.scope == scope)
            if existing is None:
                LiveStrategy.insert(
                    scope=scope,
                    strategy_id=strategy_id,
                    name=name,
                    initial_capital=initial_capital,
                    created_at=now,
                    last_cycle_at=None,
                    updated_at=now,
                ).execute()
                return
            LiveStrategy.update(
                strategy_id=strategy_id, name=name, updated_at=now
            ).where(LiveStrategy.scope == scope).execute()

    def touch_cycle(self, scope: str, at: pd.Timestamp) -> None:
        """Stamp the scope's ``last_cycle_at`` (the audit clock)."""
        with self._write():
            LiveStrategy.update(last_cycle_at=_ms(at)).where(
                LiveStrategy.scope == scope
            ).execute()

    def strategies_of(self, scope: str) -> tuple[StrategyAudit, ...]:
        """The scope's single audit row (empty if the table is absent).

        Read-only: this is the ``ibkr live pf`` view of which config revision last
        wrote the scope. A missing table (the never-written case) reads as empty,
        like the book reads; any OTHER ``OperationalError`` is a genuine read
        failure and raises :class:`LedgerReadError` rather than reporting nothing.
        """
        rows = _read(
            self._database,
            lambda: (
                LiveStrategy.select()
                .where(LiveStrategy.scope == scope)
                .order_by(LiveStrategy.created_at)
                .execute()
            ),
            (),
        )
        return tuple(_audit_of(row) for row in rows)


class BookStore(_SqliteOps):
    """Ledger mixin: our fill-derived book, its executions and its cash seed.

    Our book is the fold of the scope's stored ``live_execution`` rows. The sim
    ACCOUNT book lives OUTSIDE sqlite (the JSON ``src.live.adapters.sim.store``),
    so this is the single durable record of our own fills.
    """

    def load_book(self, scope: str) -> StrategyBook:
        """The durable rows + applied execution ids for *scope* (empty if unaware).

        The rows are DERIVED from the executions (``book_rows_from_executions``),
        so a round trip leaves one closed row and the lot's identity is the fill
        fold's — there is no separate position table to drift.
        """
        #: Deferred import: ``divergence`` imports ``ExecutionRecord`` from this
        #: module's ``types``, so a module-level import here would close a cycle.
        from src.live.divergence import book_rows_from_executions

        executions = self.executions_of(scope)
        return StrategyBook(
            rows=book_rows_from_executions(executions),
            applied=frozenset(e.execution_id for e in executions),
        )

    def save_book(
        self,
        scope: str,
        book: StrategyBook,
        executions: tuple[Execution, ...],
        initial_capital: float,
    ) -> None:
        """Persist the newly applied executions and upsert the cash seed.

        Executions are inserted ``OR IGNORE`` keyed ``(scope, execution_id)``, so
        the cash fold accumulates each fill exactly once even across re-runs.
        ``book`` is read for its ``applied`` watermark only: the position rows are
        derived from the stored fills, never written here.
        """
        with self._write():
            LiveStrategy.insert(
                scope=scope,
                strategy_id="",
                name="",
                initial_capital=initial_capital,
                created_at=_ms(pd.Timestamp.now()),
                last_cycle_at=None,
                updated_at=_ms(pd.Timestamp.now()),
            ).on_conflict("IGNORE").execute()
            for execution in executions:
                if execution.execution_id not in book.applied:
                    continue
                LiveExecution.insert(
                    scope=scope,
                    execution_id=execution.execution_id,
                    position_id=str(execution.conid),
                    symbol=execution.symbol,
                    side=execution.side.value,
                    qty=execution.qty,
                    price=execution.price,
                    cash_delta=execution_cash_delta(execution),
                    ts=_ms(execution.ts),
                ).on_conflict("IGNORE").execute()

    def executions_of(self, scope: str) -> tuple[ExecutionRecord, ...]:
        """Stored fill history for *scope*, oldest first (empty if unwritten).

        One :class:`ExecutionRecord` per applied fill, ``ts`` ascending, with
        ``symbol`` read from its OWN column. A missing table (a never-written
        scope) reads as empty; a genuine read failure raises ``LedgerReadError``.
        """
        rows = _read(
            self._database,
            lambda: (
                LiveExecution.select()
                .where(LiveExecution.scope == scope)
                .order_by(LiveExecution.ts)
                .execute()
            ),
            (),
        )
        return tuple(_model_to_execution(row) for row in rows)

    def owned_ids(self, scope: str) -> frozenset[str]:
        """The lot ids this scope currently owns (from the fill fold, empty if unwritten).

        The ids of the OPEN lots our fills account for. This scopes a close to lots
        the strategy actually opened, so a hand-edited JSON account book never
        makes a close target a lot we do not own.
        """
        #: Deferred import: see :meth:`load_book` for the cycle reason.
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


class CashStore(_SqliteOps):
    """Ledger mixin: per-scope cash, summed from the scope's OWN fills.

    N strategies share one broker account, so the account summary is never the
    authority here: cash is the scope's stored ``initial_capital`` advanced by
    the ``cash_delta`` of its own ``live_execution`` rows.
    """

    def cash_of(self, scope: str, default_initial: float = 0.0) -> float:
        """Per-scope cash: ``initial_capital`` advanced by the scope's own fills.

        The stored ``initial_capital`` (first cycle's config value) wins; a scope
        with no stored row falls back to *default_initial*. Never reads the
        account summary: N strategies share one account's cash.
        """

        def query() -> tuple[float, float]:
            """The stored (or default) initial capital and the fills' summed cash."""
            row = LiveStrategy.get_or_none(LiveStrategy.scope == scope)
            initial = float(row.initial_capital) if row is not None else default_initial
            sunk = (
                LiveExecution.select(fn.COALESCE(fn.SUM(LiveExecution.cash_delta), 0.0))
                .where(LiveExecution.scope == scope)
                .scalar()
            )
            return initial, float(sunk)

        initial, sunk = _read(self._database, query, (default_initial, 0.0))
        return initial + sunk

    def initial_capital_of(self, scope: str) -> float:
        """The scope's stored seed, or ``0.0`` when it has never written."""
        row = _read(
            self._database,
            lambda: LiveStrategy.get_or_none(LiveStrategy.scope == scope),
            None,
        )
        return float(row.initial_capital) if row is not None else 0.0

    def net_exposure(self, conid: int) -> float:
        """Signed net quantity the book holds on *conid*, summed over ALL scopes.

        The executions fold is keyed by ``str(conid)``, so the lot's conid IS its
        ``position_id`` and the fold here is arithmetic over our own fills: a BUY
        adds, a SELL subtracts, and a lot nets out at zero. A missing table reads
        as a flat (0.0) book — the dry-run case; a genuine read failure still
        raises (:class:`LedgerReadError`), never a silent zero.
        """
        rows = _read(
            self._database,
            lambda: (
                LiveExecution.select(LiveExecution.side, LiveExecution.qty)
                .where(LiveExecution.position_id == str(conid))
                .execute()
            ),
            (),
        )
        return sum(_signed(row.side, row.qty) for row in rows)


def _warn_rival_configs(scope: str, strategy_id: str) -> None:
    """Warn when another config hash already owns *scope* (a shared book).

    Ownership is the scope, so a differing hash on the same scope is NORMAL (a
    tuned config). It is a real danger only when a DIFFERENT config file reuses a
    scope, which is indistinguishable here (no path is stored), so this warns
    rather than refusing — a hard refuse would wedge every config edit.
    """
    others = (
        LiveStrategy.select(LiveStrategy.strategy_id)
        .where(
            (LiveStrategy.scope == scope) & (LiveStrategy.strategy_id != strategy_id)
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


def _signed(side: object, qty: object) -> float:
    """Signed fill size: a BUY adds, a SELL subtracts; qty is stored absolute."""
    size = float(cast("float", qty))
    return size if str(side).upper() == "BUY" else -size


def _insert_execution(record: ExecutionRecord) -> None:
    """Persist one derived fill row (the caller holds the transaction).

    Keyed ``(scope, execution_id)`` so a re-run is a no-op. Unlike the old fold,
    nothing else is written: the position rows are DERIVED from these fills by
    ``book_rows_from_executions``, so there is no second copy to keep in step.
    """
    ts_ms = 0 if record.ts is None else _ms(record.ts)
    LiveExecution.insert(
        scope=record.scope,
        execution_id=record.execution_id,
        position_id=record.position_id,
        symbol=record.symbol,
        side=record.side,
        qty=record.qty,
        price=record.price,
        cash_delta=record.cash_delta,
        ts=ts_ms,
    ).on_conflict("IGNORE").execute()


class IntentStore(_SqliteOps):
    """Ledger mixin: the durable owner of OPEN order state (``live_order_intent``).

    Reads tolerate a missing table (empty), like ``load_book``. A missing table on
    a WRITE is not tolerated: the ledger writes no DDL (``ibkr db migrate`` owns
    the schema), so an unmigrated file fails loud. The identity columns
    (scope/symbol/action/position_id) are the primary key, so two intents whose
    crc32 tokens collide stay separately addressable by IDENTITY.
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
        coexist under distinct identities.
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


class SqliteLedger(MetadataStore, BookStore, CashStore, IntentStore):
    """peewee-backed per-scope book. One database per ledger, bound on init.

    The ledger writes NO DDL, ever: construction writes nothing (a ``--dry-run``
    that only reads must leave the file untouched) and neither does a write.
    Schema is ``src.db``'s job — run ``ibkr db migrate`` before the first cycle.
    An unmigrated file therefore fails LOUD on the first write; a read of a
    genuinely missing table still returns an empty book (see
    :class:`LedgerReadError` for the drifted-table case).

    The default path is the LIVE file (:func:`src.db.path.resolve_live_db_path`),
    NOT the candle file: the book is durable state no download can rebuild, so it
    does not share a file with bulk research data. A ledger pointed at the wrong
    file reads the book as FLAT and re-opens every position.
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
    filter, because a parameter edit would then orphan a live lot.
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


def _model_to_execution(row: LiveExecution) -> ExecutionRecord:
    """Project a stored execution; ``symbol`` comes from its OWN column.

    The stored row carries no ``commission`` and no ``conid``: the conid is
    derived from a numeric lot id (``None`` for a sim lot) and the fee is
    recovered from the signed ``cash_delta`` via
    :func:`~src.live.pure_plan.commission_of`, the inverse of the cash rule.
    """
    position_id = cast("str", row.position_id)
    side = cast("str", row.side)
    qty = float(cast("float", row.qty))
    price = float(cast("float", row.price))
    cash_delta = float(cast("float", row.cash_delta))
    return ExecutionRecord(
        scope=cast("str", row.scope),
        execution_id=cast("str", row.execution_id),
        conid=_conid_of(position_id),
        position_id=position_id,
        symbol=cast("str", row.symbol),
        side=side,
        qty=qty,
        price=price,
        commission=commission_of(side, qty, price, cash_delta),
        cash_delta=cash_delta,
        ts=_ts(cast("int | None", row.ts)),
    )


def _conid_of(position_id: str) -> int | None:
    """The conid a lot id names, or ``None`` when it is not an IBKR conid.

    ``position_id`` is TEXT and carries either ``str(conid)`` (the IBKR
    executions book) or a sim lot id, so a non-numeric id is a legitimate sim
    value rather than an error; the book-row fold rejects it where one is
    required.
    """
    return int(position_id) if position_id.isdigit() else None
