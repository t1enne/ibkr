"""Per-scope sqlite book — the durable, per-strategy position store (plan §3).

The 7-day ``/iserver/account/trades`` window is a *confirmation channel*: it
advances the book. The book itself lives here, one row per ``(scope, conid)``,
plus one row per applied ``execution_id`` so re-applying the window is a no-op.
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
methods live in their own mixins (``ledger_sim``, and the stores below) so each
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
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd
import peewee
from peewee import (
    CompositeKey,
    FloatField,
    IntegerField,
    SqliteDatabase,
    TextField,
    fn,
)

from src.bt.state import ActionType
from src.data.db import _DEFAULT_DB_PATH
from src.exec.types import OrderSide
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
    _Base,
    _is_missing_table,
    _ms,
    _SqliteOps,
    _ts,
    LedgerReadError,
)
from src.live.ledger_migration import _restore_intents, migrate
from src.live.ledger_sim import LiveSimLot, SimLotBook

logger = logging.getLogger(__name__)


class LiveStrategy(_Base):
    strategy_id = TextField(primary_key=True)
    scope = TextField(null=False, default="")
    name = TextField()
    mode = TextField()
    created_at = IntegerField()
    last_cycle_at = IntegerField(null=True)

    class Meta:
        table_name = "live_strategy"


class LivePosition(_Base):
    scope = TextField()
    conid = IntegerField()
    symbol = TextField()
    side = TextField()
    qty = FloatField()
    entry_price = FloatField()
    opened_at = IntegerField(null=True)
    closed_at = IntegerField(null=True)
    stop_loss = FloatField(null=True)
    take_profit = FloatField(null=True)
    tag = TextField(default="")
    order_ref = TextField(default="")

    class Meta:
        table_name = "live_position"
        primary_key = CompositeKey("scope", "conid")
        indexes = ((("scope", "closed_at"), False),)


class LiveExecution(_Base):
    scope = TextField()
    execution_id = TextField()
    conid = IntegerField()
    side = TextField()
    qty = FloatField()
    price = FloatField()
    commission = FloatField()
    cash_delta = FloatField()
    ts = IntegerField()

    class Meta:
        table_name = "live_execution"
        primary_key = CompositeKey("scope", "execution_id")


class LiveCash(_Base):
    scope = TextField(primary_key=True)
    initial_capital = FloatField()
    updated_at = IntegerField()

    class Meta:
        table_name = "live_cash"


class LiveOrderIntent(_Base):
    """The durable owner of OPEN order state, keyed by the IDENTITY columns.

    The primary key is ``(scope, symbol, action, position_id)`` — the intent
    identity itself (``position_id`` is ``''`` for an open) — NOT the crc32
    ``token``: a crc32 collision can never alias two distinct identities' rows
    (D7). The ``token`` stays as a plain column, the bar-free cOID prefix."""

    scope = TextField()
    token = TextField()
    symbol = TextField()
    action = TextField()
    position_id = TextField(default="")
    state = TextField()
    attempt = IntegerField()
    order_ref = TextField()
    order_id = TextField(null=True)
    decision_ts = IntegerField(null=True)
    #: The time-in-force the order was placed with (see ``identity.DEFAULT_TIF``).
    tif = TextField(default="DAY")
    #: Consecutive resyncs this OPEN record stayed unresolved (wedged-key alarm).
    stuck_cycles = IntegerField(default=0)
    updated_at = IntegerField()

    class Meta:
        table_name = "live_order_intent"
        primary_key = CompositeKey("scope", "symbol", "action", "position_id")


_MODELS = (
    LiveStrategy,
    LivePosition,
    LiveExecution,
    LiveCash,
    LiveSimLot,
    LiveOrderIntent,
)


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

    ``live_execution`` stores ``conid`` but no symbol, so ``symbol`` is resolved
    from the scope's book rows (the conid's symbol, ``""`` when no row remains).
    ``ts`` is the fill's UTC instant.
    """

    scope: str
    execution_id: str
    conid: int
    symbol: str
    side: str
    qty: float
    price: float
    commission: float
    cash_delta: float
    ts: pd.Timestamp | None


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
                LiveSimLot,
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
    """Ledger mixin: the conid-keyed book, its executions and its cash seed."""

    def load_book(self, scope: str) -> StrategyBook:
        """The durable rows + applied execution ids for *scope* (empty if unaware)."""
        try:
            with self._database.bind_ctx(_MODELS):
                rows = (
                    LivePosition.select()
                    .where(LivePosition.scope == scope)
                    .order_by(LivePosition.conid)
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
            LiveCash.insert(
                scope=scope,
                initial_capital=initial_capital,
                updated_at=_ms(pd.Timestamp.now()),
            ).on_conflict("IGNORE").execute()
            for row in book.rows:
                LivePosition.insert(**_book_fields(row)).on_conflict(
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
                    conid=execution.conid,
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
        `symbol` resolved from the conid's book row (`""` when none remains). A
        missing table (a never-written scope) reads as empty; a genuine read
        failure raises :class:`LedgerReadError`.
        """
        try:
            with self._database.bind_ctx(_MODELS):
                symbols = {
                    int(row.conid): cast("str", row.symbol)
                    for row in LivePosition.select(
                        LivePosition.conid, LivePosition.symbol
                    )
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

    def cash_of(self, scope: str, default_initial: float = 0.0) -> float:
        """Per-scope cash: ``initial_capital`` advanced by the scope's own fills.

        The stored ``initial_capital`` (first cycle's config value) wins; a scope
        with no stored row falls back to *default_initial*. Never reads the
        account summary: N strategies share one account's cash.
        """
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
        return initial + float(sunk)

    def initial_capital_of(self, scope: str) -> float:
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
        account would show if every booked fill were the whole story. A missing
        table reads as a flat (0.0) book — the dry-run case; a genuine read
        failure still raises (:class:`LedgerReadError`), never a silent zero.
        """
        try:
            with self._database.bind_ctx(_MODELS):
                rows = (
                    LivePosition.select(LivePosition.side, LivePosition.qty)
                    .where(
                        (LivePosition.conid == conid) & LivePosition.closed_at.is_null()
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


class SqliteLedger(MetadataStore, BookStore, IntentStore, SimLotBook):
    """peewee-backed per-scope book. One database per ledger, bound on init.

    Construction writes NOTHING (no DDL): a ``--dry-run`` that only reads must
    leave the schema untouched. The schema is created lazily on the first
    write; reads of a missing table return an empty book.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = db_path
        self._schema_ready = False
        path = str(db_path) if db_path is not None else str(_DEFAULT_DB_PATH)
        self._database = SqliteDatabase(path)

    @property
    def db_path(self) -> str:
        """The resolved sqlite path this ledger is bound to (read-only)."""
        return str(self._database.database)

    # -- lazy DDL / migration ---------------------------------------------

    def _ready_schema(self) -> None:
        """Create the live tables exactly once, on the first WRITE only.

        The migration + DDL run in ONE ``BEGIN IMMEDIATE`` transaction. That takes
        the SQLite write lock up front, so two processes racing the first
        migration (the overlap the cycle lease guards against) are SERIALIZED: the
        second blocks until the first commits, re-reads the migrated schema and is
        a no-op. There is no check-then-act window, so neither a ``no such table``
        nor a double-drop race is reachable. It is deliberately NOT wrapped in the
        cycle lease — ``ensure_strategy``/``ensure_cash`` write before ``run_cycle``
        takes the lease, and re-taking it here would deadlock a cycle already
        holding it. What this does NOT do: it does not serialize those later
        idempotent writes against a live cycle; each is its own atomic write, and
        the lease remains the cross-process guard for placement.
        """
        if self._schema_ready:
            return
        with self._database.atomic(lock_type="IMMEDIATE"):
            legacy_intents = migrate(self._database)
            self._database.create_tables(_MODELS)
            if legacy_intents is not None:
                _restore_intents(self._database, legacy_intents)
        self._schema_ready = True

    @contextmanager
    def _write(self) -> Iterator[None]:
        """A write: models bound to this ledger's db, lazy DDL once, then atomic."""
        with self._database.bind_ctx(_MODELS):
            self._ready_schema()
            with self._database.atomic():
                yield

    def cycle_lease(self) -> AbstractContextManager[None]:
        """Exclusive cross-process lease on this ledger's DB for a full cycle.

        Held for the cycle's duration so a cron overlap or a racing human run
        REFUSES to start rather than both placing off the same pre-order book.
        The kernel drops the flock on process exit, so a crashed run never
        wedges live trading; there is no TTL.
        """
        path = self._db_path if self._db_path is not None else _DEFAULT_DB_PATH
        return file_lease(f"{path}.cycle.lock")


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


def _book_fields(row: BookRow) -> dict[str, object]:
    return {
        "scope": row.scope,
        "conid": row.conid,
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
    }


def _model_to_execution(
    row: LiveExecution, symbols: Mapping[int, str]
) -> ExecutionRecord:
    """Project a stored execution, resolving its symbol from *symbols* (conid space)."""
    conid = int(cast("int", row.conid))
    return ExecutionRecord(
        scope=cast("str", row.scope),
        execution_id=cast("str", row.execution_id),
        conid=conid,
        symbol=symbols.get(conid, ""),
        side=cast("str", row.side),
        qty=float(cast("float", row.qty)),
        price=float(cast("float", row.price)),
        commission=float(cast("float", row.commission)),
        cash_delta=float(cast("float", row.cash_delta)),
        ts=_ts(cast("int | None", row.ts)),
    )


def _model_to_book(row: LivePosition) -> BookRow:
    return BookRow(
        scope=cast("str", row.scope),
        conid=int(cast("int", row.conid)),
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
