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
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import cast

import pandas as pd
import peewee
from peewee import (
    CompositeKey,
    FloatField,
    IntegerField,
    Model,
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

#: Template binding only: the model CLASSES are defined against this placeholder.
#: Each ``SqliteLedger`` rebinds them (via ``bind_ctx``) to its OWN
#: ``SqliteDatabase`` for every operation, so two ledgers on two paths never share
#: a connection or retarget each other — peewee binds a model at CLASS level, so a
#: single module-global database cannot serve two live instances.
_TEMPLATE_DB = SqliteDatabase(None)

logger = logging.getLogger(__name__)

#: Where a pre-conid ``live_position`` is preserved instead of dropped by migration.
#: A name, not a schema: the rows kept here are the legacy table verbatim.
_LEGACY_POSITION_TABLE = "live_position_legacy"


class LedgerReadError(RuntimeError):
    """A durable read failed for a reason OTHER than the table being absent.

    Only a genuinely missing table (the ``--dry-run`` case) means "unwritten";
    a lock, a corrupt image or a shape-drifted column must fail loudly, never be
    reported as an empty book — an empty book reads downstream as "flat" and
    re-opens everything.
    """


def _is_missing_table(error: peewee.OperationalError) -> bool:
    """Whether *error* is the genuine ``no such table`` (dry-run) case.

    peewee wraps every ``sqlite3.OperationalError`` (a lock, a bad column, a
    missing table) as the same ``peewee.OperationalError`` type, so the message
    is the only discriminator left.
    """
    return "no such table" in str(error)


class _Base(Model):
    class Meta:
        database = _TEMPLATE_DB


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


class LiveSimLot(_Base):
    scope = TextField()
    position_id = TextField()
    closed_at = IntegerField(null=True)

    class Meta:
        table_name = "live_sim_lot"
        primary_key = CompositeKey("scope", "position_id")


class LiveOrderIntent(_Base):
    """The durable owner of OPEN order state, keyed by ``(scope, token)``.

    One row per intent key (``symbol``/``action``/``position_id`` split out for
    the collision check and for readability). ``state`` is an ``IntentState``
    value, ``attempt`` the re-send counter, ``order_ref`` the minted cOID and
    ``order_id`` the broker's id (null while unknown). Added to the model set
    above so the existing ``IF NOT EXISTS`` create path builds it — no ALTER,
    no rename, nothing existing is touched.
    """

    scope = TextField()
    token = TextField()
    symbol = TextField()
    action = TextField()
    position_id = TextField(null=True)
    state = TextField()
    attempt = IntegerField()
    order_ref = TextField()
    order_id = TextField(null=True)
    decision_ts = IntegerField(null=True)
    updated_at = IntegerField()

    class Meta:
        table_name = "live_order_intent"
        primary_key = CompositeKey("scope", "token")


_MODELS = (
    LiveStrategy,
    LivePosition,
    LiveExecution,
    LiveCash,
    LiveSimLot,
    LiveOrderIntent,
)

#: The terminal (closed) intent states ``prune`` is allowed to delete.
_CLOSED_INTENTS = [
    IntentState.FILLED.value,
    IntentState.UNFILLED.value,
    IntentState.REJECTED.value,
]

# Timestamps round-trip through INTEGER epoch milliseconds — the same clock the
# candle table uses, so a book row and a bar are comparable without a tz step.
_MS = 1000


def _table_exists(db: SqliteDatabase, name: str) -> bool:
    """Whether *name* is a table in *db*."""
    return bool(
        db.execute_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchall()
    )


def _table_columns(db: SqliteDatabase, name: str) -> set[str]:
    """The column names of *name* in *db* (empty when the table does not exist)."""
    return {str(row[1]) for row in db.execute_sql(f"PRAGMA table_info({name})")}


def _free_legacy_name(db: SqliteDatabase) -> str:
    """A table name to keep a legacy ``live_position`` under, never overwriting one.

    The canonical copy is ``live_position_legacy``; if that already exists (a
    legacy-shaped ``live_position`` re-created after a prior migration), a fresh
    ``live_position_legacy_<n>`` is chosen so EVERY copy is kept — a rename,
    never a drop, so no row is ever lost.
    """
    if not _table_exists(db, _LEGACY_POSITION_TABLE):
        return _LEGACY_POSITION_TABLE
    n = 1
    while _table_exists(db, f"{_LEGACY_POSITION_TABLE}_{n}"):
        n += 1
    return f"{_LEGACY_POSITION_TABLE}_{n}"


def _preserve_legacy_positions(db: SqliteDatabase) -> None:
    """Preserve a pre-conid ``live_position`` under a kept copy and warn.

    A pre-4.1 ``live_position`` cannot be read into the conid-keyed book, but
    dropping it would erase durable position state the rolling trades window
    cannot rebuild. RENAME it to a kept copy (a fresh name when one is already
    kept, so a re-created table is preserved too) instead of dropping, naming any
    still-open rows loudly: until an operator reconciles them, the strategy reads
    the symbol as flat and could open again on top of a live position.
    """
    columns = _table_columns(db, "live_position")
    target = _free_legacy_name(db)
    sql = "SELECT symbol, side, qty FROM live_position"
    for symbol, side, qty in db.execute_sql(
        sql + (" WHERE status = 'open'" if "status" in columns else "")
    ).fetchall():
        logger.warning(
            "preserved legacy live_position open row %s %s qty=%s: the "
            "conid-keyed book cannot reproduce it — reconcile it from %s before "
            "trading, or the open may be doubled",
            symbol,
            side,
            qty,
            target,
        )
    db.execute_sql(f"ALTER TABLE live_position RENAME TO {target}")


def _rekey_sim_lots(db: SqliteDatabase) -> None:
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


class SqliteLedger:
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
            self._migrate()
            self._database.create_tables(_MODELS)
        self._schema_ready = True

    def _migrate(self) -> None:
        """Preserve a pre-4.1 position table, ALTER ``live_strategy`` and re-key sim lots.

        The old ``live_position`` (PK ``(strategy_id, position_id)``) cannot
        express a ``(scope, conid)`` row, so it is renamed to a kept copy and
        warned about — never dropped, which would erase durable position state
        the rolling trades window cannot rebuild. ``live_strategy`` gains
        ``scope`` via ALTER so the audit rows survive. ``live_sim_lot`` is
        re-keyed from the config hash to the scope (rows preserved).
        """
        db = self._database
        if _table_exists(db, "live_position") and "conid" not in _table_columns(
            db, "live_position"
        ):
            _preserve_legacy_positions(db)
        if _table_exists(db, "live_strategy") and "scope" not in _table_columns(
            db, "live_strategy"
        ):
            db.execute_sql(
                "ALTER TABLE live_strategy ADD COLUMN scope TEXT NOT NULL DEFAULT ''"
            )
        _rekey_sim_lots(db)

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

    # -- audit / metadata --------------------------------------------------

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

    # -- sim/mock lot ownership (position_id keyed) ------------------------
    #
    # The per-scope book above is conid-keyed because IBKR names lots by conid.
    # The sim/mock path has no conid: its broker mints a synthetic ``position_id``
    # (``SYM_{ts}_{seq}``) and the mock fixture may hold lots the strategy never
    # opened. Ownership is recorded here so a sim close can only target a lot the
    # strategy OPENED (``owned`` in ``reconcile``); a fixture lot never opened is
    # left alone. Keyed by ``scope`` — a STABLE identity — so a config edit does
    # not orphan every previously opened lot.

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
            with self._database.bind_ctx(_MODELS):
                rows = (
                    LiveSimLot.select(LiveSimLot.position_id)
                    .where((LiveSimLot.scope == scope) & LiveSimLot.closed_at.is_null())
                    .execute()
                )
        except peewee.OperationalError as exc:
            if not _is_missing_table(exc):
                raise LedgerReadError(str(exc)) from exc
            return frozenset()
        return frozenset(cast("str", r.position_id) for r in rows)

    # -- book --------------------------------------------------------------

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

    # -- pending order intents (PendingIntents) ----------------------------
    #
    # The durable owner of OPEN order state. Reads tolerate a missing table
    # (empty), like ``load_book``; writes create it lazily. A stored row whose
    # symbol/action/position_id disagrees with the token's key is a crc32
    # collision: it is warned about and treated as ABSENT, so a foreign identity
    # is never adopted. (A true collision-bump would require identity to consult
    # the store, which the pure ``IntentKey.token`` forbids.)

    def load(self, key: IntentKey) -> IntentRecord | None:
        """The durable record for *key*, or ``None`` when unwritten/foreign."""
        try:
            with self._database.bind_ctx(_MODELS):
                row = LiveOrderIntent.get_or_none(
                    (LiveOrderIntent.scope == key.scope)
                    & (LiveOrderIntent.token == key.token())
                )
        except peewee.OperationalError as exc:
            if not _is_missing_table(exc):
                raise LedgerReadError(str(exc)) from exc
            return None
        if row is None:
            return None
        if not _row_matches_key(row, key):
            logger.warning(
                "live_order_intent token %s under scope %r collides with key "
                "%s|%s|%s; refusing to adopt a foreign identity",
                key.token(),
                key.scope,
                key.symbol,
                key.action.value,
                key.position_id,
            )
            return None
        return _model_to_intent(row)

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

    def save(self, record: IntentRecord) -> None:
        """Upsert *record* keyed ``(scope, token)``; warn on a token collision."""
        with self._write():
            occupied = LiveOrderIntent.get_or_none(
                (LiveOrderIntent.scope == record.key.scope)
                & (LiveOrderIntent.token == record.key.token())
            )
            if occupied is not None and not _row_matches_key(occupied, record.key):
                logger.warning(
                    "live_order_intent token %s under scope %r already belongs to "
                    "%s|%s|%s; replacing it with %s|%s|%s (crc32 collision)",
                    record.key.token(),
                    record.key.scope,
                    occupied.symbol,
                    occupied.action,
                    occupied.position_id,
                    record.key.symbol,
                    record.key.action.value,
                    record.key.position_id,
                )
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
        so an UNRESOLVED transition does not erase a known id.
        """
        fields: dict[str, object] = {"state": state.value, "updated_at": _ms(now)}
        if order_id is not None:
            fields["order_id"] = order_id
        with self._write():
            LiveOrderIntent.update(**fields).where(
                (LiveOrderIntent.scope == key.scope)
                & (LiveOrderIntent.token == key.token())
            ).execute()

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


def _ms(ts: pd.Timestamp) -> int:
    """Epoch-milliseconds of *ts*.

    A tz-naive input is read as LOCAL time, so the stored integer is the absolute
    instant; :func:`_ts` reads it back tagged UTC. The round trip preserves the
    absolute time, not the tz label (display tz differs, the instant does not).
    """
    return int(ts.timestamp() * _MS)


def _ts(value: int | None) -> pd.Timestamp | None:
    """Epoch-ms -> UTC ``Timestamp`` (see :func:`_ms`: absolute instant preserved)."""
    if value is None:
        return None
    return cast("pd.Timestamp", pd.Timestamp(value, unit="ms", tz="UTC"))


def _row_matches_key(row: LiveOrderIntent, key: IntentKey) -> bool:
    """Whether a stored intent row names the same trade as *key*."""
    return (
        row.symbol == key.symbol
        and row.action == key.action.value
        and (row.position_id or None) == key.position_id
    )


def _intent_fields(record: IntentRecord) -> dict[str, object]:
    now = pd.Timestamp.now()
    return {
        "scope": record.key.scope,
        "token": record.key.token(),
        "symbol": record.key.symbol,
        "action": record.key.action.value,
        "position_id": record.key.position_id,
        "state": record.state.value,
        "attempt": record.attempt,
        "order_ref": record.order_ref,
        "order_id": record.order_id,
        "decision_ts": (
            _ms(record.decision_ts) if record.decision_ts is not None else None
        ),
        "updated_at": _ms(now),
    }


def _model_to_intent(row: LiveOrderIntent) -> IntentRecord:
    return IntentRecord(
        key=IntentKey(
            scope=cast("str", row.scope),
            symbol=cast("str", row.symbol),
            action=ActionType(cast("str", row.action)),
            position_id=cast("str | None", row.position_id),
        ),
        state=IntentState(cast("str", row.state)),
        attempt=int(cast("int", row.attempt)),
        order_ref=cast("str", row.order_ref),
        order_id=cast("str | None", row.order_id),
        decision_ts=_ts(cast("int | None", row.decision_ts)),
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
