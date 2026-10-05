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
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pandas as pd

from src.data.db import get_connection
from src.exec.types import OrderSide
from src.live.adapters.ibkr.trades import (
    BookRow,
    Execution,
    StrategyBook,
    is_ours,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS live_strategy (
  strategy_id  TEXT PRIMARY KEY,
  scope        TEXT NOT NULL DEFAULT '',
  name         TEXT NOT NULL,
  mode         TEXT NOT NULL,
  created_at   INTEGER NOT NULL,
  last_cycle_at INTEGER
);
CREATE TABLE IF NOT EXISTS live_position (
  scope        TEXT NOT NULL,
  conid        INTEGER NOT NULL,
  symbol       TEXT NOT NULL,
  side         TEXT NOT NULL,
  qty          REAL NOT NULL,
  entry_price  REAL NOT NULL,
  opened_at    INTEGER,
  closed_at    INTEGER,
  stop_loss    REAL, take_profit REAL, tag TEXT NOT NULL DEFAULT '',
  order_ref    TEXT NOT NULL DEFAULT '',
  watermark    INTEGER,
  PRIMARY KEY (scope, conid)
);
CREATE TABLE IF NOT EXISTS live_execution (
  scope        TEXT NOT NULL,
  execution_id TEXT NOT NULL,
  conid        INTEGER NOT NULL,
  side         TEXT NOT NULL,
  qty          REAL NOT NULL,
  price        REAL NOT NULL,
  commission   REAL NOT NULL,
  cash_delta   REAL NOT NULL,
  ts           INTEGER NOT NULL,
  PRIMARY KEY (scope, execution_id)
);
CREATE TABLE IF NOT EXISTS live_cash (
  scope            TEXT PRIMARY KEY,
  initial_capital  REAL NOT NULL,
  updated_at       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_live_pos_open
  ON live_position(scope, closed_at);
"""

# Timestamps round-trip through INTEGER epoch milliseconds — the same clock the
# candle table uses, so a book row and a bar are comparable without a tz step.
_MS = 1000

_POSITION_COLUMNS = (
    "scope, conid, symbol, side, qty, entry_price, opened_at, closed_at, "
    "stop_loss, take_profit, tag, order_ref, watermark"
)
_UPSERT_POSITION = f"""
INSERT OR REPLACE INTO live_position ({_POSITION_COLUMNS})
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
"""


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
    """sqlite-backed per-scope book. Short-lived connection per method.

    Construction writes NOTHING (no DDL): a ``--dry-run`` that only reads must
    leave the schema untouched. The schema is created (and migrated) lazily on
    the first write; reads of a missing table return an empty book.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = db_path
        self._schema_ready = False

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open, commit on clean exit, always close — one connection per call."""
        con = get_connection(self._db_path)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """A write connection with the schema ensured exactly once per ledger."""
        with self._connect() as con:
            if not self._schema_ready:
                self._migrate(con)
                con.executescript(SCHEMA)
                self._schema_ready = True
            yield con

    @staticmethod
    def _migrate(con: sqlite3.Connection) -> None:
        """Drop/migrate a pre-4.1 schema that cannot carry a per-conid book.

        The old ``live_position`` (PK ``(strategy_id, position_id)``) cannot
        express a ``(scope, conid)`` row; it is dropped and rebuilt from the
        trades window. ``live_strategy`` gains ``scope`` via ALTER so the audit
        rows survive.
        """
        columns = {
            row[1] for row in con.execute("PRAGMA table_info(live_position)").fetchall()
        }
        if columns and "conid" not in columns:
            con.execute("DROP TABLE live_position")
        strategy_columns = {
            row[1] for row in con.execute("PRAGMA table_info(live_strategy)").fetchall()
        }
        if strategy_columns and "scope" not in strategy_columns:
            con.execute(
                "ALTER TABLE live_strategy ADD COLUMN scope TEXT NOT NULL DEFAULT ''"
            )

    # -- audit / metadata --------------------------------------------------

    def ensure_strategy(
        self, strategy_id: str, scope: str, name: str, mode: str
    ) -> None:
        with self._write() as con:
            con.execute(
                "INSERT OR IGNORE INTO live_strategy "
                "(strategy_id, scope, name, mode, created_at) VALUES (?,?,?,?,?)",
                (strategy_id, scope, name, mode, _ms(pd.Timestamp.now())),
            )

    def ensure_cash(self, scope: str, initial_capital: float) -> None:
        """Record the scope's ``initial_capital`` once; a later cycle never resets it."""
        with self._write() as con:
            con.execute(
                "INSERT OR IGNORE INTO live_cash (scope, initial_capital, updated_at) "
                "VALUES (?,?,?)",
                (scope, initial_capital, _ms(pd.Timestamp.now())),
            )

    def touch_cycle(self, strategy_id: str, at: pd.Timestamp) -> None:
        with self._write() as con:
            con.execute(
                "UPDATE live_strategy SET last_cycle_at=? WHERE strategy_id=?",
                (_ms(at), strategy_id),
            )

    # -- book --------------------------------------------------------------

    def load_book(self, scope: str) -> StrategyBook:
        """The durable rows + applied execution ids for *scope* (empty if unaware)."""
        try:
            with self._connect() as con:
                rows = con.execute(
                    f"SELECT {_POSITION_COLUMNS} FROM live_position WHERE scope=? "
                    "ORDER BY conid",
                    (scope,),
                ).fetchall()
                applied = {
                    r[0]
                    for r in con.execute(
                        "SELECT execution_id FROM live_execution WHERE scope=?",
                        (scope,),
                    ).fetchall()
                }
        except sqlite3.OperationalError:
            return StrategyBook()
        return StrategyBook(
            rows=tuple(_row_to_book(row) for row in rows), applied=frozenset(applied)
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
        with self._write() as con:
            con.execute(
                "INSERT OR IGNORE INTO live_cash (scope, initial_capital, updated_at) "
                "VALUES (?,?,?)",
                (scope, initial_capital, _ms(pd.Timestamp.now())),
            )
            for row in book.rows:
                con.execute(_UPSERT_POSITION, _book_to_row(row))
            for execution in executions:
                if execution.execution_id not in book.applied:
                    continue
                if not is_ours(scope, execution):
                    continue
                con.execute(
                    "INSERT OR IGNORE INTO live_execution "
                    "(scope, execution_id, conid, side, qty, price, commission, "
                    "cash_delta, ts) VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        scope,
                        execution.execution_id,
                        execution.conid,
                        execution.side.value,
                        execution.qty,
                        execution.price,
                        execution.commission,
                        _cash_delta(execution),
                        _ms(execution.ts),
                    ),
                )

    def cash_of(self, scope: str, default_initial: float = 0.0) -> float:
        """Per-scope cash: ``initial_capital`` advanced by the scope's own fills.

        The stored ``initial_capital`` (first cycle's config value) wins; a scope
        with no stored row falls back to *default_initial*. Never reads the
        account summary: N strategies share one account's cash.
        """
        try:
            with self._connect() as con:
                row = con.execute(
                    "SELECT initial_capital FROM live_cash WHERE scope=?", (scope,)
                ).fetchone()
                initial = float(row[0]) if row else default_initial
                (delta,) = con.execute(
                    "SELECT COALESCE(SUM(cash_delta), 0.0) FROM live_execution "
                    "WHERE scope=?",
                    (scope,),
                ).fetchone()
        except sqlite3.OperationalError:
            return default_initial
        return initial + float(delta)

    def initial_capital_of(self, scope: str) -> float:
        try:
            with self._connect() as con:
                row = con.execute(
                    "SELECT initial_capital FROM live_cash WHERE scope=?", (scope,)
                ).fetchone()
        except sqlite3.OperationalError:
            return 0.0
        return float(row[0]) if row else 0.0

    def prune_closed(self, before: pd.Timestamp) -> int:
        """Delete closed rows older than *before*; return the deleted count.

        The only DELETE over book rows — open rows are never pruned.
        """
        with self._write() as con:
            cur = con.execute(
                "DELETE FROM live_position WHERE closed_at IS NOT NULL AND closed_at < ?",
                (_ms(before),),
            )
            return cur.rowcount


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * _MS)


def _ts(value: int | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return cast("pd.Timestamp", pd.Timestamp(value, unit="ms", tz="UTC"))


def _book_to_row(row: BookRow) -> tuple[object, ...]:
    return (
        row.scope,
        row.conid,
        row.symbol,
        row.side,
        row.qty,
        row.entry_price,
        _ms(row.opened_at) if row.opened_at is not None else None,
        _ms(row.closed_at) if row.closed_at is not None else None,
        row.stop_loss,
        row.take_profit,
        row.tag,
        row.order_ref,
        _ms(row.watermark) if row.watermark is not None else None,
    )


def _row_to_book(row: tuple[object, ...]) -> BookRow:
    (
        scope,
        conid,
        symbol,
        side,
        qty,
        entry_price,
        opened_at,
        closed_at,
        stop_loss,
        take_profit,
        tag,
        order_ref,
        watermark,
    ) = row
    return BookRow(
        scope=cast("str", scope),
        conid=int(cast("int", conid)),
        symbol=cast("str", symbol),
        side=cast("str", side),
        qty=float(cast("float", qty)),
        entry_price=float(cast("float", entry_price)),
        opened_at=_ts(cast("int | None", opened_at)),
        closed_at=_ts(cast("int | None", closed_at)),
        stop_loss=cast("float | None", stop_loss),
        take_profit=cast("float | None", take_profit),
        tag=cast("str", tag),
        order_ref=cast("str", order_ref),
        watermark=_ts(cast("int | None", watermark)),
    )
