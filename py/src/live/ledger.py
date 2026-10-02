"""SQLite strategy→lot ledger — which open lots belong to this strategy.

Broker is the source of truth for cash/qty; this only answers "which open lots
are mine, and what is their broker id?". A close **mark-closes** the row (status
+ ``closed_at``), never a hard delete; ``prune_closed`` ages it out later.
Live-only: the backtest persists nothing. Tables live in the SAME candle DB and
are created idempotently, so there is one sqlite path (``src.data.db``).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol, cast

import pandas as pd

from src.bt.state import ActionType, Position
from src.data.db import get_connection

PositionStatus = Literal["open", "closed"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS live_strategy (
  strategy_id  TEXT PRIMARY KEY,
  name         TEXT NOT NULL,
  mode         TEXT NOT NULL,
  created_at   INTEGER NOT NULL,
  last_cycle_at INTEGER
);
CREATE TABLE IF NOT EXISTS live_position (
  strategy_id  TEXT NOT NULL,
  position_id  TEXT NOT NULL,
  symbol       TEXT NOT NULL,
  side         TEXT NOT NULL,
  qty          REAL NOT NULL,
  entry_price  REAL NOT NULL,
  entry_time   INTEGER NOT NULL,
  stop_loss    REAL, take_profit REAL, tag TEXT NOT NULL DEFAULT '',
  status       TEXT NOT NULL,
  opened_at    INTEGER NOT NULL,
  closed_at    INTEGER,
  PRIMARY KEY (strategy_id, position_id)
);
CREATE INDEX IF NOT EXISTS ix_live_pos_open
  ON live_position(strategy_id, status);
"""

# Timestamps round-trip through INTEGER epoch milliseconds — the same clock the
# candle table uses, so a ledger row and a bar are comparable without a tz step.
_MS = 1000


@dataclass(frozen=True)
class PositionRecord:
    """One strategy-owned live lot. ``position_id`` IS the broker lot id."""

    strategy_id: str
    position_id: str
    symbol: str
    side: str  # "long" | "short"
    qty: float  # absolute shares
    entry_price: float
    entry_time: pd.Timestamp
    stop_loss: float | None
    take_profit: float | None
    tag: str
    status: PositionStatus
    opened_at: pd.Timestamp
    closed_at: pd.Timestamp | None = None


class Ledger(Protocol):
    """Strategy→lot ownership record. Broker is cash/qty truth; this is a handle."""

    def ensure_strategy(self, strategy_id: str, name: str, mode: str) -> None: ...
    def record_open(self, rec: PositionRecord) -> None: ...
    def mark_closed(
        self, strategy_id: str, position_id: str, closed_at: pd.Timestamp
    ) -> None: ...
    def open_positions(self, strategy_id: str) -> tuple[PositionRecord, ...]: ...
    def prune_closed(self, before: pd.Timestamp) -> int: ...
    def touch_cycle(self, strategy_id: str, at: pd.Timestamp) -> None: ...


def config_hash(config: Mapping[str, object]) -> str:
    """Pure: sha256 of canonical JSON (sorted keys, no whitespace) -> hex.

    The strategy scope key: order-insensitive over dict keys, sensitive to
    values (and nesting), so a mutated config hashes differently by design.
    """
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def from_position(rec: PositionRecord, pos: Position) -> PositionRecord:
    """Pure: lift a live ``Position`` into a ledger record (identity from ``pos``)."""
    return replace(
        rec,
        position_id=pos.position_id,
        symbol=pos.symbol,
        side="long" if pos.type is ActionType.long else "short",
        qty=pos.qty,
        entry_price=pos.entry_price,
        entry_time=pos.entry_time,
        stop_loss=pos.stop_loss,
        take_profit=pos.take_profit,
        tag=pos.tag,
    )


_COLUMNS = (
    "strategy_id, position_id, symbol, side, qty, entry_price, entry_time, "
    "stop_loss, take_profit, tag, status, opened_at, closed_at"
)
_UPSERT = f"""
INSERT OR REPLACE INTO live_position ({_COLUMNS})
VALUES (?,?,?,?,?,?,?,?,?,?,'open',?,NULL)
"""


class SqliteLedger:
    """sqlite3-backed Ledger. One short-lived connection per method, no held handle.

    ``record_open`` UPSERTs (``INSERT OR REPLACE`` on the ``(strategy_id,
    position_id)`` primary key) and therefore **resurrects** a previously closed
    row if the same broker ``position_id`` is re-recorded — the REPLACE clears
    ``closed_at`` and resets ``status='open'``. This is deliberate: a broker id
    is expected to be unique per lot, so re-seeing one means the lot is open
    again (or was never truly closed).
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = db_path
        with self._connect() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open, commit on clean exit, always close — one connection per call."""
        con = get_connection(self._db_path)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def ensure_strategy(self, strategy_id: str, name: str, mode: str) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT OR IGNORE INTO live_strategy "
                "(strategy_id, name, mode, created_at) VALUES (?,?,?,?)",
                (strategy_id, name, mode, _ms(pd.Timestamp.now())),
            )

    def record_open(self, rec: PositionRecord) -> None:
        with self._connect() as con:
            con.execute(
                _UPSERT,
                (
                    rec.strategy_id,
                    rec.position_id,
                    rec.symbol,
                    rec.side,
                    rec.qty,
                    rec.entry_price,
                    _ms(rec.entry_time),
                    rec.stop_loss,
                    rec.take_profit,
                    rec.tag,
                    _ms(rec.opened_at),
                ),
            )

    def mark_closed(
        self, strategy_id: str, position_id: str, closed_at: pd.Timestamp
    ) -> None:
        """Flip status + stamp ``closed_at``; row retained. Unknown id is a no-op."""
        with self._connect() as con:
            con.execute(
                "UPDATE live_position SET status='closed', closed_at=? "
                "WHERE strategy_id=? AND position_id=?",
                (_ms(closed_at), strategy_id, position_id),
            )

    def open_positions(self, strategy_id: str) -> tuple[PositionRecord, ...]:
        """Only ``status='open'`` rows, ordered for determinism."""
        with self._connect() as con:
            rows = con.execute(
                f"SELECT {_COLUMNS} FROM live_position "
                "WHERE strategy_id=? AND status='open' "
                "ORDER BY opened_at, position_id",
                (strategy_id,),
            ).fetchall()
        return tuple(_row_to_record(row) for row in rows)

    def prune_closed(self, before: pd.Timestamp) -> int:
        """Delete closed rows older than *before*; return the deleted count.

        The only DELETE in the module — open rows are never pruned.
        """
        with self._connect() as con:
            cur = con.execute(
                "DELETE FROM live_position WHERE status='closed' AND closed_at < ?",
                (_ms(before),),
            )
            return cur.rowcount

    def touch_cycle(self, strategy_id: str, at: pd.Timestamp) -> None:
        with self._connect() as con:
            con.execute(
                "UPDATE live_strategy SET last_cycle_at=? WHERE strategy_id=?",
                (_ms(at), strategy_id),
            )


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * _MS)


_Row = tuple[
    str,  # strategy_id
    str,  # position_id
    str,  # symbol
    str,  # side
    float,  # qty
    float,  # entry_price
    int,  # entry_time (ms)
    float | None,  # stop_loss
    float | None,  # take_profit
    str,  # tag
    str,  # status
    int,  # opened_at (ms)
    int | None,  # closed_at (ms)
]


def _row_to_record(row: _Row) -> PositionRecord:
    (
        strategy_id,
        position_id,
        symbol,
        side,
        qty,
        entry_price,
        entry_time,
        stop_loss,
        take_profit,
        tag,
        status,
        opened_at,
        closed_at,
    ) = row
    return PositionRecord(
        strategy_id=strategy_id,
        position_id=position_id,
        symbol=symbol,
        side=side,
        qty=qty,
        entry_price=entry_price,
        entry_time=cast("pd.Timestamp", pd.Timestamp(entry_time, unit="ms")),
        stop_loss=stop_loss,
        take_profit=take_profit,
        tag=tag,
        status=cast("PositionStatus", status),
        opened_at=cast("pd.Timestamp", pd.Timestamp(opened_at, unit="ms")),
        closed_at=(
            cast("pd.Timestamp", pd.Timestamp(closed_at, unit="ms"))
            if closed_at is not None
            else None
        ),
    )
