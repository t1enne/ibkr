"""DB helpers — connection and query functions.

Pure sqlite3 wrappers. No ORM dependency at this layer.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

#: Overrides the default DB file. Lets a test (or a second book) point at its own
#: sqlite without changing every call site that relies on the default path.
_DB_PATH_ENV = "IBKR_DB_PATH"

_DEFAULT_DB_PATH = Path(
    os.environ.get(_DB_PATH_ENV)
    or Path(__file__).resolve().parent.parent.parent.parent / "data" / "db.sqlite"
)


def get_connection(db_path: Optional[str | Path] = None) -> sqlite3.Connection:
    """Return a sqlite3 connection to the candle database."""
    path = Path(db_path) if db_path else _DEFAULT_DB_PATH
    return sqlite3.connect(str(path))


def query_candles(
    symbol: str,
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
    bar: str = "1h",
    db_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Load OHLCV candles for a symbol from the local database.

    Returns DataFrame with columns: symbol, open, high, low, close, volume,
    indexed by timestamp (DatetimeIndex).
    """
    con = get_connection(db_path)
    cur = con.cursor()

    _sd = int(start_ts.timestamp() * 1000) if start_ts else None
    _ed = int(end_ts.timestamp() * 1000) if end_ts else None

    from_clause = f"AND timestamp >= {_sd}" if _sd else ""
    to_clause = f"AND timestamp <= {_ed}" if _ed else ""

    # Query candle.ticker directly (indexed, uppercase as stored) rather than a
    # join through the symbol table: the candle.conid -> symbol.conid lookup is
    # unindexed and makes the load ~20x slower.
    q = f"""
        SELECT ticker AS symbol,
               timestamp,
               open, high, low, close, volume
        FROM candle
        WHERE ticker = UPPER('{symbol}')
        {from_clause} {to_clause}
        ORDER BY timestamp ASC
    """
    rows = cur.execute(q).fetchall()
    con.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"],
    )
    df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("Date").drop(columns=["timestamp"])

    if bar != "1h":
        from src.data.resample import resample_ohlcv

        return resample_ohlcv(df, bar, completed_only=True)

    return df
