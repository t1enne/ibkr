"""Candle reads — the one query this module still owns.

Connection handling and path resolution moved to :mod:`src.db` (``path`` +
``connection``). ``get_connection`` and ``_DEFAULT_DB_PATH`` are RE-EXPORTED here
so existing callers keep working; the path they resolve is now file-relative
instead of ``os.getcwd()``-dependent.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.db.connection import get_connection  # noqa: F401  (re-export)
from src.db.path import DEFAULT_DB_PATH

#: Kept for back-compat with callers that read the private name from here.
_DB_PATH_ENV = "IBKR_DB_PATH"
_DEFAULT_DB_PATH = DEFAULT_DB_PATH

__all__ = ["get_connection", "query_candles"]


def query_candles(
    symbol: str,
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
    bar: str = "1h",
    db_path: Optional[str | None] = None,
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
