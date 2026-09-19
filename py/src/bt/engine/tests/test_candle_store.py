"""Tests for CandleStore — lazy DataFrame view over numpy column arrays."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.bt.engine.candle_store import CandleStore, CandleRows


def _ts(s: str) -> pd.Timestamp:
    """Parse a string timestamp, returning a typed pd.Timestamp."""
    return cast(pd.Timestamp, pd.Timestamp(s))


def _make_rows() -> CandleRows:
    """Build a CandleRows accumulator with one (symbol, interval) key, 3 rows."""
    n = 3
    ts = np.array(
        [
            np.datetime64("2024-01-02 10:00"),
            np.datetime64("2024-01-02 11:00"),
            np.datetime64("2024-01-02 12:00"),
        ],
        dtype="datetime64[ms]",
    )
    rows: CandleRows = {}
    rows[("SPY", "1h")] = {
        "timestamp": np.empty(8, dtype="datetime64[ms]"),
        "open": np.empty(8, dtype=np.float64),
        "high": np.empty(8, dtype=np.float64),
        "low": np.empty(8, dtype=np.float64),
        "close": np.empty(8, dtype=np.float64),
        "volume": np.empty(8, dtype=np.float64),
        "_len": np.array([n], dtype=np.int64),
    }
    rows[("SPY", "1h")]["timestamp"][:n] = ts
    rows[("SPY", "1h")]["open"][:n] = [100.0, 101.0, 102.0]
    rows[("SPY", "1h")]["high"][:n] = [103.0, 104.0, 105.0]
    rows[("SPY", "1h")]["low"][:n] = [99.0, 100.0, 101.0]
    rows[("SPY", "1h")]["close"][:n] = [101.0, 102.0, 103.0]
    rows[("SPY", "1h")]["volume"][:n] = [1000.0, 1100.0, 1200.0]
    return rows


def test_latest_and_count():
    """Fast-path O(1) reads. Missing key returns None/0."""
    store = CandleStore(_make_rows())
    assert store.latest("SPY", "1h") == 103.0
    assert store.count("SPY", "1h") == 3
    assert store.latest("AAPL", "1h") is None
    assert store.count("AAPL", "1h") == 0


def test_latest_and_count_ignore_cursor():
    """Cursor truncation only affects DataFrame builds, not latest()/count()."""
    store = CandleStore(_make_rows())
    store.advance(_ts("2024-01-02 10:30"))  # only first row
    assert store.latest("SPY", "1h") == 103.0  # not 101.0
    assert store.count("SPY", "1h") == 3  # not 1


def test_advance_truncation():
    """Cursor truncates DataFrame to rows at or before timestamp."""
    store = CandleStore(_make_rows())

    store.advance(_ts("2024-01-02 10:30"))
    df = store[("SPY", "1h")]
    assert len(df) == 1
    assert float(df["close"].iloc[0]) == 101.0

    store.advance(_ts("2024-01-02 09:00"))
    df = store[("SPY", "1h")]
    assert len(df) == 0
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    store.advance(_ts("2024-01-02 23:59"))
    assert len(store[("SPY", "1h")]) == 3


def test_mapping_interface():
    """__getitem__, get, contains, len, iter, items, keys work."""
    store = CandleStore(_make_rows())

    df = store[("SPY", "1h")]
    assert len(df) == 3
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    with pytest.raises(KeyError):
        _ = store[("AAPL", "1h")]

    assert store.get(("AAPL", "1h")) is None
    assert store.get(("AAPL", "1h"), "fallback") == "fallback"

    assert ("SPY", "1h") in store
    assert len(store) == 1
    assert list(store.keys()) == [("SPY", "1h")]


def test_wraps_by_reference():
    """Mutations to underlying rows dict are visible through the store."""
    rows = _make_rows()
    store = CandleStore(rows)

    cols = rows[("SPY", "1h")]
    old_n = int(cols["_len"][0])
    cols["timestamp"][old_n] = np.datetime64("2024-01-02 13:00")
    cols["open"][old_n] = 104.0
    cols["high"][old_n] = 105.0
    cols["low"][old_n] = 103.0
    cols["close"][old_n] = 104.0
    cols["volume"][old_n] = 1300.0
    cols["_len"][0] = old_n + 1

    assert store.count("SPY", "1h") == 4
    assert store.latest("SPY", "1h") == 104.0


def test_is_exhausted_requires_cursor_at_final_bar():
    """is_exhausted is False before any advance and while truncated."""
    store = CandleStore(_make_rows())
    assert store.is_exhausted is False  # no cursor yet
    store.advance(_ts("2024-01-02 11:00"))
    assert store.is_exhausted is False  # one bar still ahead
    store.advance(_ts("2024-01-02 12:00"))
    assert store.is_exhausted is True
    store.advance(_ts("2024-01-03 00:00"))
    assert store.is_exhausted is True  # cursor past the end still exhaustive


def test_is_exhausted_empty_store_is_false():
    assert CandleStore({}).is_exhausted is False
