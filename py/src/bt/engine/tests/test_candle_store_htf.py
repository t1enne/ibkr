"""CandleStore HTF completion lag — the forming bucket must never leak.

Methodology: build a ``CandleRows`` accumulator holding one base interval and one
HTF interval whose rows are labelled by bucket START but carry the whole bucket
(exactly what ``resample_ohlcv`` + ``candle_generator`` produce). Construct the
store with the HTF interval set, advance the cursor to a timestamp INSIDE the
last bucket, and assert every read path hides that bucket. The pre-fix store had
no notion of bucket widths, so it exposed the forming bucket's future-inclusive
OHLC at its start label — these tests fail there.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from src.bt.engine.candle_store import CandleStore, CandleRows


def _ts(s: str) -> pd.Timestamp:
    """Parse a string timestamp into a typed ``pd.Timestamp``."""
    return cast(pd.Timestamp, pd.Timestamp(s))


def _rows() -> CandleRows:
    """Two HTF buckets (4h) labelled by start, plus their base bars.

    HTF bucket 14:00 spans 14:00..18:00 and aggregates base bars 14:00..17:00
    (close 103). HTF bucket 18:00 spans 18:00..22:00 and aggregates 18:00..21:00
    (close 107) — its future-inclusive close is 107 while the cursor sits at
    18:00.
    """
    rows: CandleRows = {}
    htf_ts = np.array(
        [np.datetime64("2025-01-06 14:00"), np.datetime64("2025-01-06 18:00")],
        dtype="datetime64[ms]",
    )
    rows[("SPY", "4h")] = {
        "timestamp": np.empty(4, dtype="datetime64[ms]"),
        "open": np.empty(4, dtype=np.float64),
        "high": np.empty(4, dtype=np.float64),
        "low": np.empty(4, dtype=np.float64),
        "close": np.empty(4, dtype=np.float64),
        "volume": np.empty(4, dtype=np.float64),
        "_len": np.array([2], dtype=np.int64),
    }
    c = rows[("SPY", "4h")]
    c["timestamp"][:2] = htf_ts
    c["open"][:2] = [100.0, 104.0]
    c["high"][:2] = [103.0, 107.0]
    c["low"][:2] = [99.0, 103.0]
    c["close"][:2] = [103.0, 107.0]
    c["volume"][:2] = [4000.0, 4000.0]

    base_ts = np.array(
        [
            np.datetime64("2025-01-06 14:00"),
            np.datetime64("2025-01-06 15:00"),
            np.datetime64("2025-01-06 16:00"),
            np.datetime64("2025-01-06 17:00"),
            np.datetime64("2025-01-06 18:00"),
        ],
        dtype="datetime64[ms]",
    )
    rows[("SPY", "1h")] = {
        "timestamp": np.empty(8, dtype="datetime64[ms]"),
        "open": np.empty(8, dtype=np.float64),
        "high": np.empty(8, dtype=np.float64),
        "low": np.empty(8, dtype=np.float64),
        "close": np.empty(8, dtype=np.float64),
        "volume": np.empty(8, dtype=np.float64),
        "_len": np.array([5], dtype=np.int64),
    }
    b = rows[("SPY", "1h")]
    b["timestamp"][:5] = base_ts
    for f in ("open", "high", "low", "close"):
        b[f][:5] = [100.0, 101.0, 102.0, 103.0, 104.0]
    b["volume"][:5] = [1000.0] * 5
    return rows


def _store() -> CandleStore:
    store = CandleStore(_rows(), htf_intervals=frozenset({"4h"}))
    # Cursor at the first base bar of the SECOND bucket: bucket 14:00 is
    # complete, bucket 18:00 is forming.
    store.advance(_ts("2025-01-06 18:00"))
    return store


def test_candle_store_htf_hides_forming_bucket() -> None:
    """REGRESSION: every read path hides the forming HTF bucket.

    Pre-fix the store truncated by timestamp only, so the 18:00 bucket (carrying
    18:00..21:00's aggregate, close 107) was visible at the 18:00 cursor. Covers
    ``cursor_count``, ``__getitem__``/``get``, ``latest``, ``count``, and
    iteration (``items``/``values``).
    """
    store = _store()

    # cursor_count: 2 rows match <= 18:00, but only 1 bucket has completed.
    assert store.cursor_count("SPY", "4h") == 1
    assert store.cursor_count("SPY", "1h") == 5  # base interval: unchanged

    # __getitem__ / get build a DataFrame truncated to completed buckets only.
    df = store[("SPY", "4h")]
    assert len(df) == 1
    assert float(df["close"].iloc[-1]) == 103.0  # not 107.0
    got = store.get(("SPY", "4h"))
    assert got is not None and len(got) == 1

    # latest / count: the forming bucket is not the "latest".
    assert store.latest("SPY", "4h") == 103.0  # not 107.0
    assert store.count("SPY", "4h") == 1

    # iteration / items / values all honour the same truncation.
    assert len(dict(store.items())[("SPY", "4h")]) == 1
    assert len(list(store.values())[0]) == 1


def test_candle_store_base_interval_unlagged() -> None:
    """CHARACTERISATION: the base interval keeps its exact prior behaviour.

    With the same cursor, the base key returns every row up to the cursor with no
    completion lag — proving the HTF change did not touch the base path.
    """
    store = _store()

    assert store.cursor_count("SPY", "1h") == 5
    assert len(store[("SPY", "1h")]) == 5
    assert float(store[("SPY", "1h")]["close"].iloc[-1]) == 104.0
