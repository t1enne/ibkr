"""HTF (higher-timeframe) reads through TaContext.

Methodology: build a small hourly base feed spanning one session, resample it
with ``build_htf_feeds``, feed both the base frame and the HTF mapping into a
``TaContext``, and bind a ``CandleStore`` whose cursor we advance by hand. Each
test asserts the lookahead contract: an HTF read exposes COMPLETED buckets only,
and an HTF indicator is cached independently of the base interval's.

Regression vs characterisation is stated in each test's docstring.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from src.bt.engine.candle_store import CandleStore, CandleRows
from src.bt.engine.utils import build_htf_feeds
from src.bt.strategies.ta_context import TaContext


def _ts(s: str) -> pd.Timestamp:
    """Parse a string timestamp into a typed ``pd.Timestamp``."""
    return cast(pd.Timestamp, pd.Timestamp(s))


def _hourly_df(sym: str = "SPY", n_bars: int = 8) -> pd.DataFrame:
    """Base feed: ``n_bars`` hourly bars in one session, close == 100 + index."""
    idx = pd.date_range("2025-01-06 14:00", periods=n_bars, freq="1h")
    data = {
        (sym, "open"): [100.0 + i for i in range(n_bars)],
        (sym, "high"): [101.0 + i for i in range(n_bars)],
        (sym, "low"): [99.0 + i for i in range(n_bars)],
        (sym, "close"): [100.0 + i for i in range(n_bars)],
        (sym, "volume"): [1000.0] * n_bars,
    }
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _rows_from_frames(
    base: pd.DataFrame, htf: dict[str, pd.DataFrame], sym: str
) -> CandleRows:
    """Build a CandleRows accumulator from the same frames the engine would feed.

    Base rows are the finite-close base bars; HTF rows are the resampled buckets.
    """
    rows: CandleRows = {}

    def _add(key: tuple[str, str], frame: pd.DataFrame) -> None:
        n = len(frame)
        cols = {
            "timestamp": np.empty(max(n, 1), dtype="datetime64[ms]"),
            "open": np.empty(max(n, 1), dtype=np.float64),
            "high": np.empty(max(n, 1), dtype=np.float64),
            "low": np.empty(max(n, 1), dtype=np.float64),
            "close": np.empty(max(n, 1), dtype=np.float64),
            "volume": np.empty(max(n, 1), dtype=np.float64),
            "_len": np.array([n], dtype=np.int64),
        }
        cols["timestamp"][:n] = frame.index.to_numpy()
        for f in ("open", "high", "low", "close", "volume"):
            cols[f][:n] = frame[f].to_numpy()
        rows[key] = cols

    base_fields = pd.DataFrame(
        {
            f: base[(sym, f)].to_numpy()
            for f in ("open", "high", "low", "close", "volume")
        },
        index=base.index,
    )
    _add((sym, "1h"), base_fields)
    for freq, feed in htf.items():
        sym_feed = feed.xs(sym, level="symbol")
        _add((sym, freq), sym_feed)
    return rows


def _context(base: pd.DataFrame, htf: dict[str, pd.DataFrame], sym: str) -> TaContext:
    ta = TaContext.from_data(base, [sym], "1h", htf_feeds=htf)
    store = CandleStore(_rows_from_frames(base, htf, sym), htf_intervals=frozenset(htf))
    ta.bind(store)
    return ta


def test_htf_series_hides_forming_bucket() -> None:
    """REGRESSION: an HTF view mid-bucket excludes the current (forming) bucket.

    Pre-fix the HTF read was blocked entirely (``_series`` raised), so this
    exercises new behaviour; more importantly it fails any implementation that
    exposes the forming bucket's future-inclusive OHLC at its start label.
    """
    base = _hourly_df()
    htf = build_htf_feeds(base, ["4h"])
    ta = _context(base, htf, "SPY")
    store = cast(CandleStore, ta._store)

    # Cursor at the first base bar of the second 4h bucket: the first bucket is
    # complete, the second is forming.
    store.advance(_ts("2025-01-06 18:00"))
    view = ta.close("SPY", interval="4h")
    assert len(view) == 1
    # The completed bucket's close is the close of its last base bar (17:00).
    assert float(view[-1]) == 103.0
    # The forming bucket's future-inclusive close (from 21:00) must not leak.
    assert float(view[-1]) != 107.0


def test_htf_lengther_lags_one_bucket() -> None:
    """REGRESSION: HTF visible count == store cursor_count - 1 (one bucket lag).

    Pre-fix the count was the naive ``cursor_count`` (too many), so the forming
    bucket was counted. Here the visible length must trail the store's raw count
    by exactly one bucket at every cursor inside the session.
    """
    base = _hourly_df()
    htf = build_htf_feeds(base, ["4h"])
    ta = _context(base, htf, "SPY")
    store = cast(CandleStore, ta._store)

    # Cursor inside the second (last) bucket 18:00..21:00: raw count 2, visible 1.
    store.advance(_ts("2025-01-06 19:00"))
    cols = store._rows[("SPY", "4h")]
    raw = int(
        np.searchsorted(
            cols["timestamp"][: int(cols["_len"][0])],
            np.datetime64(pd.Timestamp("2025-01-06 19:00").to_datetime64()),
            side="right",
        )
    )
    assert raw == 2
    assert store.cursor_count("SPY", "4h") == 1
    assert len(ta.close("SPY", interval="4h")) == 1


def test_cache_key_is_interval_scoped() -> None:
    """REGRESSION: a 4h EMA and a 1h EMA of the same period are distinct arrays.

    Pre-fix ``interval`` was folded into nothing (the key was ``(kind, sym,
    args)``) and HTF was refused, so this both proves the new key shape and that
    two intervals cost two computes.
    """
    base = _hourly_df()
    htf = build_htf_feeds(base, ["4h"])
    ta = _context(base, htf, "SPY")

    base_ema = ta.ema("SPY", 2)
    after_base = ta.compute_count
    htf_ema = ta.ema("SPY", 2, interval="4h")
    after_htf = ta.compute_count

    assert after_base == 1
    assert after_htf == 2  # distinct key => a second compute
    assert not np.array_equal(base_ema._values, htf_ema._values)

    # Re-reading either key is a cache hit (no further compute).
    ta.ema("SPY", 2)
    ta.ema("SPY", 2, interval="4h")
    assert ta.compute_count == 2


def test_series_raises_on_unknown_interval() -> None:
    """REGRESSION: an unknown interval raises and lists the available intervals.

    Pre-fix any non-base interval raised a generic "only serves the signal
    interval" error; the contract now requires the available intervals be named.
    """
    base = _hourly_df()
    htf = build_htf_feeds(base, ["4h"])
    ta = _context(base, htf, "SPY")

    try:
        ta.close("SPY", interval="1d")
    except KeyError as exc:
        assert "1d" in str(exc)
        assert "1h" in str(exc) and "4h" in str(exc)
    else:  # pragma: no cover - the raise is the contract
        raise AssertionError("expected KeyError for unknown interval")


def test_htf_alignment_feed_rows_match_store_rows() -> None:
    """CHARACTERISATION: HTF feed rows equal the store's HTF rows, same order.

    Both derive from the one ``build_htf_feeds`` mapping, so this holds before
    and after the cursor changes; it guards against a future divergence.
    """
    base = _hourly_df()
    htf = build_htf_feeds(base, ["4h"])
    ta = _context(base, htf, "SPY")

    feed = ta._complete[("SPY", "4h")]
    sym_feed = htf["4h"].xs("SPY", level="symbol")
    assert len(feed["close"]) == len(sym_feed)
    assert np.allclose(feed["close"], sym_feed["close"].to_numpy())
    assert np.allclose(feed["high"], sym_feed["high"].to_numpy())
