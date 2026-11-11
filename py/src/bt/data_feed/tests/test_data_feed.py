"""Tests for the data feed gap-integrity guard.

``load_candles`` must abort (raise) when a loaded symbol contains a
discontinuity > 96h between consecutive bars — otherwise rolling/indicator
math treats the two sides of the hole as adjacent and silently corrupts the
backtest. ``detect_gaps`` is the pure detector behind that guard.
"""

from __future__ import annotations

import pandas as pd
import pytest

import src.bt.data_feed as feed
from src.bt.data_feed import (
    DataIntegrityError,
    GapBreak,
    detect_gaps,
    load_candles,
)


def _ts(v: str) -> pd.Timestamp:
    ts = pd.Timestamp(v)
    assert isinstance(ts, pd.Timestamp)
    return ts


def _frame(symbols: list[str], template):
    """Drop the per-symbol rows in ``template`` into a MultiIndex-column frame."""
    dfs = {}
    for s in symbols:
        rows = template(s)
        dfs[s] = pd.DataFrame(
            {
                "open": [1.0] * len(rows),
                "high": [1.0] * len(rows),
                "low": [1.0] * len(rows),
                "close": [1.0] * len(rows),
                "volume": [1.0] * len(rows),
            },
            index=pd.DatetimeIndex(rows),
        )
    return pd.concat(list(dfs.values()), axis=1, keys=dfs.keys(), sort=False)


def _hourly(*stamps: str) -> list[str]:
    return list(pd.Series(stamps).astype("datetime64[ns]"))


def _two_bars(a: str, b: str) -> list[str]:
    return _hourly(a, b)


# ── detect_gaps ────────────────────────────────────────────────────────
def test_detect_gaps_clean_series():
    def template(_s: str) -> list[str]:
        return _hourly("2024-01-01 00:00", "2024-01-01 23:00", "2024-01-02 12:00")

    df = _frame(["A"], template)
    report = detect_gaps(df, ["A"])
    assert report == {}


def test_detect_gaps_reports_gap_over_threshold():
    # 120h between bars 2 and 3 — above the 96h DEFAULT_MAX_GAP.
    def template(_s: str) -> list[str]:
        return _hourly("2024-01-01 00:00", "2024-01-02 00:00", "2024-01-07 00:00")

    df = _frame(["A"], template)
    report = detect_gaps(df, ["A"])
    assert "A" in report
    breaks = report["A"]
    assert len(breaks) == 1
    b = breaks[0]
    assert isinstance(b, GapBreak)
    assert b.prev_ts == pd.Timestamp("2024-01-02 00:00")
    assert b.next_ts == pd.Timestamp("2024-01-07 00:00")
    assert b.duration == pd.Timedelta(hours=120)


def test_detect_gaps_ignores_market_closures():
    # Christmas: 12/24 half-day close -> 12/28 open is 4d0.5h (> 96h), but every
    # day in between is a weekend/holiday -- that is a market closure, not a
    # hole in the data.
    def template(_s: str) -> list[str]:
        return _hourly("2024-12-24 14:30", "2024-12-24 18:00", "2024-12-28 15:00")

    assert detect_gaps(_frame(["A"], template), ["A"]) == {}


def test_a_us_listing_reports_a_hole_an_italian_listing_does_not():
    # Easter 2026: Good Friday (4/3) shut in both, Easter Monday (4/6) shut only
    # in Milan. So 4/2 -> 4/7 skips one New York session and NO Milan one: the
    # same frame is a real hole for the US series and complete for the BVME one.
    def template(_s: str) -> list[str]:
        return _hourly("2026-04-02 15:00", "2026-04-07 07:00")

    df = _frame(["US", "IT"], template)
    report = detect_gaps(df, ["US", "IT"], markets={"US": "SMART", "IT": "BVME"})

    assert "US" in report, "a skipped New York session is missing data"
    assert "IT" not in report, "both Milan closures are holidays, not a hole"


# ── load_candles guard ─────────────────────────────────────────────────
def _load_df(*, hourly_spans: dict):
    def factory(symbol):
        span = hourly_spans[symbol]
        return _hourly(*span)

    # get_local_candles yields a plain OHLCV frame (single-symbol columns).
    # xs(symbol, axis=1) drops the (symbol, field) MultiIndex level 0.
    def fake_get_local_candles(symbol, start=None, end=None, bar="1h"):
        return _frame([symbol], factory).xs(symbol, axis=1)

    return fake_get_local_candles


def test_load_candles_raises_on_gap(monkeypatch):
    monkeypatch.setattr(
        feed,
        "get_local_candles",
        _load_df(
            hourly_spans={
                "A": ("2024-01-01 00:00", "2024-01-02 00:00"),
                "B": ("2024-01-01 00:00", "2024-01-06 00:00"),  # 120h gap
            }
        ),
    )
    with pytest.raises(DataIntegrityError) as exc:
        load_candles(["A", "B"], _ts("2024-01-01"), _ts("2024-01-07"), "1h")
    assert "B" in str(exc.value)
    assert "gap" in str(exc.value)


def test_load_candles_passes_when_clean(monkeypatch):
    monkeypatch.setattr(
        feed,
        "get_local_candles",
        _load_df(
            hourly_spans={
                "A": ("2024-01-01 00:00", "2024-01-02 00:00"),
                "B": ("2024-01-01 00:00", "2024-01-01 23:00"),
            }
        ),
    )
    df = load_candles(["A", "B"], _ts("2024-01-01"), _ts("2024-01-02"), "1h")
    assert set(df.columns.get_level_values(0)) == {"A", "B"}


def test_load_candles_max_gap_disables_guard(monkeypatch):
    monkeypatch.setattr(
        feed,
        "get_local_candles",
        _load_df(
            hourly_spans={
                "A": ("2024-01-01 00:00", "2024-01-07 00:00"),  # 120h gap
            }
        ),
    )
    # Large sentinel skips the check entirely.
    df = load_candles(
        ["A"],
        _ts("2024-01-01"),
        _ts("2024-01-08"),
        "1h",
        max_gap=pd.Timedelta.max,
    )
    assert len(df["A"].index) == 2


def test_load_candles_uses_the_stored_venue_of_each_symbol(monkeypatch):
    """The venue is resolved at the load boundary, from the ``symbol`` row.

    ``IT`` holds only the two bars straddling a Milan-only closure, so the load
    must pass on ITS calendar even though the same two stamps are a hole for a
    New York series.
    """

    def template(_s: str) -> list[str]:
        return _hourly("2026-04-02 15:00", "2026-04-07 07:00")

    def fake_get_local_candles(symbol, start=None, end=None, bar="1h"):
        return _frame([symbol], template).xs(symbol, axis=1)

    monkeypatch.setattr(feed, "get_local_candles", fake_get_local_candles)
    monkeypatch.setattr(
        feed, "market_for_ticker", lambda symbol: {"IT": "BVME"}.get(symbol)
    )

    df = load_candles(["IT"], _ts("2026-04-02"), _ts("2026-04-07"), "1h")

    assert len(df["IT"].index) == 2
