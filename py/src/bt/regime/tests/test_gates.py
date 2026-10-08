"""Tests for src.bt.regime.gates — the strategy-facing trend gate layer."""

from __future__ import annotations

import pandas as pd

from src.bt.regime.gates import (
    TrendGate,
    above_sma,
    series_above_sma,
    sma_trend,
    weekly_above_sma,
)
from src.bt.state import create_initial_backtest_state
from src.bt.engine.utils import merge_bt_state


def _mk_df(closes: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": closes * 0.999,
            "high": closes * 1.002,
            "low": closes * 0.998,
            "close": closes,
            "volume": 1_000_000,
        },
        index=closes.index,
    )


def _state(symbols: list[str], closes_by_symbol: dict[str, pd.Series], bar: str = "1d"):
    state = create_initial_backtest_state(
        symbols=symbols,
        initial_capital=10_000,
        start_timestamp=closes_by_symbol[symbols[0]].index[0],
    )
    candles = {(sym, bar): _mk_df(closes_by_symbol[sym]) for sym in symbols}
    return merge_bt_state(state, dict(candles=candles))


def _rising(n: int = 300, factor: float = 1.0005) -> pd.Series:
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.Series(50.0 * (factor ** pd.Series(range(n), index=idx)), index=idx)


# ------------------------------------------------ TrendGate policy ----


def test_trend_gate_label_mapping():
    assert TrendGate("BULL").bull and not TrendGate("BULL").bear
    assert TrendGate("BEAR").bear and not TrendGate("BEAR").bull
    assert TrendGate("RANGE").range
    assert not TrendGate(None).known


def test_allows_long_range_opt_in():
    assert not TrendGate("RANGE").allows_long()
    assert TrendGate("RANGE").allows_long(allow_range=True)


def test_hostile_to():
    assert TrendGate("BEAR").hostile_to("long")
    assert not TrendGate("BEAR").hostile_to("short")
    assert TrendGate("BULL").hostile_to("short")
    assert TrendGate("RANGE").hostile_to("long", allow_range=True)
    assert not TrendGate("RANGE").hostile_to("long")
    assert not TrendGate(None).hostile_to("long")


# ------------------------------------------------ sma_trend ----


def test_sma_trend_bull():
    st = _state(["A"], {"A": _rising()})
    assert sma_trend(st, "A", fast=50, slow=200, bar="1d").bull


def test_sma_trend_bear():
    st = _state(["A"], {"A": _rising(factor=0.999)})
    assert sma_trend(st, "A", fast=50, slow=200, bar="1d").bear


def test_sma_trend_unknown_during_warmup():
    st = _state(["A"], {"A": _rising(n=50)})
    gate = sma_trend(st, "A", fast=20, slow=200, bar="1d")
    assert gate.label is None


# ------------------------------------------------ above_sma ----


def test_above_sma_rising():
    st = _state(["A"], {"A": _rising(n=300)})
    assert above_sma(st, "A", window=50, bar="1d") is True


def test_above_sma_missing_symbol_false():
    st = _state(["A"], {"A": _rising(n=300)})
    assert above_sma(st, "NOPE", window=50, bar="1d") is False


def test_series_above_sma_nan_guard():
    # NaN SMA → False (not above), not an error.
    assert series_above_sma(pd.Series([float("nan")] * 10), 5) is False


def test_series_above_sma_short_series():
    assert series_above_sma(pd.Series([1.0] * 3), 50) is False


# ------------------------------------------------ weekly_above_sma ----


def test_weekly_above_sma_rising():
    st = _state(["A"], {"A": _rising(n=400)})
    cache: dict = {}
    assert weekly_above_sma(st, "A", window=50, bar="1d", cache=cache) is True
    assert weekly_above_sma(st, "A", window=50, bar="1d", cache=cache) is True  # cached


def test_weekly_above_sma_falling():
    st = _state(["A"], {"A": _rising(n=400, factor=0.999)})
    assert weekly_above_sma(st, "A", window=50, bar="1d") is False


def test_weekly_above_sma_short_history():
    st = _state(["A"], {"A": _rising(n=100)})
    assert weekly_above_sma(st, "A", window=50, bar="1d") is False


# ------------------------------------------------ gates read state.candles ----
