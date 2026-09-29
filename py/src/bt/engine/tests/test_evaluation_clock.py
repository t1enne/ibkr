"""Tests for the evaluation-clock guard.

``symbols[-1]`` IS the run's evaluation clock: the engine fires ``on_candle``
only on the tail symbol's candle, so a tail whose history starts inside the
trading window silently truncates the whole run. These tests pin the guard that
rejects such a config, plus the edge cases (empty/single symbol) and the
pre-existing benchmark-tail regression.
"""

from dataclasses import dataclass

import pandas as pd
import pytest

from src.bt.engine.backtest import (
    Backtest,
    _assert_evaluation_clock_covers_window,
    run,
)
from src.bt.strategies.dsl import StrategyContext, strategy
from src.bt.types import StrategyConfig


@dataclass
class _FixtureMod:
    """Module-shaped holder: ``.on_candle`` is a ``@strategy`` adapter."""

    on_candle: object


@strategy(bars="1d")
def _noop_on_candle(ctx: StrategyContext):
    """Runs to completion without emitting — the guard, not the strategy, is
    under test."""
    return None


def _cfg(symbols: list[str], trading_start: str = "2025-01-03") -> StrategyConfig:
    return StrategyConfig(
        name="clock",
        strategy_type="momentum_compression_breakout_dsl",  # registered
        symbols=symbols,
        initial_capital=10000.0,
        commission=0.5,
        warmup="0d",
        trading_start=trading_start,
        trading_end="2025-12-31",
        bars=["1d"],
        strategy_params={},
    )


def _frame(symbol_starts: dict[str, str], n: int = 8) -> pd.DataFrame:
    """MultiIndex feed where each symbol's own (non-NaN) bars begin at its
    declared start, outer-joined and NaN-padded like ``load_candles``."""
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    data: dict = {}
    for sym, start in symbol_starts.items():
        live = idx >= pd.Timestamp(start)
        for field, value in (
            ("open", 100.0),
            ("high", 105.0),
            ("low", 95.0),
            ("close", 102.0),
            ("volume", 1000.0),
        ):
            col = [value if live[i] else float("nan") for i in range(n)]
            data[(sym, field)] = col
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_full_parity_tail_passes():
    """Every symbol starts before trading_start -> clock covers the window."""
    frame = _frame({"AAPL": "2025-01-01", "MSFT": "2025-01-01", "PATH": "2025-01-01"})
    _assert_evaluation_clock_covers_window(frame, _cfg(["AAPL", "MSFT", "PATH"]))


def test_multiday_stagger_does_not_trip():
    """A few days' spread in first bars (the real 2019-11-06 vs 11-15 case) is
    normal and must pass as long as the tail starts at/before trading_start."""
    frame = _frame({"AAPL": "2025-01-01", "MSFT": "2025-01-02", "PATH": "2025-01-03"})
    _assert_evaluation_clock_covers_window(frame, _cfg(["AAPL", "MSFT", "PATH"]))


def test_late_tail_raises():
    """Tail's first bar lands AFTER trading_start -> clock truncates the run."""
    frame = _frame({"AAPL": "2025-01-01", "MSFT": "2025-01-01", "PATH": "2025-01-05"})
    cfg = _cfg(["AAPL", "MSFT", "PATH"], trading_start="2025-01-03")
    with pytest.raises(AssertionError) as exc:
        _assert_evaluation_clock_covers_window(frame, cfg)
    msg = str(exc.value)
    assert "PATH" in msg  # names the offending tail symbol
    assert "2025-01-05" in msg  # its first-bar date
    assert "trading_start" in msg
    assert "reorder config.symbols" in msg  # the remedy
    assert "AAPL" in msg  # comparable symbols


def test_tail_all_nan_column_raises():
    """A tail symbol present as a column but with zero real bars is the extreme
    truncation — the strategy would never be evaluated at all."""
    frame = _frame({"AAPL": "2025-01-01", "MSFT": "2025-01-01"})
    # Add a PATH column that exists but has no bars (all-NaN), like load_candles
    # would emit for a symbol with no rows in range.
    for field in ("open", "high", "low", "close", "volume"):
        frame[("PATH", field)] = float("nan")
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    with pytest.raises(AssertionError, match="no candles in the loaded feed"):
        _assert_evaluation_clock_covers_window(frame, _cfg(["AAPL", "MSFT", "PATH"]))


def test_tail_column_absent_is_out_of_scope():
    """Tail symbol in ``symbols`` but not a column at all is a different failure
    (surfaced downstream); the clock guard returns rather than misreport it."""
    frame = _frame({"AAPL": "2025-01-01", "MSFT": "2025-01-01"})
    _assert_evaluation_clock_covers_window(frame, _cfg(["AAPL", "MSFT", "PATH"]))


def test_single_symbol_config_passes():
    """One symbol has nothing to be late relative to — never trips."""
    frame = _frame({"PATH": "2025-01-05"})
    _assert_evaluation_clock_covers_window(frame, _cfg(["PATH"]))


def test_empty_symbols_passes():
    """Empty symbols -> no clock; guard is a no-op."""
    _assert_evaluation_clock_covers_window(pd.DataFrame(), _cfg([]))


def test_run_end_to_end_raises_on_late_tail():
    """The guard is wired into ``run`` — a late tail aborts before any backtest."""
    frame = _frame({"AAPL": "2025-01-01", "MSFT": "2025-01-01", "PATH": "2025-01-05"})
    bt = Backtest(_cfg(["AAPL", "MSFT", "PATH"], trading_start="2025-01-03"))
    with pytest.raises(AssertionError, match="evaluation clock"):
        run(bt, frame, _FixtureMod(_noop_on_candle))


def test_run_end_to_end_full_history_tail_runs():
    """A full-history tail runs straight through — the guard is not a blanket
    rejection."""
    frame = _frame({"AAPL": "2025-01-01", "MSFT": "2025-01-01", "PATH": "2025-01-01"})
    bt = Backtest(_cfg(["AAPL", "MSFT", "PATH"], trading_start="2025-01-03"))
    results = run(bt, frame, _FixtureMod(_noop_on_candle))
    assert results is not None


def test_benchmark_tail_regression_still_raises():
    """Pre-existing benchmark-tail guard must still fire (regression)."""
    from src.bt.engine.backtest import _assert_benchmark_symbols_last

    cfg = StrategyConfig(
        name="bm",
        strategy_type="momentum_compression_breakout_dsl",
        symbols=["SPY", "AAPL"],  # benchmark SPY not at the tail
        initial_capital=10000.0,
        commission=0.5,
        warmup="0d",
        trading_start="2025-01-03",
        trading_end="2025-12-31",
        bars=["1d"],
        strategy_params={},
        benchmark_symbols=["SPY"],
    )
    with pytest.raises(AssertionError, match="benchmark symbols"):
        _assert_benchmark_symbols_last(cfg)
