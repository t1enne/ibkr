"""Tests for backtest engine — critical paths only."""

import pandas as pd
import pytest
from src.bt.engine.backtest import Backtest, run_backtest, candle_generator
from src.bt.engine.handlers import default_execution_handler, default_risk_handler
from src.bt.types import StrategyConfig
from src.utils import parse_timestamp


def _make_multi_idx_df(symbols: list[str], n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    data = {}
    for sym in symbols:
        data.update(
            {
                (sym, "open"): [100] * n,
                (sym, "high"): [105] * n,
                (sym, "low"): [95] * n,
                (sym, "close"): [102] * n,
                (sym, "volume"): [1000] * n,
            }
        )
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_candle_generator_multi_symbol():
    df = _make_multi_idx_df(["AAPL", "GOOGL"])
    candles = list(candle_generator(df, ["AAPL", "GOOGL"]))
    assert len([c for c in candles if c.symbol == "AAPL"]) == 5
    assert len([c for c in candles if c.symbol == "GOOGL"]) == 5


def _stub_strategy():
    """Power-user (non-DSL) strategy_mod; fires one long on the 3rd on_candle."""
    from src.bt.state import ActionType, TradeSignal

    class _Mod:
        def __init__(self) -> None:
            self.calls = 0

        def on_candle(self, state, candle, params):
            self.calls += 1
            if self.calls == 3:
                return [
                    TradeSignal(
                        action=ActionType.long,
                        symbol=candle.symbol,
                        timestamp=candle.timestamp,
                        price=candle.close,
                        reason="test long",
                    )
                ]
            return []

    return _Mod()


def _daily_df(symbols: list[str], n: int = 5) -> pd.DataFrame:
    """MultiIndex-column frame over ``n`` daily bars aligned to a 1d config."""
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    data: dict = {}
    for sym in symbols:
        data.update(
            {
                (sym, "open"): [100] * n,
                (sym, "high"): [105] * n,
                (sym, "low"): [95] * n,
                (sym, "close"): [102] * n,
                (sym, "volume"): [1000] * n,
            }
        )
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _cfg(symbols):
    """1d config tied to a registered strategy type with a 2025 trading window
    over a 5-day daily feed (an injected strategy_mod overrides the type)."""
    return StrategyConfig(
        name="test",
        strategy_type="momentum_compression_breakout_dsl",  # registered
        symbols=symbols,
        initial_capital=10000.0,
        commission=0.5,
        training_start="2024-01-01",
        training_end="2024-12-31",
        trading_start="2025-01-01",
        trading_end="2025-12-31",
        bars=["1d"],
        strategy_params={},
    )


def test_signal_observer_fires_once_per_fresh_signal():
    """The observer (screen hook) must capture each fresh strategy emission
    exactly once — never doubled by _execute_pending re-draining pending buckets."""
    cfg = _cfg(["AAPL"])
    bt = Backtest(cfg)
    # Config-object generator (daily bars) so the engine treats candles as base.
    gen = candle_generator(_daily_df(["AAPL"]), bt.config)

    mod = _stub_strategy()
    seen = []
    results, state = run_backtest(
        bt,
        gen,
        default_execution_handler(),
        default_risk_handler(),
        strategy_mod=mod,
        signal_observer=lambda sig: seen.append(sig),
    )
    # Exactly the single emission the stub produced mid-run.
    assert len(seen) == 1
    assert seen[0].reason == "test long"


def test_signal_observer_none_is_behavior_neutral():
    """Default (no observer) must behave identically to an idle observer — the
    optional hook adds no branch to engine behavior when not collecting."""
    cfg = _cfg(["AAPL"])

    def _run(observer):
        bt = Backtest(cfg)
        gen = candle_generator(_daily_df(["AAPL"]), bt.config)
        _, st = run_backtest(
            bt,
            gen,
            default_execution_handler(),
            default_risk_handler(),
            strategy_mod=_stub_strategy(),
            signal_observer=observer,
        )
        return st.portfolio.cash

    base = _run(None)
    idle = _run(lambda _: None)
    assert idle == base


def test_run_backtest_no_crash():
    cfg = StrategyConfig(
        name="test",
        strategy_type="momentum_compression_breakout_dsl",
        symbols=["AAPL"],
        initial_capital=10000.0,
        commission=0.5,
        training_start="2024-01-01",
        training_end="2024-12-31",
        trading_start="2025-01-01",
        trading_end="2025-12-31",
        bars=["1d"],
        strategy_params={},
    )
    bt = Backtest(cfg)
    gen = candle_generator(_daily_df(["AAPL"]), bt.config)
    results, state = run_backtest(
        bt, gen, default_execution_handler(), default_risk_handler()
    )
    assert results is not None
    assert state is not None
    assert state.portfolio is not None


def test_build_benchmark_curves_slices_and_normalizes():
    from src.bt.engine.backtest import build_benchmark_curves

    # MultiIndex (symbol, field) with rising close prices
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    data = {
        ("SPY", "close"): [100.0, 110.0, 120.0, 130.0, 140.0],
        ("SPY", "open"): [100.0] * 5,
    }
    bm_df = pd.DataFrame(data, index=idx)
    bm_df.columns = pd.MultiIndex.from_tuples(bm_df.columns)

    cfg = StrategyConfig(
        name="t",
        strategy_type="momentum_regime",
        symbols=["AAPL"],
        initial_capital=1000.0,
        commission=0.5,
        training_start="2025-01-01",
        training_end="2025-01-01",
        trading_start="2025-01-01",
        trading_end="2025-01-05",
        bars=["1d"],
        strategy_params={"position_size": 0.2, "stop_loss": 0.05, "take_profit": 0.1},
        benchmark_symbols=["SPY"],
    )
    curves = build_benchmark_curves(
        bm_df, cfg, parse_timestamp("2025-01-02"), parse_timestamp("2025-01-04")
    )
    assert "SPY" in curves
    # window-sliced to 3 points (2025-01-02..04) but normalized against the
    # in-window first close, so the first sampled point = initial_capital
    ser = curves["SPY"]
    assert len(ser) == 3
    assert ser.iloc[0] == pytest.approx(cfg.initial_capital)


def test_build_benchmark_curves_empty_when_fewer_than_two_points():
    from src.bt.engine.backtest import build_benchmark_curves

    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    data = {("SPY", "close"): [100.0, 101.0, 102.0], ("SPY", "open"): [100.0] * 3}
    bm_df = pd.DataFrame(data, index=idx)
    bm_df.columns = pd.MultiIndex.from_tuples(bm_df.columns)
    cfg = StrategyConfig(
        name="t",
        strategy_type="momentum_regime",
        symbols=["AAPL"],
        initial_capital=1000.0,
        commission=0.5,
        training_start="2025-01-01",
        training_end="2025-01-01",
        trading_start="2025-01-01",
        trading_end="2025-01-03",
        bars=["1d"],
        strategy_params={"position_size": 0.2, "stop_loss": 0.05, "take_profit": 0.1},
        benchmark_symbols=["SPY"],
    )
    # window has a single point -> <2 -> excluded
    curves = build_benchmark_curves(
        bm_df, cfg, parse_timestamp("2025-01-03"), parse_timestamp("2025-01-03")
    )
    assert "SPY" not in curves
