"""Tests for backtest engine — critical paths only."""

from dataclasses import dataclass
import pandas as pd
import pytest
from src.bt.engine.backtest import Backtest, run_backtest, candle_generator, run
from src.bt.engine.handlers import default_execution_handler, default_risk_handler
from src.bt.types import StrategyConfig
from src.utils import parse_timestamp
from src.bt.strategies.dsl import StrategyContext, strategy


@dataclass
class _FixtureMod:
    """Mirrors a real strategy module's engine-facing surface: ``.on_candle``
    is the ``@strategy`` adapter (carrying ``ctx_fn``). Strat passed straight to
    ``run``/``run_backtest`` as this holder, exactly as ``init_strat`` does for
    production modules."""

    on_candle: object


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


@strategy(bars="1d", stateful=True)
def _fixture_dsl_on_candle(ctx: StrategyContext):
    """Real ``@strategy`` fixture — DSL replacement for the removed plain
    ``_stub_strategy``. Fires one long on the 3rd on_candle dispatch using a
    stateful ``ctx.shared`` counter. Does NOT read ``ctx.params`` (the engine
    resolves params off ``config.strategy_type``'s registered type, not this
    fixture's), so observer/cash tests sidestep that mismatch entirely."""
    assert ctx.shared is not None  # stateful adapter binds a per-run holder
    n = ctx.shared.setdefault("_n", 0)
    ctx.shared["_n"] = n + 1
    if n == 2:  # 3rd dispatch (0-indexed counter)
        ctx.long(ctx.candle.symbol, size=0.05, reason="test long")


def _fixture_dsl():
    """DSL strategy_mod for the engine: the ``@strategy`` adapter wrapped in the
    module-shaped holder the engine consumes (``.on_candle`` = adapter)."""
    return _FixtureMod(_fixture_dsl_on_candle)


def _kit(cfg=None):
    """A ready-to-run ``Backtest`` over a 1-daily-bar AAPL feed."""
    cfg = cfg or _cfg(["AAPL"])
    return Backtest(cfg), _daily_df(["AAPL"])


def test_candle_generator_multi_symbol():
    df = _make_multi_idx_df(["AAPL", "GOOGL"])
    candles = list(candle_generator(df, ["AAPL", "GOOGL"]))
    assert len([c for c in candles if c.symbol == "AAPL"]) == 5
    assert len([c for c in candles if c.symbol == "GOOGL"]) == 5


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
        warmup="0d",
        trading_start="2025-01-01",
        trading_end="2025-12-31",
        bars=["1d"],
        strategy_params={},
    )


def test_signal_observer_fires_once_per_fresh_signal():
    """The observer (screen hook) must capture each fresh strategy emission
    exactly once — never doubled by _execute_pending re-draining pending buckets."""
    bt, data = _kit()
    seen = []
    run(
        bt,
        data,
        _fixture_dsl(),
        signal_observer=lambda sig: seen.append(sig),
    )
    # Exactly the single emission the fixture produced mid-run.
    assert len(seen) == 1
    assert seen[0].reason == "test long"


def test_signal_observer_none_is_behavior_neutral():
    """Default (no observer) must behave identically to an idle observer — the
    optional hook adds no branch to engine behavior when not collecting."""

    def _run(observer):
        bt, data = _kit()
        return run(
            bt,
            data,
            _fixture_dsl(),
            signal_observer=observer,
        ).final_state.portfolio.cash

    base = _run(None)
    idle = _run(lambda _: None)
    assert idle == base


def test_dsl_fixture_runs_stateful():
    """A real stateful DSL fixture runs to completion through run() without
    hand-rolled Ta wiring; the 3rd-candle long lands in the book."""
    bt, data = _kit()
    results = run(bt, data, _fixture_dsl())
    trades = results.final_state.portfolio.trades
    assert len(trades) == 1  # the single mid-run long, closed at finalize


def test_run_backtest_rejects_plain_strategy_mod():
    """A hand-rolled (non-@strategy) module handed to run_backtest raises;
    None stays a legal no-strategy loop."""
    from pytest import raises

    cfg = _cfg(["AAPL"])
    bt = Backtest(cfg)
    gen = candle_generator(_daily_df(["AAPL"]), bt.config)
    cl = type("_Mod", (object,), {"on_candle": lambda self, s, c, p: []})()
    with raises(TypeError):
        run_backtest(
            bt,
            gen,
            default_execution_handler(),
            default_risk_handler(),
            strategy_mod=cl,
        )


def test_run_backtest_no_crash():
    cfg = StrategyConfig(
        name="test",
        strategy_type="momentum_compression_breakout_dsl",
        symbols=["AAPL"],
        initial_capital=10000.0,
        commission=0.5,
        warmup="0d",
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
        warmup="0d",
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
        warmup="0d",
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
