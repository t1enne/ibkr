"""Tests for backtest engine — critical paths only."""

from dataclasses import dataclass
import pandas as pd
from src.bt.engine.backtest import Backtest, run_backtest, candle_generator
from src.bt.exchange import default_exchange
from src.bt.risk.handlers import default_risk_handler
from src.bt.types import StrategyConfig
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
            default_exchange(),
            default_risk_handler(),
            strategy_mod=cl,
        )
