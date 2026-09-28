"""BT module — backtesting engine."""

from src.bt.cli import bt_group

from src.bt.strategies import init_strat
from src.bt.engine.backtest import Backtest, candle_generator, run_backtest, run
from src.bt.engine.handlers import (  # noqa: F401
    ExecutionHandler,
    RiskHandler,
    default_execution_handler,
    default_risk_handler,
)
from src.bt.metrics import get_backtest_results_analysis, build_symbol_attribution
from src.bt.types import StrategyConfig, PortfolioResult

import json
import asyncio

import pandas as pd


def load_strategy(path: str) -> StrategyConfig:
    with open(path, "r") as f:
        data = json.load(f)
    return StrategyConfig(**data)


def warmup_load_start(
    config: StrategyConfig,
    trading_start: pd.Timestamp,
) -> pd.Timestamp:
    """Feed start for a window: ``trading_start`` minus the config's warmup span.

    A calendar subtraction — the ``warmup`` duration is a span, not a bar
    count, so no per-symbol history probe is needed. Symbols whose history
    starts later simply warm with the bars they have (the engine walks
    whatever is in the feed); their own bar-count readiness gate keeps them
    out of the market until they are ready.
    """
    from src.bt.warmup import warmup_start

    return warmup_start(trading_start, config.warmup)


def run_backtest_results(strategy_conf: StrategyConfig):
    """Run a strategy backtest and return the structured BacktestResults.

    Returns the full ``BacktestResults`` (metrics, trades, equity curve,
    benchmark curves) — not a rendered string. The CLI renders text from
    this; structured output uses it directly with no text round-trip.
    """
    from src.bt.data_feed import load_candles

    bt = Backtest(strategy_conf)
    df = load_candles(
        strategy_conf.symbols,
        warmup_load_start(strategy_conf, bt.window.test_start),
        bt.window.test_end,
        strategy_conf.bars[0],
    )
    strat_mod = init_strat(strategy_conf.strategy_type)
    return run(bt, df, strat_mod=strat_mod)


async def backtest_async(strategy_conf: StrategyConfig) -> str:
    """Backtest a trading strategy (async version). Returns text report."""
    results = run_backtest_results(strategy_conf)
    return get_backtest_results_analysis(
        results.pf, benchmark_curves=results.benchmark_curves
    )


def backtest(strategy_conf: StrategyConfig) -> str:
    """Backtest a trading strategy (sync version)."""
    return asyncio.run(backtest_async(strategy_conf))


__all__ = [
    "bt_group",
    "Backtest",
    "candle_generator",
    "run_backtest",
    "run",
    "load_strategy",
    "backtest",
    "backtest_async",
    "run_backtest_results",
    "warmup_load_start",
    "get_backtest_results_analysis",
    "build_symbol_attribution",
    "StrategyConfig",
    "PortfolioResult",
    "init_strat",
]
