"""Shared IS/OOS window-running helpers.

`bt split` and `bt optimize` both evaluate strategies over IS/OOS windows
drawn from a shared candle feed. This module owns the per-window mechanics so
the two entry points stay in sync:

- `run_window` slices the feed to the window's tradable end (no look-ahead) and
  reuses pre-loaded benchmark candles. Strategy isolation is per-run: the engine
  mints a fresh ``ctx.shared`` for every ``run`` / window, so no cross-window
  state reset is needed.
- `window_has_data` guards against windows that fall entirely in a data gap.

Pure window math lives here (test-friendly); fold builders live in `split.py`
and optimization orchestration in `optimize.py`.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from src.bt.types import StrategyConfig, PortfolioResult


def window_df(
    data: pd.DataFrame,
    trading_end: pd.Timestamp,
) -> pd.DataFrame:
    """Slice the shared feed down to one window's tradable range.

    Drops everything past `trading_end` so the engine never processes future
    data (which leaked out-of-window closes, model updates, and marks across
    the boundary). Keeps the head (`data` already starts at the warmup/train
    start), so the model still warms up on prior history before the window.
    """
    return data.loc[:trading_end]


def window_has_data(
    data: pd.DataFrame, trading_start: pd.Timestamp, trading_end: pd.Timestamp
) -> bool:
    """True if the tradable window region contains at least one candle.

    `window_df` keeps the warmup head, so checking the sliced frame alone
    would not catch a window that falls entirely in a data gap. Inspect the
    window's own timestamp range instead. Uses ``len(data)`` (not ``.empty``)
    because a rows-but-no-columns frame reports ``.empty == True``.
    """
    if len(data) == 0:
        return False
    return bool(((data.index >= trading_start) & (data.index <= trading_end)).any())


def run_window(
    cfg: StrategyConfig,
    strat_mod,
    data: pd.DataFrame,
    bm_df: pd.DataFrame | None,
    trading_start: pd.Timestamp,
    trading_end: pd.Timestamp,
) -> PortfolioResult:
    """Run one IS or OOS window by overriding the config's trading window.

    Data is loaded once per split, then sliced per window so the engine only
    sees candles up to `trading_end` — no post-window data (fixes out-of-window
    trade closes, model-updater leakage, and wasted full-feed iteration).

    Benchmark candles are loaded once and sliced per window too, avoiding a
    DB reload for every IS/OOS window.
    """
    from src.bt.engine.backtest import Backtest, build_benchmark_curves, run

    window_cfg = replace(
        cfg,
        trading_start=trading_start.isoformat(),
        trading_end=trading_end.isoformat(),
    )
    bt = Backtest(window_cfg)
    bm_curves = (
        build_benchmark_curves(bm_df, cfg, trading_start, trading_end)
        if bm_df is not None
        else None
    )
    results = run(
        bt,
        window_df(data, trading_end),
        strat_mod=strat_mod,
        benchmark_curves=bm_curves,
    )
    return results.pf
