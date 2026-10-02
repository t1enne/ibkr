"""Walk-forward parameter optimization.

Bridges `bt sweep` (tune params, whole window) and `bt split` (validate
locked params across IS/OOS). Per fold:

1. Sweep the param grid ON the fold's in-sample window — pick the combo
   with the best IS score (default: Sharpe).
2. Lock those params and run the fold's out-of-sample window with them.
3. Record both. OOS metrics are genuinely out-of-sample — they were never
   optimized against.

Answers the question `bt sweep` can't: "given I overfit a little per fold,
does the edge survive the next unseen window?" Mean OOS Sharpe + degradation
across folds summarise robustness. Tuning happens per fold (each fold's IS
ends before its OOS), so no lookahead: a fold's params never see its OOS.

Pure optimization logic lives here (test-friendly); engine wiring is
`run_optimize`. Reuses fold builders from `split.py` and grid/patch helpers
from `sweep.py`. Mirrors the repo's `pure.py` convention.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable

import pandas as pd

from src.bt.report_metrics import (
    metric_cells,
    metric_cells_from_dict,
    metric_dict,
    metric_labels,
)
from src.bt.split import TestFold
from src.bt.sweep import build_config, _flat_overrides, grid_combos
from src.bt.types import StrategyConfig, PortfolioResult
from src.bt.window import run_window, window_has_data


@dataclass(frozen=True)
class OptimizeResult:
    """One fold's IS-tuned params and their OOS outcome."""

    fold: TestFold
    best_params: dict[
        str, Any
    ]  # swept leaf values chosen on IS (dot-joined path -> value)
    is_metrics: dict[str, float | int]  # canonical metric_dict of the best IS combo
    oos: PortfolioResult

    def oos_metric(self, name: str) -> float:
        return float(getattr(self.oos, name))


def _metric_names() -> set[str]:
    return {f.name for f in fields(PortfolioResult)}


def _best_combo_on_window(
    cfg: StrategyConfig,
    merged_patches: list[dict[str, Any]],
    strat_mod,
    data: pd.DataFrame,
    bm_df: pd.DataFrame | None,
    is_start: pd.Timestamp,
    is_end: pd.Timestamp,
    sort_metric: str,
) -> tuple[dict[str, Any], PortfolioResult]:
    """Run every patch combo on the IS window; return (best_patch, is_pf).

    The best combo is the one maximizing ``sort_metric`` on the IS window. All
    combos share the window-sliced feed and pre-loaded benchmark candles via
    ``run_window`` — no per-combo data or benchmark reload.
    """
    best_patch: dict[str, Any] = merged_patches[0]
    best_pf: PortfolioResult | None = None
    best_val: float = float("-inf")

    for patch in merged_patches:
        conf = build_config(cfg, patch)
        pf = run_window(conf, strat_mod, data, bm_df, is_start, is_end)
        val = getattr(pf, sort_metric)
        if val > best_val:
            best_val = val
            best_patch = patch
            best_pf = pf

    assert best_pf is not None  # merged_patches is non-empty
    return best_patch, best_pf


def _is_metrics(pf: PortfolioResult) -> dict[str, float | int]:
    """Canonical per-run metric record for an IS window (see report_metrics)."""
    return metric_dict(pf)


def _optimize_fold_worker(fold: TestFold) -> OptimizeResult:
    """Run one fold's IS sweep + OOS validation inside a worker (or seq).

    The shared candle feed, benchmark feed, base config, merge and the IS
    sort metric come from the per-worker WORKER_STATE cache (pool initializer)
    so they pickle once per worker, not per fold.
    """
    from src.bt.parallel import WORKER_STATE
    from src.bt.strategies import init_strat
    from src.bt.window import run_window

    cfg = WORKER_STATE["cfg"]
    merge = WORKER_STATE["merge"]
    merged_patches = WORKER_STATE["merged_patches"]
    sort_metric = WORKER_STATE["sort_metric"]
    data = WORKER_STATE["data"]
    bm_df = WORKER_STATE.get("bm_df")

    strat_mod = init_strat(cfg.strategy_type)
    best_patch, is_pf = _best_combo_on_window(
        cfg,
        merged_patches,
        strat_mod,
        data,
        bm_df,
        fold.is_start,
        fold.is_end,
        sort_metric,
    )
    best_conf = build_config(cfg, best_patch)
    oos_pf = run_window(best_conf, strat_mod, data, bm_df, fold.oos_start, fold.oos_end)

    return OptimizeResult(
        fold=fold,
        best_params=_flat_overrides(merge, best_patch),
        is_metrics=_is_metrics(is_pf),
        oos=oos_pf,
    )


def run_optimize(
    cfg: StrategyConfig,
    folds: list[TestFold],
    merge: dict[str, Any],
    sort_metric: str = "sharpe_ratio",
    on_result: Callable[..., None] | None = None,
    workers: int = 1,
) -> tuple[list[OptimizeResult], dict[str, float]]:
    """Walk-forward optimize: tune params per fold's IS, validate on its OOS.

    Args:
        cfg: base strategy config.
        folds: IS/OOS fold windows (from split.walk_forward_folds/anchor_split).
        merge: partial config JSON; list-valued leaves are swept (cartesian).
        sort_metric: PortfolioResult metric maximized on each fold's IS.

    Returns:
        (per-fold OptimizeResult, aggregate summary dict).
        Candles load once over the full window; every IS sweep and OOS run
        reuses them. DSL strategies get fresh per-run ``ctx.shared`` state per
        window, so folds are independent and parallelizable.

    ``workers`` parallelizes folds over a process pool (default 1 = sequential).
    The shared candle + benchmark feeds pickle once per worker, not per fold.
    """
    if sort_metric not in _metric_names():
        raise ValueError(
            f"Unknown sort metric {sort_metric!r}; available: "
            f"{', '.join(sorted(_metric_names()))}"
        )

    from src.bt import warmup_load_start
    from src.bt.engine.backtest import Backtest
    from src.bt.data_feed import load_candles
    from src.bt.parallel import run_in_processes

    if not folds:
        raise ValueError("No folds to run — check the split windows")

    merged_patches = grid_combos(merge)

    # Load once over ``warmup + earliest fold`` so every fold — including the
    # first — has its own warmup span of history in front of its bars. Each
    # window re-derives its warmup from its own ``trading_start`` in
    # ``run_window``; there is no per-run reload.
    probe = Backtest(cfg)
    load_start = warmup_load_start(cfg, min(fold.is_start for fold in folds))
    data = load_candles(
        cfg.symbols,
        load_start,
        probe.window.test_end,
        cfg.bars[0],
    )

    # Benchmark candles are stateless — load once, slice per window. Even a
    # single IS sweep runs `combos` window backtests, so this avoids a DB
    # read per combo.
    bm_df: pd.DataFrame | None = None
    if cfg.benchmark_symbols:
        bm_df = load_candles(
            cfg.benchmark_symbols,
            load_start,
            probe.window.test_end,
            cfg.bars[0],
        )

    # Fail fast on windows that fall in a data gap instead of sweeping an
    # empty window (which would silently produce degenerate IS scores).
    for fold in folds:
        for label, start, end in (
            ("IS", fold.is_start, fold.is_end),
            ("OOS", fold.oos_start, fold.oos_end),
        ):
            if not window_has_data(data, start, end):
                raise ValueError(
                    f"Fold {fold.index + 1}: no candles in {label} [{start}→{end}] "
                    "— the split may fall in a data gap or past the loaded range."
                )

    def _stream(i: int, r: OptimizeResult) -> None:
        if on_result is not None:
            on_result(r.fold, r.best_params, r.is_metrics, r.oos)

    if workers <= 1:
        # Sequential path: reuse module-level names so monkeypatched
        # `run_window` / `init_strat` in tests keep working. `best_params` is
        # already flattened, matching the pooled `on_result` contract.
        from src.bt.strategies import init_strat

        strat_mod = init_strat(cfg.strategy_type)
        results: list[OptimizeResult] = []
        for fold in folds:
            best_patch, is_pf = _best_combo_on_window(
                cfg,
                merged_patches,
                strat_mod,
                data,
                bm_df,
                fold.is_start,
                fold.is_end,
                sort_metric,
            )
            best_conf = build_config(cfg, best_patch)
            oos_pf = run_window(
                best_conf, strat_mod, data, bm_df, fold.oos_start, fold.oos_end
            )
            best_is = _is_metrics(is_pf)
            results.append(
                OptimizeResult(
                    fold=fold,
                    best_params=_flat_overrides(merge, best_patch),
                    is_metrics=best_is,
                    oos=oos_pf,
                )
            )
            if on_result is not None:
                on_result(fold, _flat_overrides(merge, best_patch), best_is, oos_pf)
    else:
        results = run_in_processes(
            _optimize_fold_worker,
            list(folds),
            workers=workers,
            init_data={
                "cfg": cfg,
                "merge": merge,
                "merged_patches": merged_patches,
                "sort_metric": sort_metric,
                "data": data,
                "bm_df": bm_df,
            },
            on_complete=_stream,
        )

    agg = {
        "mean_oos_sharpe": (
            sum(float(r.oos.sharpe_ratio) for r in results) / len(results)
        ),
        "min_oos_sharpe": min(float(r.oos.sharpe_ratio) for r in results),
        "folds": len(results),
    }
    return results, agg


def _fold_row(r: OptimizeResult) -> tuple[str, ...]:
    """One fold's IS|OOS cells, interleaved from the canonical metric set.

    The IS side is read from the serialized ``is_metrics`` dict (only the dict
    survives a worker round-trip); the OOS side from the live result.
    """
    f = r.fold
    is_cells = metric_cells_from_dict(r.is_metrics)
    oos_cells = metric_cells(r.oos)
    interleaved = tuple(v for pair in zip(is_cells, oos_cells) for v in pair)
    return (
        str(f.index + 1),
        f"{f.is_start.date()}→{f.is_end.date()}",
        f"{f.oos_start.date()}→{f.oos_end.date()}",
        "\n".join(f"{k}={v}" for k, v in r.best_params.items()) or "—",
        *interleaved,
    )


def _optimize_columns() -> tuple[tuple[str, str], ...]:
    """Fold/window/params columns plus an IS|OOS pair per canonical metric."""
    cols: list[tuple[str, str]] = [
        ("Fold", "<"),
        ("IS window", "<"),
        ("OOS window", "<"),
        ("Chosen params", "<"),
    ]
    for label in metric_labels():
        cols.append((f"IS {label}", ">"))
        cols.append((f"OOS {label}", ">"))
    return tuple(cols)


def render_optimize_report(results: list[OptimizeResult], agg: dict[str, float]) -> str:
    """Render every fold's IS-tuned/OOS-validated metrics as ONE wide table.

    One row per fold; IS and OOS columns side by side so degradation is read
    horizontally. Kurtosis/skewness/win-rate/trade-count (and scaled-fill
    count) carry the tail risk and sample-size story a Sharpe-only view hides.
    """
    from src.bt.table import Col, Table, render

    table = Table(
        columns=tuple(Col(label, align) for label, align in _optimize_columns()),
        rows=tuple(_fold_row(r) for r in results),
    )
    lines = render(table)
    lines.append("")
    lines.append(
        f"AGGREGATE: mean OOS Sharpe {agg['mean_oos_sharpe']:.2f} · "
        f"min OOS Sharpe {agg['min_oos_sharpe']:.2f} · "
        f"{agg['folds']} fold(s)"
    )
    return "\n".join(lines)


def optimize_report_to_json(
    results: list[OptimizeResult], agg: dict[str, float]
) -> dict:
    """Serialize per-fold optimization results into a JSON-ready dict.

    Each fold's IS/OOS record is the canonical :func:`metric_dict` — the same
    field set as sweep/split JSON.
    """
    return {
        "folds": [
            {
                "index": r.fold.index,
                "is_window": f"{r.fold.is_start.date()}→{r.fold.is_end.date()}",
                "oos_window": (f"{r.fold.oos_start.date()}→{r.fold.oos_end.date()}"),
                "chosen_params": r.best_params,
                "is": dict(r.is_metrics),
                "oos": metric_dict(r.oos),
            }
            for r in results
        ],
        "agg": agg,
    }


__all__ = [
    "OptimizeResult",
    "run_optimize",
    "render_optimize_report",
    "optimize_report_to_json",
]
