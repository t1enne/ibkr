"""Walk-forward / single-anchor IS-OOS split validation.

Evaluates a strategy's FIXED parameter set across in-sample (IS) and
out-of-sample (OOS) windows. It does NOT re-tune params per fold — the
engine has no optimizer. Answers the honest-validation question:
"given these locked params, how does performance hold up out-of-sample?"

Pure fold math lives here (test-friendly); engine wiring is `run_split`.
Mirrors the repo's `pure.py` convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import pandas as pd

from src.bt.types import StrategyConfig, PortfolioResult
from src.bt.window import run_window, window_has_data
from src.utils import parse_timestamp

DAY = pd.offsets.BDay(1)


@dataclass(frozen=True, slots=True)
class TestFold:
    """One IS/OOS evaluation window pair.

    IS = [is_start, is_end]; OOS = [oos_start, oos_end]. OOS begins on the
    next trading day after IS ends, so the windows are disjoint and adjacent.
    """

    # Not a pytest test collection target — pure domain type. The ``Test``
    # prefix makes pytest otherwise try to collect it and warn.
    __test__ = False

    index: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp

    def is_trading_window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return (self.is_start, self.is_end)

    def oos_trading_window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return (self.oos_start, self.oos_end)


@dataclass(frozen=True)
class FoldMetrics:
    fold: TestFold
    in_sample: PortfolioResult
    out_of_sample: PortfolioResult


@dataclass(frozen=True)
class SplitReport:
    config_name: str
    params: Mapping[str, object]
    folds: tuple[FoldMetrics, ...]

    def oos_sharpe_series(self) -> pd.Series:
        return pd.Series(
            [f.out_of_sample.sharpe_ratio for f in self.folds],
            index=[f.fold.index for f in self.folds],
            name="oos_sharpe",
        )

    def mean_oos_sharpe(self) -> float:
        if not self.folds:
            return 0.0
        return float(self.oos_sharpe_series().mean())

    def min_oos_sharpe(self) -> float:
        if not self.folds:
            return 0.0
        return float(self.oos_sharpe_series().min())

    def oos_vs_is_degradation(self) -> float:
        """mean(IS Sharpe) − mean(OOS Sharpe) — IS→OOS Sharpe decay.

        Positive = performance degraded out-of-sample; negative = OOS beat IS.
        Reported as a delta (not a ratio) because a ratio of two signed Sharpes
        is meaningless when IS is negative or near zero.
        """
        if not self.folds:
            return 0.0
        is_avg = sum(f.in_sample.sharpe_ratio for f in self.folds) / len(self.folds)
        return is_avg - self.mean_oos_sharpe()


def anchor_split(
    cfg: StrategyConfig,
    is_end: pd.Timestamp,
) -> list[TestFold]:
    """Single split: IS=[trading_start, is_end], OOS=[is_end+1d, trading_end].

    Raises ValueError when is_end is not strictly before trading_end (the OOS
    window would be empty), or when is_end precedes trading_start.
    """
    start = parse_timestamp(cfg.trading_start)
    end = parse_timestamp(cfg.trading_end)
    is_end = parse_timestamp(is_end)

    if is_end <= start:
        raise ValueError(
            f"--is-end {is_end.date()} must be after trading_start {start.date()}"
        )
    if is_end >= end:
        raise ValueError(
            f"--is-end {is_end.date()} must be before trading_end {end.date()} "
            "(OOS window would be empty)"
        )

    return [
        TestFold(
            index=0,
            is_start=start,
            is_end=is_end,
            oos_start=is_end + DAY,
            oos_end=end,
        )
    ]


def walk_forward_folds(
    cfg: StrategyConfig,
    n_folds: int,
    *,
    min_is_years: float = 5.0,
    oos_length: str | pd.DateOffset = "auto",
) -> list[TestFold]:
    """Expansion-window walk-forward: IS always starts at trading_start, grows.

    Produces exactly `n_folds` non-empty folds. With `oos_length="auto"` the
    OOS chunk is `span/(n_folds+1)`, so the leading IS plus `n_folds` OOS
    chunks tile `[trading_start, trading_end]` (a leading IS anchor of one
    chunk + n OOS chunks = n+1 equal slices). Fold i's IS = [is_start, is_end_i]
    (is_end_i = the boundary before OOS chunk i+1, so IS grows monotonically)
    and its OOS is the next chunk reaching the following boundary (the last
    one reaches trading_end).

    With an explicit `oos_length` DateOffset the boundaries step by that
    offset, clamped to `trading_end`; a trailing chunk too short to be
    non-empty is skipped, and fewer-than-requested folds warn (plan edge
    case 2).

    `is_start` is ``trading_start``; the warmup span in front of it is walked
    by the engine (see ``EngineWindow.warmup_bars``).

    Warns once when the first fold's IS is shorter than min_is_years.
    """
    start = parse_timestamp(cfg.trading_start)
    end = parse_timestamp(cfg.trading_end)
    span = end - start

    if n_folds < 1:
        raise ValueError("--folds must be >= 1")

    if oos_length == "auto":
        oos = span / (n_folds + 1)
        boundaries: list[pd.Timestamp] = [
            (start + oos * (i + 1)).normalize() for i in range(n_folds)
        ]
    else:
        assert isinstance(oos_length, pd.DateOffset), (
            "oos_length must be 'auto' or a pd.DateOffset"
        )
        cursor = start
        boundaries: list[pd.Timestamp] = []
        for _ in range(n_folds):
            cursor = (cursor + oos_length).normalize()
            if cursor >= end:
                cursor = end  # clamp — never walk past trading_end
            boundaries.append(cursor)
            if cursor == end:
                break

    is_start = start

    first_is_len = boundaries[0] - is_start
    if first_is_len.days / 365.25 < min_is_years:
        import warnings

        warnings.warn(
            f"First fold IS is only {first_is_len.days / 365.25:.1f}y "
            f"(< min_is_years {min_is_years:g}); later folds have more history. "
            "Runs may be thin early on.",
            stacklevel=2,
        )

    folds: list[TestFold] = []
    for i, is_end in enumerate(boundaries):
        oos_start_val = is_end + DAY
        oos_end_val = boundaries[i + 1] if i + 1 < len(boundaries) else end
        if oos_end_val <= oos_start_val:
            continue  # empty OOS chunk — skip (plan edge case 2)
        folds.append(
            TestFold(
                index=len(folds),  # contiguous, even after empty-chunk skips
                is_start=is_start,
                is_end=is_end,
                oos_start=oos_start_val,
                oos_end=oos_end_val,
            )
        )
    if len(folds) < n_folds:
        import warnings

        warnings.warn(
            f"Only {len(folds)} non-empty fold(s) possible within "
            f"[{start.date()}, {end.date()}] — requested {n_folds}. "
            "Reduce --folds or shrink the OOS offset.",
            stacklevel=2,
        )
    return folds


# ---------------------------------------------------------------------------
# engine wiring
# ---------------------------------------------------------------------------


def _split_fold_worker(fold: TestFold) -> FoldMetrics:
    """Run one fold's IS+OOS windows inside a worker (or sequentially).

    The shared candle feed, benchmark feed and strategy config come from the
    per-worker WORKER_STATE cache (pool initializer) so they pickle once per
    worker, not per fold.
    """
    from src.bt.parallel import WORKER_STATE
    from src.bt.strategies import init_strat

    cfg = WORKER_STATE["cfg"]
    data = WORKER_STATE["data"]
    bm_df = WORKER_STATE.get("bm_df")

    strat_mod = init_strat(cfg.strategy_type)
    is_result = run_window(cfg, strat_mod, data, bm_df, fold.is_start, fold.is_end)
    oos_result = run_window(cfg, strat_mod, data, bm_df, fold.oos_start, fold.oos_end)
    return FoldMetrics(fold=fold, in_sample=is_result, out_of_sample=oos_result)


def run_split(
    cfg: StrategyConfig,
    folds: list[TestFold],
    on_result: Callable[[TestFold, PortfolioResult, PortfolioResult], None]
    | None = None,
    workers: int = 1,
) -> SplitReport:
    """Run one backtest per IS and OOS window of every fold.

    - strategy_params are NEVER mutated across folds (locked params).
    - Loads candles once over [warmup_start, trading_end], window-sliced per
      fold via trading-window overrides (no per-fold data reload); each fold
      gets its own warmup span in front of its bars. DSL
      strategies get fresh per-run ``ctx.shared`` state minted by the engine
      each window, so folds are independent and parallelizable.

    ``on_result`` (optional) pulls (fold, is_result, oos_result) as each fold
    completes, letting callers stream results live.

    ``workers`` parallelizes folds over a process pool (default 1 = sequential).
    The shared candle + benchmark feeds pickle once per worker, not per fold.
    """
    if not folds:
        raise ValueError("No folds to run — check the split windows")

    from src.bt import warmup_load_start
    from src.bt.data_feed import load_candles
    from src.bt.parallel import run_in_processes

    # Every fold is fed ``warmup + fold bars``: the load reaches back far enough
    # that the EARLIEST fold's ``is_start`` has its own warmup span of history
    # in front of it. Each window then re-derives its warmup from its own
    # ``trading_start`` inside ``run_window``.
    load_start = warmup_load_start(cfg, min(f.is_start for f in folds))
    data = load_candles(
        cfg.symbols,
        load_start,
        parse_timestamp(cfg.trading_end),
        cfg.bars[0],
    )

    # Benchmark candles are stateless — load once, slice per window.
    bm_df: pd.DataFrame | None = None
    if cfg.benchmark_symbols:
        bm_df = load_candles(
            cfg.benchmark_symbols,
            load_start,
            parse_timestamp(cfg.trading_end),
            cfg.bars[0],
        )

    # Fail fast on windows that fall in a data gap before dispatching workers.
    for fold in folds:
        windows = [
            ("IS", fold.is_start, fold.is_end),
            ("OOS", fold.oos_start, fold.oos_end),
        ]
        missing = [
            f"{label} [{start}→{end}]"
            for label, start, end in windows
            if not window_has_data(data, start, end)
        ]
        if missing:
            raise ValueError(
                f"Fold {fold.index + 1}: no candles in "
                + ", ".join(missing)
                + " — the split may fall in a data gap or past the loaded range."
            )

    def _stream(i: int, fm: FoldMetrics) -> None:
        if on_result is not None:
            on_result(fm.fold, fm.in_sample, fm.out_of_sample)

    fold_metrics = run_in_processes(
        _split_fold_worker,
        list(folds),
        workers=workers,
        init_data={"cfg": cfg, "data": data, "bm_df": bm_df},
        on_complete=_stream,
    )

    return SplitReport(
        config_name=cfg.name,
        params=dict(cfg.strategy_params),
        folds=tuple(fold_metrics),
    )


# ---------------------------------------------------------------------------
# rendering / serialization
# ---------------------------------------------------------------------------


def _fold_rows(fm: FoldMetrics) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """One fold's IS row and OOS row — never the same row.

    Side-by-side IS|OOS columns overflowed (one row per fold wrapped off the
    terminal); stacking the pair keeps the table readable, matching
    ``bt optimize``. Both rows use the same canonical metric set, so the
    columns align and degradation is read vertically.
    """
    from src.bt.report_metrics import metric_cells

    f = fm.fold
    fold_col = str(f.index + 1)
    is_row = (
        fold_col,
        "IS",
        f"{f.is_start.date()}→{f.is_end.date()}",
        *metric_cells(fm.in_sample),
    )
    oos_row = (
        "",
        "OOS",
        f"{f.oos_start.date()}→{f.oos_end.date()}",
        *metric_cells(fm.out_of_sample),
    )
    return is_row, oos_row


def _split_columns() -> tuple[tuple[str, str], ...]:
    """Fold/phase/window columns plus the canonical metric columns."""
    from src.bt.report_metrics import metric_labels

    cols: list[tuple[str, str]] = [
        ("Fold", "<"),
        ("Phase", "<"),
        ("Window", "<"),
    ]
    cols.extend((label, ">") for label in metric_labels())
    return tuple(cols)


def render_split_report(report: SplitReport) -> str:
    """Render every fold's IS vs OOS metrics as ONE table.

    Two rows per fold — the in-sample row then its out-of-sample row — so
    degradation is read vertically without the column overflow of an IS|OOS
    pair on a single row. Kurtosis/skewness/win-rate/trade-count (and
    scaled-fill count) carry the tail-risk and sample-size story a Sharpe-only
    view hides.
    """
    from src.bt.table import Col, Table, render

    if not report.folds:
        return f"\nSplit: {report.config_name} (no folds)"

    table = Table(
        columns=tuple(Col(label, align) for label, align in _split_columns()),
        rows=tuple(row for fm in report.folds for row in _fold_rows(fm)),
    )
    lines = [f"\nSplit: {report.config_name}"]
    lines.extend(render(table))
    lines.append("")
    lines.append(
        f"AGGREGATE: mean OOS Sharpe {report.mean_oos_sharpe():.2f} · "
        f"min OOS Sharpe {report.min_oos_sharpe():.2f} · "
        f"IS→OOS decay {report.oos_vs_is_degradation():+.2f} · "
        f"{len(report.folds)} fold(s)"
    )
    return "\n".join(lines).rstrip()


def split_report_to_dict(report: SplitReport) -> dict:
    """Serialize a SplitReport into a plain JSON-ready dict.

    Each fold's IS/OOS records are the canonical :func:`metric_dict` — the
    same field set as sweep/optimize JSON.
    """
    from src.bt.report_metrics import metric_dict

    return {
        "config": report.config_name,
        "params": {str(k): v for k, v in report.params.items()},
        "folds": [
            {
                "index": fm.fold.index,
                "is_window": fm.fold.is_trading_window()[0].date().isoformat(),
                "is_end": fm.fold.is_end.date().isoformat(),
                "oos_window": fm.fold.oos_trading_window()[0].date().isoformat(),
                "oos_end": fm.fold.oos_end.date().isoformat(),
                "is": metric_dict(fm.in_sample),
                "oos": metric_dict(fm.out_of_sample),
            }
            for fm in report.folds
        ],
        "agg": {
            "mean_oos_sharpe": report.mean_oos_sharpe(),
            "min_oos_sharpe": report.min_oos_sharpe(),
            "oos_vs_is_degradation": report.oos_vs_is_degradation(),
        },
    }


__all__ = [
    "TestFold",
    "FoldMetrics",
    "SplitReport",
    "anchor_split",
    "walk_forward_folds",
    "run_split",
    "render_split_report",
    "split_report_to_dict",
]
