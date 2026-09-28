"""Tests for the walk-forward optimizer (IS-tune → OOS-validate)."""

from __future__ import annotations

import pandas as pd
from src.utils import parse_timestamp

from src.bt.optimize import (
    OptimizeResult,
    run_optimize,
    render_optimize_report,
    optimize_report_to_json,
)
import src.bt.optimize as _impl
from src.bt.split import TestFold
from src.bt.state import PortfolioResult
from src.bt.types import StrategyConfig


def _cfg() -> StrategyConfig:
    return StrategyConfig(
        name="opt",
        strategy_type="dummy",
        symbols=["A"],
        initial_capital=100000.0,
        commission=0.05,
        warmup="0d",
        trading_start="2020-01-02",
        trading_end="2024-01-01",
        bars=["1d"],
        strategy_params={"position_size": 0.95, "stop_loss": 0.5, "take_profit": 0.8},
        benchmark_symbols=[],
    )


def _fake_pf(sharpe: float, ann: float) -> PortfolioResult:
    return PortfolioResult(
        total_return=ann,
        sharpe_ratio=sharpe,
        trades=(),
        equity_curve=pd.Series([100000.0, 100000.0 * (1 + ann)]),
        annual_return=ann,
    )


def _folds() -> list[TestFold]:
    return [
        TestFold(
            index=0,
            is_start=parse_timestamp("2020-01-02"),
            is_end=parse_timestamp("2021-01-01"),
            oos_start=parse_timestamp("2021-01-04"),
            oos_end=parse_timestamp("2022-01-01"),
        ),
        TestFold(
            index=1,
            is_start=parse_timestamp("2020-01-02"),
            is_end=parse_timestamp("2022-01-01"),
            oos_start=parse_timestamp("2022-01-03"),
            oos_end=parse_timestamp("2023-01-01"),
        ),
    ]


def test_flat_overrides_flattens_swept_leaves():
    merge = {"strategy_params": {"ma_slow": [50], "nested": {"x": [1.5]}}}
    patch = {"strategy_params": {"ma_slow": 200, "nested": {"x": 2.0}}}
    out = _impl._flat_overrides(merge, patch)
    assert out == {"strategy_params.ma_slow": 200, "strategy_params.nested.x": 2.0}


def test_flat_overrides_empty_merge():
    assert _impl._flat_overrides({}, {}) == {}


def test_run_optimize_tunes_on_oos_bounded_by_scoped_windows(monkeypatch):
    """IS tuning picks the max-sharpe combo per fold; OOS validates the winner.

    `run_window` is stubbed so a combo's sharpe equals its ``x`` param on the
    IS window, and the OOS window returns the ``x`` of whatever config it is
    handed (the IS-best combo). The runner must therefore: run every combo per
    fold IS, pick the largest x, and pass that exact config to the OOS run.
    """
    opt = _impl

    cfg = _cfg()
    # Grid: x in [1, 2, 3] → combos x=1,x=2,x=3. IS-best per fold = x=3.
    merge = {"strategy_params": {"x": [1, 2, 3]}}

    seen: dict[str, list] = {"oos_x": []}

    def fake_run_window(cfg, strat_mod, data, bm_df, t_start, t_end):
        x = cfg.strategy_params["x"]
        # A tuning window is one whose END equals a fold's IS end. We detect
        # it by comparing to the fold set — but simpler: OOS runs pass a
        # config whose trading_start == oos_start of a known fold.
        if t_start >= pd.Timestamp("2021-01-04"):
            seen["oos_x"].append(x)
            return _fake_pf(float(x), float(x) / 10.0)
        return _fake_pf(float(x), float(x) / 100.0)

    monkeypatch.setattr(opt, "run_window", fake_run_window)
    monkeypatch.setattr(
        "src.bt.data_feed.load_candles",
        lambda *a, **k: pd.DataFrame(
            index=pd.date_range("2020-01-02", "2023-01-01", freq="D")
        ),
    )
    monkeypatch.setattr(
        "src.bt.strategies.init_strat",
        lambda name: type(
            "Mod", (), {"reset_global": lambda: None, "STRATEGY_TYPE": "dummy"}
        )(),
    )

    results, agg = run_optimize(cfg, _folds(), merge, sort_metric="sharpe_ratio")

    assert len(results) == 2
    # Best IS sharpe per fold is x=3 (largest sharpe by construction).
    assert all(r.best_params == {"strategy_params.x": 3} for r in results)
    # OOS ran with the IS-best config (x=3) for each fold.
    assert seen["oos_x"] == [3, 3]
    # OOS sharpe of the best combo = 3.0 for both.
    assert all(abs(r.oos.sharpe_ratio - 3.0) < 1e-9 for r in results)
    assert agg["folds"] == 2
    assert abs(agg["mean_oos_sharpe"] - 3.0) < 1e-9


def test_run_optimize_rejects_unknown_sort_metric():
    import pytest

    opt = _impl

    cfg = _cfg()
    with pytest.raises(ValueError):
        opt.run_optimize(cfg, _folds(), {}, sort_metric="not_a_metric")


def test_optimize_agg_and_serialization():
    """Aggregate + JSON round-trip shape."""
    r = OptimizeResult(
        fold=_folds()[0],
        best_params={"strategy_params.x": 3},
        is_metrics={
            "annual_return": 0.3,
            "sharpe_ratio": 1.9,
            "max_drawdown": -0.1,
            "calmar_ratio": 3.0,
        },
        oos=_fake_pf(1.4, 0.12),
    )
    agg = {"mean_oos_sharpe": 1.4, "min_oos_sharpe": 1.4, "folds": 1}
    text = render_optimize_report([r], agg)
    assert "strategy_params.x=3" in text
    assert "OOS Kurt" in text
    assert "OOS Win" in text
    assert "OOS Trd" in text
    assert "mean OOS Sharpe 1.40" in text

    js = optimize_report_to_json([r], agg)
    assert js["folds"][0]["oos"]["sharpe_ratio"] == 1.4
    assert "kurtosis" in js["folds"][0]["oos"]
    assert "win_rate" in js["folds"][0]["oos"]
    assert "trade_count" in js["folds"][0]["oos"]
    assert js["folds"][0]["is"]["sharpe_ratio"] == 1.9
    assert js["folds"][0]["chosen_params"] == {"strategy_params.x": 3}
    assert js["agg"]["mean_oos_sharpe"] == 1.4


def test_is_metrics_includes_trade_derived_stats():
    """``_is_metrics`` folds trade-derived win rate / count into the dict."""
    from src.bt.metrics import trade_count, win_rate

    pf = _fake_pf(1.2, 0.2)
    m = _impl._is_metrics(pf)
    assert m["win_rate"] == win_rate(pf) == 0.0
    assert m["trade_count"] == float(trade_count(pf)) == 0.0
    assert "kurtosis" in m
