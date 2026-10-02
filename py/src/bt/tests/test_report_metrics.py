"""Tests for the shared canonical metric formatter (text/JSON parity)."""

import pandas as pd

from src.bt.metrics import calculate_portfolio_result
from src.bt.output import render_result_json, _metric_pairs
from src.bt.report_metrics import (
    CANONICAL_METRICS,
    metric_cells,
    metric_dict,
    metric_labels,
)
from src.bt.state import PortfolioResult


def _pf(scaled_trades: int = 0) -> PortfolioResult:
    equity = pd.Series(
        [100000.0, 101000.0, 100500.0, 102000.0],
        index=pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"]),
    )
    return PortfolioResult(
        total_return=0.02,
        sharpe_ratio=1.5,
        trades=(),
        equity_curve=equity,
        annual_return=0.2,
        max_drawdown=-0.1,
        kurtosis=3.2,
        skewness=-0.4,
        scaled_trades=scaled_trades,
    )


def test_text_cells_and_json_dict_share_one_canonical_set() -> None:
    """Labels, cells and dict keys are driven by the same spec table."""
    pf = _pf()
    assert len(metric_cells(pf)) == len(CANONICAL_METRICS)
    assert metric_labels() == (
        "Sharpe",
        "Ann",
        "MaxDD",
        "Kurt",
        "Skew",
        "Win",
        "Trades",
        "Scaled",
    )
    assert tuple(metric_dict(pf)) == tuple(s.key for s in CANONICAL_METRICS)
    # cells are the formatted form of the dict values
    d = metric_dict(pf)
    assert metric_cells(pf) == tuple(s.fmt(float(d[s.key])) for s in CANONICAL_METRICS)


def test_canonical_set_includes_tail_and_scaled_metrics() -> None:
    d = metric_dict(_pf(scaled_trades=4))
    assert d["kurtosis"] == 3.2
    assert d["skewness"] == -0.4
    assert d["scaled_trades"] == 4
    assert d["trade_count"] == 0
    assert d["win_rate"] == 0.0


def test_scaled_trades_flows_into_portfolio_result_and_json() -> None:
    """Engine passes rejection count -> PortfolioResult -> run JSON output."""
    equity = pd.Series([1000.0, 1010.0], index=[0, 1])
    pf = calculate_portfolio_result(equity, (), 1000.0, scaled_trades=3)
    assert pf.scaled_trades == 3
    assert dict(_metric_pairs(pf))["scaled_trades"] == 3.0
    r = render_result_json(type("R", (), {"pf": pf, "benchmark_curves": {}})())
    assert r["metrics"]["scaled_trades"] == 3.0


def test_scaled_trades_defaults_to_zero() -> None:
    assert _pf().scaled_trades == 0
