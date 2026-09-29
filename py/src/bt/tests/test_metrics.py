"""Tests for capital utilization helper in metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from src.bt.metrics import (
    capital_utilization,
    drawdown_periods,
    trade_count,
    win_rate,
)
from src.bt.state import Trade, TradeStatus
from src.bt.state import PortfolioResult
from src.bt.types import ActionType
from src.utils import parse_timestamp


class _Pt:
    """Minimal EquityPoint stand-in (equity, positions_value)."""

    def __init__(self, equity: float, positions_value: float) -> None:
        self.equity = equity
        self.positions_value = positions_value


def test_capital_utilization_empty() -> None:
    assert capital_utilization(None) == 0.0
    assert capital_utilization(()) == 0.0


def test_capital_utilization_deployed() -> None:
    pts = (
        _Pt(equity=100_000, positions_value=50_000),  # 50%
        _Pt(equity=110_000, positions_value=110_000),  # 100%
        _Pt(equity=90_000, positions_value=0),  # 0%
    )
    result = capital_utilization(pts)
    assert result == pytest.approx((0.5 + 1.0 + 0.0) / 3)


def test_capital_utilization_skips_nonpositive_equity() -> None:
    pts = (
        _Pt(equity=100_000, positions_value=25_000),  # 25%
        _Pt(equity=0, positions_value=0),  # skipped
        _Pt(equity=-5, positions_value=1),  # skipped
    )
    result = capital_utilization(pts)
    assert result == pytest.approx(0.25)


def test_capital_utilization_all_invalid() -> None:
    pts = (
        _Pt(equity=0, positions_value=0),
        _Pt(equity=-1, positions_value=0),
    )
    assert capital_utilization(pts) == 0.0


def test_capital_utilization_full_investment() -> None:
    pts = tuple(_Pt(equity=50_000, positions_value=50_000) for _ in range(10))
    assert capital_utilization(pts) == pytest.approx(1.0)


def _trade(pnl: float, status: TradeStatus) -> Trade:
    """Minimal Trade: only ``pnl`` + ``status`` are read by the helpers."""
    ts = parse_timestamp("2020-01-02")
    return Trade(
        entry_time=ts,
        entry_price=100.0,
        exit_time=ts,
        exit_price=100.0,
        last_price=100.0,
        symbol="A",
        position=ActionType.long,
        qty=1.0,
        stop_loss=0.0,
        take_profit=0.0,
        pnl=pnl,
        status=status,
    )


def _pf(trades: tuple[Trade, ...]) -> PortfolioResult:
    return PortfolioResult(
        total_return=0.0,
        sharpe_ratio=0.0,
        trades=trades,
        equity_curve=pd.Series([100_000.0, 100_000.0]),
    )


def test_win_rate_and_trade_count_ignore_open_trades() -> None:
    pf = _pf(
        (
            _trade(10.0, TradeStatus.closed),
            _trade(-5.0, TradeStatus.closed),
            _trade(99.0, TradeStatus.open),
        )
    )
    assert trade_count(pf) == 2
    assert win_rate(pf) == pytest.approx(0.5)


def test_win_rate_zero_without_closed_trades() -> None:
    pf = _pf((_trade(1.0, TradeStatus.open),))
    assert trade_count(pf) == 0
    assert win_rate(pf) == 0.0


# ---------------------------------------------------------------------------
# drawdown_periods
# ---------------------------------------------------------------------------


def _curve(values: list[float]) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


def test_drawdown_periods_empty_curve() -> None:
    assert drawdown_periods(pd.Series([], dtype=float)) == []


def test_drawdown_periods_monotone_rise_has_none() -> None:
    assert drawdown_periods(_curve([100.0, 101.0, 102.0, 103.0])) == []


def test_drawdown_periods_peak_precedes_valley() -> None:
    # Peak at index 1, valley at index 2 (-50%), recovered at index 4.
    periods = drawdown_periods(_curve([100.0, 100.0, 50.0, 60.0, 100.0]))
    assert len(periods) == 1
    p = periods[0]
    assert p.net_drawdown_pct == pytest.approx(50.0)
    assert p.peak_date == pd.Timestamp("2020-01-02")
    assert p.valley_date == pd.Timestamp("2020-01-03")
    assert p.recovery_date == pd.Timestamp("2020-01-05")
    assert p.duration == 3


def test_drawdown_periods_open_episode_uses_local_valley() -> None:
    """Regression: the still-open episode must NOT use a global argmin.

    The deep valley (-60%) belongs to the FIRST, recovered episode; the open
    episode's own valley is only -30%. The old implementation applied
    ``np.argmin`` over the whole series, printing a valley *before* the peak.
    """
    periods = drawdown_periods(_curve([100.0, 40.0, 100.0, 100.0, 70.0, 60.0]))
    assert len(periods) == 2
    closed, open_ep = periods[0], periods[1]
    assert closed.net_drawdown_pct == pytest.approx(60.0)
    assert closed.recovery_date is not None
    assert open_ep.net_drawdown_pct == pytest.approx(40.0)
    assert open_ep.recovery_date is None
    # The open episode's valley is inside the episode, after its peak.
    assert open_ep.peak_date == pd.Timestamp("2020-01-04")
    assert open_ep.valley_date == pd.Timestamp("2020-01-06")
    assert open_ep.peak_date is not None and open_ep.valley_date is not None
    assert open_ep.peak_date <= open_ep.valley_date


def test_drawdown_periods_never_reports_valley_before_peak() -> None:
    """Invariant: peak <= valley <= recovery, for every emitted episode."""
    curve = _curve([100.0, 90.0, 85.0, 100.0, 120.0, 60.0, 70.0, 119.0, 100.0, 80.0])
    periods = drawdown_periods(curve, min_drawdown=0.05)
    assert periods, "expected at least one episode"
    for p in periods:
        assert p.peak_date is not None and p.valley_date is not None
        assert p.peak_date <= p.valley_date
        if p.recovery_date is not None:
            assert p.valley_date <= p.recovery_date


def test_drawdown_periods_sorted_deepest_first() -> None:
    periods = drawdown_periods(_curve([100.0, 70.0, 100.0, 100.0, 60.0, 100.0]))
    dd_pcts = [p.net_drawdown_pct for p in periods]
    assert dd_pcts == sorted(dd_pcts, reverse=True)


def test_drawdown_periods_filters_below_threshold() -> None:
    # -3% dip must not be emitted at the default 5% threshold.
    assert drawdown_periods(_curve([100.0, 97.0, 100.0])) == []
    shallow = drawdown_periods(_curve([100.0, 97.0, 100.0]), min_drawdown=0.01)
    assert len(shallow) == 1
    assert shallow[0].net_drawdown_pct == pytest.approx(3.0)
