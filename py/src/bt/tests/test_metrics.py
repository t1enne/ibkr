"""Tests for capital utilization helper in metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.bt.metrics import (
    capital_utilization,
    drawdown_periods,
    get_backtest_results_analysis,
    kurtosis,
    skewness,
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


def test_text_report_omits_trade_list_keeps_count() -> None:
    pf = _pf(
        (
            _trade(10.0, TradeStatus.closed),
            _trade(-5.0, TradeStatus.closed),
        )
    )
    lean = get_backtest_results_analysis(pf, include_trades=False)
    # header survives, body collapses to one count note, no per-trade rows
    assert "\nTrades" in lean
    assert "2 trades — list omitted (pass --trades to print)" in lean
    assert "Exit Reason" not in lean
    assert "Cost$" not in lean  # no trade table columns at all


# ---------------------------------------------------------------------------
# drawdown_periods
# ---------------------------------------------------------------------------


def _curve(values: list[float]) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


def test_drawdown_periods_empty_curve() -> None:
    assert drawdown_periods(pd.Series([], dtype=float)) == []


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


# ---------------------------------------------------------------------------
# skewness / kurtosis — cadence invariance
# ---------------------------------------------------------------------------


def _daily_equity_path(n_days: int = 150) -> pd.Series:
    """A deterministic dated equity path, one point per business day."""
    rng = np.random.default_rng(1234)
    rets = rng.normal(0.0004, 0.02, n_days)
    equity = 100_000.0 * np.cumprod(1.0 + rets)
    idx = pd.date_range("2023-01-02", periods=n_days, freq="B")
    return pd.Series(equity, index=idx, dtype=float)


def _intraday_sample(daily: pd.Series, bars_per_day: int = 7) -> pd.Series:
    """The same dated path sampled ``bars_per_day`` times a session.

    Within a session the mark walks from the prior session's close to this
    session's close, so the session's last intraday point equals the daily
    path's value for that date. Bars are stamped at session times, not
    midnight, so an intraday point shares its date with its daily point.
    """
    dates = pd.DatetimeIndex(daily.index)
    stamps = pd.DatetimeIndex(
        [
            date + pd.Timedelta(hours=13 + k)
            for date in dates
            for k in range(bars_per_day)
        ]
    )
    values: list[float] = []
    prev = float(daily.iloc[0])
    for close in daily.values:
        walk = np.linspace(prev, float(close), bars_per_day)
        values.extend(float(v) for v in walk)
        prev = float(close)
    return pd.Series(values, index=stamps, dtype=float)


def test_kurtosis_is_cadence_invariant() -> None:
    """Regression: the same dated equity path sampled intraday vs daily.

    Pre-fix, ``kurtosis`` took per-bar returns, so the intraday sample spread
    each session's move across its intraday bars with exact-zero returns
    between them and reported a mechanically inflated excess kurtosis. After
    the fix both cadences collapse to one return per session and agree.

    Verified to FAIL against the pre-fix implementation: the intraday sample
    reported a different (inflated) kurtosis than the daily sample.
    """
    daily = _daily_equity_path()
    intraday = _intraday_sample(daily)
    assert len(intraday) == len(daily) * 7
    assert kurtosis(intraday) == pytest.approx(kurtosis(daily), rel=1e-9)


def test_skewness_is_cadence_invariant() -> None:
    """Regression: skewness, same two-cadence construction as kurtosis.

    Verified to FAIL against the pre-fix implementation.
    """
    daily = _daily_equity_path()
    intraday = _intraday_sample(daily)
    assert skewness(intraday) == pytest.approx(skewness(daily), rel=1e-9)


def test_daily_base_curve_numbers_unchanged() -> None:
    """Characterisation: a one-point-per-date curve is untouched by the fix.

    The collapsed session returns of a daily curve are its own per-step
    returns, so a daily-base run's reported numbers cannot move.
    """
    daily = _daily_equity_path()
    per_step = daily.pct_change().dropna()
    assert len(per_step) == len(daily) - 1
    assert kurtosis(daily) == pytest.approx(float(per_step.kurt()))
    assert skewness(daily) == pytest.approx(float(per_step.skew()))


def test_moments_are_deterministic() -> None:
    """Characterisation: repeated calls on the same curve are identical."""
    intraday = _intraday_sample(_daily_equity_path())
    assert kurtosis(intraday) == kurtosis(intraday)
    assert skewness(intraday) == skewness(intraday)


def test_moments_edge_cases() -> None:
    """Characterisation: empty / tiny / flat / NaN curves stay well-defined."""
    empty = pd.Series([], dtype=float)
    assert kurtosis(empty) == 0.0
    assert skewness(empty) == 0.0

    single = pd.Series([100.0], index=pd.date_range("2024-01-01", periods=1))
    assert kurtosis(single) == 0.0
    assert skewness(single) == 0.0

    # all-flat: zero variance, one session per date
    flat = pd.Series(
        [100.0] * 10, index=pd.date_range("2024-01-01", periods=10, freq="B")
    )
    assert kurtosis(flat) == pytest.approx(0.0) or np.isnan(kurtosis(flat))
    assert np.isfinite(skewness(flat))

    # fewer than 4 session returns -> kurtosis undefined, returns 0.0
    three = pd.Series(
        [100.0, 101.0, 102.0], index=pd.date_range("2024-01-01", periods=3, freq="B")
    )
    assert kurtosis(three) == 0.0
    assert skewness(three) == 0.0

    # NaN in the curve: dropna in pct_change keeps the defined stretch
    nan_curve = pd.Series(
        [100.0, np.nan, 102.0, 101.0, 103.0, 105.0, 104.0],
        index=pd.date_range("2024-01-01", periods=7, freq="B"),
    )
    assert np.isfinite(kurtosis(nan_curve))
    assert np.isfinite(skewness(nan_curve))


def test_positional_index_uses_per_step_returns() -> None:
    """Characterisation: a non-datetime index has no calendar to collapse on."""
    curve = pd.Series([100.0, 101.0, 99.0, 102.0, 103.0, 98.0])
    per_step = curve.pct_change().dropna()
    assert kurtosis(curve) == pytest.approx(float(per_step.kurt()))
    assert skewness(curve) == pytest.approx(float(per_step.skew()))
