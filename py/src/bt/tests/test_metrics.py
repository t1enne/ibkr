"""Tests for capital utilization helper in metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from src.bt.metrics import capital_utilization, trade_count, win_rate
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
