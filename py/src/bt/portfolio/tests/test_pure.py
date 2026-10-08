"""Tests for pure portfolio functions — critical paths only."""

from typing import cast

import pandas as pd
import pytest

from src.bt.portfolio.pure import (
    apply_fill,
)
from src.bt.state import (
    ActionType,
    EquityPoint,
    FillEvent,
    PortfolioState,
    Position,
    Trade,
    TradeSignal,
    TradeStatus,
    create_initial_portfolio,
)


def _ts(val: str) -> pd.Timestamp:
    result = cast(pd.Timestamp, pd.Timestamp(val))
    assert not pd.isna(result)
    return result


def test_open_long_position():
    portfolio = create_initial_portfolio(
        initial_capital=10000, start_timestamp=_ts("2024-01-01")
    )
    fill = FillEvent(
        signal=TradeSignal(
            action=ActionType.long,
            symbol="AAPL",
            timestamp=_ts("2024-01-01"),
            price=100.0,
            qty=10.0,
            stop_loss=95.0,
            take_profit=110.0,
        ),
        filled_qty=10.0,
        executed_price=100.0,
        commission=1.0,
        slippage=0.0,
        timestamp=_ts("2024-01-01"),
    )
    new = apply_fill(portfolio, fill)
    assert "AAPL" in new.positions
    assert len(new.trades) == 1
    assert portfolio.cash == 10000  # immutable


def test_close_long_position():
    pid = "AAPL_close"
    position = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=_ts("2024-01-01"),
        stop_loss=95.0,
        take_profit=110.0,
        last_price=100.0,
        type=ActionType.long,
        position_id=pid,
    )
    portfolio = PortfolioState(
        cash=5000,
        positions={"AAPL": (position,)},
        trades=(
            Trade(
                entry_time=_ts("2024-01-01"),
                entry_price=100.0,
                exit_time=None,
                exit_price=None,
                last_price=100.0,
                reason="",
                symbol="AAPL",
                position=ActionType.long,
                qty=10.0,
                stop_loss=95.0,
                take_profit=110.0,
                pnl=0.0,
                status=TradeStatus.open,
                position_id=pid,
            ),
        ),
        equity_curve=(
            EquityPoint(
                timestamp=_ts("2024-01-01"),
                equity=6000,
                cash=5000,
                positions_value=1000,
            ),
        ),
        initial_capital=10000,
    )
    fill = FillEvent(
        signal=TradeSignal(
            action=ActionType.close,
            symbol="AAPL",
            timestamp=_ts("2024-01-02"),
            price=110.0,
            position_id=pid,
        ),
        filled_qty=10.0,
        executed_price=110.0,
        commission=1.0,
        slippage=0.0,
        timestamp=_ts("2024-01-02"),
    )
    new = apply_fill(portfolio, fill)
    assert "AAPL" not in new.positions
    assert new.trades[0].status == TradeStatus.closed
    assert new.trades[0].pnl == 99.0  # (110-100)*10 - 1 close commission
    assert "AAPL" in portfolio.positions  # immutable


def test_close_requires_position_id():
    portfolio = create_initial_portfolio(
        initial_capital=10000, start_timestamp=_ts("2024-01-01")
    )
    fill_open = FillEvent(
        signal=TradeSignal(
            action=ActionType.long,
            symbol="AAPL",
            timestamp=_ts("2024-01-01"),
            price=100.0,
            qty=10.0,
            position_id="AAPL_open",
        ),
        filled_qty=10.0,
        executed_price=100.0,
        commission=1.0,
        slippage=0.0,
        timestamp=_ts("2024-01-01"),
    )
    portfolio = apply_fill(portfolio, fill_open)
    fill_close = FillEvent(
        signal=TradeSignal(
            action=ActionType.close,
            symbol="AAPL",
            timestamp=_ts("2024-01-02"),
            price=110.0,
        ),
        filled_qty=10.0,
        executed_price=110.0,
        commission=1.0,
        slippage=0.0,
        timestamp=_ts("2024-01-02"),
    )
    with pytest.raises(ValueError, match="requires position_id"):
        apply_fill(portfolio, fill_close)


# ---------------------------------------------------------------------------
# multi-position: tag propagation, lot resolution, net reads
# ---------------------------------------------------------------------------


def _long_position(pid: str, qty: float, price: float, tag: str = "") -> Position:
    return Position(
        symbol="AAPL",
        qty=qty,
        entry_price=price,
        entry_time=_ts("2024-01-01"),
        stop_loss=None,
        take_profit=None,
        last_price=price,
        type=ActionType.long,
        position_id=pid,
        tag=tag,
    )


def _fill_long(qty: float, price: float, position_id: str, tag: str = "") -> FillEvent:
    return FillEvent(
        signal=TradeSignal(
            action=ActionType.long,
            symbol="AAPL",
            timestamp=_ts("2024-01-01"),
            price=price,
            qty=qty,
            position_id=position_id,
            tag=tag,
        ),
        filled_qty=qty,
        executed_price=price,
        commission=1.0,
        slippage=0.0,
        timestamp=_ts("2024-01-01"),
    )


def _fill_short(qty: float, price: float, position_id: str, tag: str = "") -> FillEvent:
    return FillEvent(
        signal=TradeSignal(
            action=ActionType.short,
            symbol="AAPL",
            timestamp=_ts("2024-01-01"),
            price=price,
            qty=qty,
            position_id=position_id,
            tag=tag,
        ),
        filled_qty=qty,
        executed_price=price,
        commission=1.0,
        slippage=0.0,
        timestamp=_ts("2024-01-01"),
    )


def test_open_stores_tag_on_position():
    from src.bt.portfolio.pure import get_symbol_positions, resolve_lot

    portfolio = create_initial_portfolio(
        initial_capital=10000, start_timestamp=_ts("2024-01-01")
    )
    portfolio = apply_fill(portfolio, _fill_long(10.0, 100.0, "AAPL_1", tag="spy-r1"))
    pos = get_symbol_positions(portfolio, "AAPL")[0]
    assert pos.tag == "spy-r1"
    assert resolve_lot((pos,), tag="spy-r1") is pos
    assert resolve_lot((pos,), lot="AAPL_1") is pos


def test_short_partial_then_full_cover_mirror_equity():
    """Full short cover after an earlier partial: equity closes the round trip.

    Pins the numeric target the mirror basis implies: after the partial cover
    equity is 100098 (realized 40 + unrealized 60 - 2 comm); the final cover of
    the remaining 6 realizes the other 60 and one more commission -> 100097.
    A stale "qty*fill" cover would leave equity hundreds off and never catch it.
    """
    from src.bt.portfolio.pure import calculate_equity

    portfolio = create_initial_portfolio(
        initial_capital=100000, start_timestamp=_ts("2024-01-01")
    )
    portfolio = apply_fill(portfolio, _fill_short(10.0, 100.0, "SNPS_1"))
    partial = FillEvent(
        signal=TradeSignal(
            action=ActionType.rebalance,
            symbol="AAPL",
            timestamp=_ts("2024-01-02"),
            price=90.0,
            qty=-4.0,
            reason="partial-cover",
            position_id="SNPS_1",
        ),
        filled_qty=4.0,
        executed_price=90.0,
        commission=1.0,
        slippage=0.0,
        timestamp=_ts("2024-01-02"),
    )
    portfolio = apply_fill(portfolio, partial)
    assert calculate_equity(portfolio) == pytest.approx(100098.0)

    final = FillEvent(
        signal=TradeSignal(
            action=ActionType.rebalance,
            symbol="AAPL",
            timestamp=_ts("2024-01-03"),
            price=90.0,
            qty=-6.0,
            reason="full-cover",
            position_id="SNPS_1",
        ),
        filled_qty=6.0,
        executed_price=90.0,
        commission=1.0,
        slippage=0.0,
        timestamp=_ts("2024-01-03"),
    )
    portfolio = apply_fill(portfolio, final)
    assert "AAPL" not in portfolio.positions
    assert portfolio.cash == pytest.approx(100097.0)
    assert calculate_equity(portfolio) == pytest.approx(100097.0)


def test_long_partial_reduce_cash_and_equity():
    """Long partial reduce cash/equity is exact already; pin the symmetry.

    Long and short covers of the same realized + unrealized path land the same
    equity (100098 here) — a regression tripping only the short branch must not
    silently move the long side.
    """
    from src.bt.portfolio.pure import calculate_equity

    portfolio = create_initial_portfolio(
        initial_capital=100000, start_timestamp=_ts("2024-01-01")
    )
    portfolio = apply_fill(portfolio, _fill_long(10.0, 100.0, "AAPL_1"))
    reduce = FillEvent(
        signal=TradeSignal(
            action=ActionType.rebalance,
            symbol="AAPL",
            timestamp=_ts("2024-01-02"),
            price=110.0,
            qty=-4.0,
            reason="partial",
            position_id="AAPL_1",
        ),
        filled_qty=4.0,
        executed_price=110.0,
        commission=1.0,
        slippage=0.0,
        timestamp=_ts("2024-01-02"),
    )
    portfolio = apply_fill(portfolio, reduce)
    # cash = 100000 - open (10*100+1) + cover (4*110-1) = 99438.
    assert portfolio.cash == pytest.approx(99438.0)
    # remaining 6@100 last 110 -> value 660; equity = 99438 + 660 = 100098.
    assert calculate_equity(portfolio) == pytest.approx(100098.0)
