"""Tests for pure portfolio functions — critical paths only."""

from typing import cast

import pandas as pd
import pytest

from src.bt.portfolio.pure import (
    apply_fill,
    update_prices,
    mark_to_market_list,
)
from src.bt.state import (
    ActionType,
    Candle,
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


def test_update_prices():
    position = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=_ts("2024-01-01"),
        stop_loss=95.0,
        take_profit=110.0,
        last_price=100.0,
        type=ActionType.long,
    )
    portfolio = PortfolioState(
        cash=5000,
        positions={"AAPL": (position,)},
        trades=(),
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
    tick = Candle(
        timestamp=_ts("2024-01-02"),
        symbol="AAPL",
        open=100.0,
        high=105.0,
        low=99.0,
        close=105.0,
        volume=1000.0,
    )
    new = update_prices(portfolio, tick)
    aapl = new.positions["AAPL"]
    assert len(aapl) == 1
    assert aapl[0].last_price == 105.0
    assert len(new.equity_curve) == 2


def test_mark_to_market_list_matches_update_prices():
    """Engine-buffered mark-to-market must emit identical equity points but
    without rebuilding the O(n) immutable tuple each candle."""
    position = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=_ts("2024-01-01"),
        stop_loss=95.0,
        take_profit=110.0,
        last_price=100.0,
        type=ActionType.long,
    )
    portfolio = PortfolioState(
        cash=5000,
        positions={"AAPL": (position,)},
        trades=(),
        equity_curve=(),
        initial_capital=10000,
    )
    ticks = [
        Candle(
            timestamp=_ts(f"2024-01-0{i}"),
            symbol="AAPL",
            open=100.0,
            high=105.0,
            low=99.0,
            close=100 + i,
            volume=1000.0,
        )
        for i in range(1, 4)
    ]

    # Reference: tuple-backed update_prices accumulates the full curve.
    ref = portfolio
    for tick in ticks:
        ref = update_prices(ref, tick)
    assert [p.equity for p in ref.equity_curve] == [6010.0, 6020.0, 6030.0]

    # Engine path: same points land in the caller-owned buffer, interim
    # portfolio carries an empty curve (frozen to tuple at finalize).
    buf: list = []
    cur = portfolio
    for tick in ticks:
        cur = mark_to_market_list(cur, tick, buf)

    assert [p.equity for p in buf] == [p.equity for p in ref.equity_curve]
    assert [p.timestamp for p in buf] == [p.timestamp for p in ref.equity_curve]
    assert [p.cash for p in buf] == [p.cash for p in ref.equity_curve]
    # Interim portfolio keeps updated positions but a placeholder curve.
    assert cur.positions["AAPL"][0].last_price == ticks[-1].close
    assert cur.equity_curve == ()
    # Freezing the buffer yields exactly the reference tuple.
    assert tuple(buf) == ref.equity_curve


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


def test_long_always_opens_fresh_lot():
    from src.bt.portfolio.pure import count_positions, get_symbol_positions

    portfolio = create_initial_portfolio(
        initial_capital=10000, start_timestamp=_ts("2024-01-01")
    )
    portfolio = apply_fill(portfolio, _fill_long(10.0, 100.0, "AAPL_1", tag="r1"))
    portfolio = apply_fill(portfolio, _fill_long(5.0, 100.0, "AAPL_2", tag="r2"))
    lots = get_symbol_positions(portfolio, "AAPL")
    assert count_positions(portfolio) == 2
    assert [p.position_id for p in lots] == ["AAPL_1", "AAPL_2"]
    assert [p.tag for p in lots] == ["r1", "r2"]
    assert lots[-1].position_id == "AAPL_2"  # newest lot


def test_partial_close_releases_fraction_of_lot():
    from src.bt.portfolio.pure import get_symbol_positions

    portfolio = create_initial_portfolio(
        initial_capital=10000, start_timestamp=_ts("2024-01-01")
    )
    portfolio = apply_fill(portfolio, _fill_long(10.0, 100.0, "AAPL_1", tag="r1"))
    portfolio = apply_fill(portfolio, _fill_long(10.0, 100.0, "AAPL_2", tag="r2"))

    # Release 25% of lot r1 only (2.5 shares at 100 => +250 cash, PnL zero).
    reduce = FillEvent(
        signal=TradeSignal(
            action=ActionType.rebalance,
            symbol="AAPL",
            timestamp=_ts("2024-01-02"),
            price=100.0,
            qty=-2.5,
            reason="partial",
            position_id="AAPL_1",
        ),
        filled_qty=2.5,
        executed_price=100.0,
        commission=1.0,
        slippage=0.0,
        timestamp=_ts("2024-01-02"),
    )
    portfolio = apply_fill(portfolio, reduce)
    lots = get_symbol_positions(portfolio, "AAPL")
    assert len(lots) == 2
    assert lots[0].position_id == "AAPL_1"
    assert lots[0].qty == pytest.approx(7.5)
    assert lots[0].tag == "r1"  # tag preserved across partial close
    assert lots[1].position_id == "AAPL_2"
    assert lots[1].qty == 10.0
    # The partially-closed trade remains open with the released PnL booked.
    r1_trade = [t for t in portfolio.trades if t.position_id == "AAPL_1"][0]
    assert r1_trade.status == TradeStatus.open
    assert r1_trade.qty == pytest.approx(7.5)


def test_partial_close_full_release_closes_lot():
    from src.bt.portfolio.pure import get_symbol_positions

    portfolio = create_initial_portfolio(
        initial_capital=10000, start_timestamp=_ts("2024-01-01")
    )
    portfolio = apply_fill(portfolio, _fill_long(10.0, 100.0, "AAPL_1", tag="r1"))
    portfolio = apply_fill(portfolio, _fill_long(10.0, 100.0, "AAPL_2", tag="r2"))

    # Release 100% of r1 => fully closes that lot, leaves r2 untouched.
    reduce = FillEvent(
        signal=TradeSignal(
            action=ActionType.rebalance,
            symbol="AAPL",
            timestamp=_ts("2024-01-02"),
            price=110.0,
            qty=-10.0,
            reason="full",
            position_id="AAPL_1",
        ),
        filled_qty=10.0,
        executed_price=110.0,
        commission=1.0,
        slippage=0.0,
        timestamp=_ts("2024-01-02"),
    )
    portfolio = apply_fill(portfolio, reduce)
    lots = get_symbol_positions(portfolio, "AAPL")
    assert len(lots) == 1
    assert lots[0].position_id == "AAPL_2"
    closed = [t for t in portfolio.trades if t.position_id == "AAPL_1"][0]
    assert closed.status == TradeStatus.closed
    # PnL is net of the round-trip commissions (entry 1.0 + close 1.0).
    assert closed.pnl == pytest.approx((110.0 - 100.0) * 10.0 - 2.0)


def test_open_short_stored_as_negative_net_bearing_magnitude():
    from src.bt.portfolio.pure import net_quantity

    portfolio = create_initial_portfolio(
        initial_capital=10000, start_timestamp=_ts("2024-01-01")
    )
    portfolio = apply_fill(portfolio, _fill_short(10.0, 100.0, "SNPS_1"))
    pos = portfolio.positions["AAPL"][0]
    # Position.qty is a positive MAGNITUDE; the side is the ``type``.
    assert pos.qty == pytest.approx(10.0)
    assert pos.type == ActionType.short
    assert net_quantity(portfolio, "AAPL") == pytest.approx(-10.0)


def test_partial_close_releases_fraction_of_short():
    """A short lot scales out by buying-to-cover a fraction of its magnitude.

    Mirrors the long partial-close behaviour on the magnitude model: reduces
    ``Position.qty`` in place, keeps the short type + tag + cost basis, books
    realized short PnL (entry - fill) on the covered shares, and stays open.
    """

    portfolio = create_initial_portfolio(
        initial_capital=100000, start_timestamp=_ts("2024-01-01")
    )
    portfolio = apply_fill(portfolio, _fill_short(10.0, 100.0, "SNPS_1", tag="r1"))

    # Cover 40% of the short (4 shares) at 90: short PnL = (100-90)*4.
    reduce = FillEvent(
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
    portfolio = apply_fill(portfolio, reduce)

    pos = portfolio.positions["AAPL"][0]
    assert len(portfolio.positions["AAPL"]) == 1  # never closed/duplicated
    assert pos.qty == pytest.approx(6.0)  # 40% covered, 60% remains
    assert pos.type == ActionType.short
    assert pos.tag == "r1"
    assert pos.entry_price == pytest.approx(100.0)  # basis unchanged on reduce
    trade = [t for t in portfolio.trades if t.position_id == "SNPS_1"][0]
    assert trade.status == TradeStatus.open
    assert trade.qty == pytest.approx(6.0)
    # Realized PnL on the covered shares; commission tracked separately on the
    # trade (matching test_close_long_position's convention).
    assert trade.pnl == pytest.approx((100.0 - 90.0) * 4.0)
    # CASH + EQUITY, not just pnl/qty (that gap let the mirror-basis bug slip):
    # a short partial cover must release the open's mirror cash
    # qty*(2*entry - fill) - comm, not qty*fill - comm. Reference: entry 10@100
    # comm 1, cover 4@90 comm 1 -> cash 99438, equity = 100000 + realized 40
    # + unrealized 60 - 2 comm = 100098.
    from src.bt.portfolio.pure import calculate_equity

    assert portfolio.cash == pytest.approx(
        100000 - (10 * 100.0 + 1) + (4 * (2 * 100.0 - 90.0) - 1)
    )
    assert calculate_equity(portfolio) == pytest.approx(100098.0)


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


def test_partial_close_full_cover_closes_short():
    from src.bt.portfolio.pure import net_quantity

    portfolio = create_initial_portfolio(
        initial_capital=100000, start_timestamp=_ts("2024-01-01")
    )
    portfolio = apply_fill(portfolio, _fill_short(10.0, 100.0, "SNPS_1", tag="r1"))

    reduce = FillEvent(
        signal=TradeSignal(
            action=ActionType.rebalance,
            symbol="AAPL",
            timestamp=_ts("2024-01-02"),
            price=90.0,
            qty=-10.0,
            reason="full-cover",
            position_id="SNPS_1",
        ),
        filled_qty=10.0,
        executed_price=90.0,
        commission=1.0,
        slippage=0.0,
        timestamp=_ts("2024-01-02"),
    )
    portfolio = apply_fill(portfolio, reduce)

    assert "AAPL" not in portfolio.positions  # fully covered short is gone
    assert net_quantity(portfolio, "AAPL") == 0.0
    closed = [t for t in portfolio.trades if t.position_id == "SNPS_1"][0]
    assert closed.status == TradeStatus.closed
    # PnL is net of the round-trip commissions (entry 1.0 + close 1.0).
    assert closed.pnl == pytest.approx((100.0 - 90.0) * 10.0 - 2.0)


def test_net_quantity_and_avg_entry():
    from src.bt.portfolio.pure import avg_entry, net_quantity

    portfolio = create_initial_portfolio(
        initial_capital=10000, start_timestamp=_ts("2024-01-01")
    )
    portfolio = apply_fill(portfolio, _fill_long(10.0, 100.0, "AAPL_1"))  # 1000 @ 100
    portfolio = apply_fill(portfolio, _fill_long(10.0, 120.0, "AAPL_2"))  # 1200 @ 120
    assert net_quantity(portfolio, "AAPL") == pytest.approx(20.0)
    assert avg_entry(portfolio, "AAPL") == pytest.approx(
        (10.0 * 100.0 + 10.0 * 120.0) / 20.0
    )
    assert net_quantity(portfolio, "MSFT") == 0.0
    assert avg_entry(portfolio, "MSFT") is None


def test_tag_preserved_through_price_update():
    portfolio = PortfolioState(
        cash=5000,
        positions={"AAPL": (_long_position("AAPL_1", 10.0, 100.0, tag="r1"),)},
        trades=(),
        equity_curve=(),
        initial_capital=10000,
    )
    tick = Candle(
        timestamp=_ts("2024-01-02"),
        symbol="AAPL",
        open=100.0,
        high=105.0,
        low=99.0,
        close=105.0,
        volume=1000.0,
    )
    new = update_prices(portfolio, tick)
    assert new.positions["AAPL"][0].tag == "r1"
