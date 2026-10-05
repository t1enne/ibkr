"""Tests for the ``SimExchange`` broker adapter's order surface."""

from __future__ import annotations

import pandas as pd

from src.bt.exchange import SimExchange, default_exchange
from src.exec.types import (
    Fill,
    FixedCommission,
    OrderRequest,
    OrderSide,
    OrderState,
    OrderType,
    RejectReason,
)

TS = pd.Timestamp("2025-01-02 15:00")


def _order(
    side: OrderSide, order_type: OrderType, limit: float | None = None
) -> OrderRequest:
    return OrderRequest(
        symbol="AAPL",
        side=side,
        qty=10.0,
        order_type=order_type,
        order_ref="strat-20250102T1500-000",
        limit_price=limit,
    )


def _bars(open_: float, high: float, low: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": [open_], "high": [high], "low": [low], "close": [open_]},
        index=pd.DatetimeIndex([TS]),
    )


def _match(ex: SimExchange, order: OrderRequest) -> Fill | None:
    """Match with zero spread/slippage so executed price == the base price."""
    return ex.match_bar(
        order,
        _bars(100.0, 105.0, 95.0),
        spread_bps=0.0,
        slippage_bps=0.0,
        commission_model=FixedCommission(0.5),
    )


def test_match_bar_records_an_mkt_fill() -> None:
    ex = SimExchange()
    order = _order(OrderSide.BUY, OrderType.MKT)
    ex.submit(order)
    fill = _match(ex, order)
    assert fill is not None
    assert fill.price == 100.0  # zero friction -> executed == base
    assert fill.commission == 0.5
    assert ex.fills() == (fill,)


def test_match_bar_records_nothing_on_an_lmt_miss() -> None:
    ex = SimExchange()
    order = _order(OrderSide.BUY, OrderType.LMT, limit=90.0)
    ex.submit(order)
    assert _match(ex, order) is None
    assert ex.fills() == ()


def test_submit_then_match_clears_pending() -> None:
    ex = default_exchange()
    order = _order(OrderSide.BUY, OrderType.MKT)
    assert ex.submit(order).state is OrderState.PENDING
    assert _match(ex, order) is not None
    # A filled order is no longer cancellable: it left the pending set.
    assert ex.cancel(order.order_ref).reason is RejectReason.UNKNOWN_ORDER


def test_cancel_pending_order_succeeds() -> None:
    ex = SimExchange()
    order = _order(OrderSide.SELL, OrderType.LMT, limit=110.0)
    ex.submit(order)
    ack = ex.cancel(order.order_ref)
    assert ack.accepted
    assert ack.state is OrderState.CANCELLED


def test_cancel_unknown_order_is_rejected_not_raised() -> None:
    ack = SimExchange().cancel("nope")
    assert not ack.accepted
    assert ack.reason is RejectReason.UNKNOWN_ORDER


def test_close_drops_pending_orders() -> None:
    ex = SimExchange()
    order = _order(OrderSide.BUY, OrderType.MKT)
    ex.submit(order)
    ex.close()
    assert ex.cancel(order.order_ref).accepted is False


def test_cancelled_order_does_not_fill() -> None:
    ex = SimExchange()
    order = _order(OrderSide.BUY, OrderType.MKT)
    ex.submit(order)
    ex.cancel(order.order_ref)
    # A cancelled ref is no longer pending, so a bar touching it cannot fill.
    assert _match(ex, order) is None
    assert ex.fills() == ()


def test_unknown_ref_does_not_fill() -> None:
    ex = SimExchange()
    order = _order(OrderSide.BUY, OrderType.MKT)  # never submitted
    assert _match(ex, order) is None
    assert ex.fills() == ()


def test_lmt_without_limit_is_rejected_and_never_pending() -> None:
    ex = SimExchange()
    order = _order(OrderSide.BUY, OrderType.LMT, limit=None)
    ack = ex.submit(order)
    assert not ack.accepted
    assert ack.state is OrderState.REJECTED
    assert ack.reason is RejectReason.INVALID
    # The rejected order never entered the pending set, so it cannot fill.
    assert _match(ex, order) is None


def test_duplicate_order_ref_is_rejected_not_replaced() -> None:
    ex = SimExchange()
    order = _order(OrderSide.BUY, OrderType.MKT)
    assert ex.submit(order).accepted
    ack = ex.submit(order)
    assert not ack.accepted
    assert ack.reason is RejectReason.INVALID
