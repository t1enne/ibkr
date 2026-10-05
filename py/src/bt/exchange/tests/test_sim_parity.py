"""Parity: ``Fill.price`` means the SAME executed price on both surfaces.

``SimExchange.match_bar`` (the order-port path) and ``execute_signal`` (the
legacy signal path) must charge friction + commission identically, so one
``Fill.price``/``FillEvent.executed_price`` meaning holds across the adapter.
"""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from src.bt.exchange import SimExchange, execute_signal
from src.bt.state import ActionType, Candle, TradeSignal, create_execution_params
from src.exec.types import OrderRequest, OrderSide, OrderType

TS = cast("pd.Timestamp", pd.Timestamp("2025-01-02 15:00"))
_QTY = 10.0


def _params(spread_bps: float = 10.0, slippage_bps: float = 4.0):
    return create_execution_params(
        spread_bps=spread_bps, slippage_bps=slippage_bps, fixed_commission=0.5
    )


def _candle(open_: float, high: float, low: float, close: float) -> Candle:
    return Candle(
        timestamp=TS,
        symbol="AAPL",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=0.0,
    )


def _frame(candle: Candle) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [candle.open],
            "high": [candle.high],
            "low": [candle.low],
            "close": [candle.close],
        },
        index=pd.DatetimeIndex([candle.timestamp]),
    )


def _order(side: OrderSide, order_type: OrderType, limit: float | None = None):
    return OrderRequest(
        symbol="AAPL",
        side=side,
        qty=_QTY,
        order_type=order_type,
        order_ref="strat-20250102T1500-000",
        limit_price=limit,
    )


def _signal(action: ActionType) -> TradeSignal:
    return TradeSignal(
        action=action,
        symbol="AAPL",
        timestamp=TS,
        price=100.0,
        qty=_QTY,
        fill_at_next_open=True,
    )


def _match(ex: SimExchange, order: OrderRequest, candle: Candle, params):
    return ex.match_bar(
        order,
        _frame(candle),
        spread_bps=params.spread_bps,
        slippage_bps=params.slippage_bps,
        commission_model=params.commission_model,
    )


def test_mkt_match_equals_execute_signal_next_open() -> None:
    params = _params()
    # close < open: an adverse bar for a long, so the 1.5x slippage multiplier
    # must be applied on BOTH paths.
    candle = _candle(100.0, 105.0, 95.0, 99.0)
    ex = SimExchange()
    order = _order(OrderSide.BUY, OrderType.MKT)
    ex.submit(order)
    fill = _match(ex, order, candle, params)
    event = execute_signal(_signal(ActionType.long), candle, params)

    assert fill is not None
    assert fill.price == event.executed_price
    assert fill.commission == event.commission
    assert fill.spread == event.spread
    assert fill.slippage == event.slippage
    assert fill.price == pytest.approx(100.11)  # base + half-spread + 1.5*slip


def test_mkt_sell_match_equals_execute_signal_short() -> None:
    params = _params()
    candle = _candle(100.0, 105.0, 95.0, 101.0)  # close > open: adverse for short
    ex = SimExchange()
    order = _order(OrderSide.SELL, OrderType.MKT)
    ex.submit(order)
    fill = _match(ex, order, candle, params)
    event = execute_signal(_signal(ActionType.short), candle, params)

    assert fill is not None
    assert fill.price == event.executed_price
    assert fill.price == pytest.approx(99.89)  # base - half-spread - 1.5*slip


def test_lmt_gapped_through_fills_at_open_plus_friction() -> None:
    params = _params()
    # Limit 90 but the bar gapped open at 85: fill base is min(limit, open)=85,
    # NOT improved to the limit; friction is then applied on top.
    candle = _candle(85.0, 88.0, 84.0, 86.0)
    ex = SimExchange()
    order = _order(OrderSide.BUY, OrderType.LMT, limit=90.0)
    ex.submit(order)
    fill = _match(ex, order, candle, params)

    assert fill is not None
    assert fill.price == pytest.approx(85.0765)  # 85 + 85*5bp + 85*4bp
