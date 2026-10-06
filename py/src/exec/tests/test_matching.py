"""Table tests for the pure candle matcher."""

from __future__ import annotations

import pandas as pd
import pytest

from src.exec.matching import match_bar, match_price
from src.exec.types import OrderRequest, OrderSide, OrderType, TimeInForce

TS = pd.Timestamp("2025-01-02 15:00")


def _order(side: OrderSide, order_type: OrderType, limit: float | None = None):
    return OrderRequest(
        symbol="AAPL",
        side=side,
        qty=10.0,
        order_type=order_type,
        order_ref="strat-20250102T1500-000",
        limit_price=limit,
        tif=TimeInForce.DAY,
    )


def _bars(open_: float, high: float, low: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": [open_], "high": [high], "low": [low], "close": [open_]},
        index=pd.DatetimeIndex([TS]),
    )


def test_mkt_fills_at_open() -> None:
    fill = match_bar(_order(OrderSide.BUY, OrderType.MKT), _bars(100.0, 105.0, 95.0))
    assert fill is not None
    assert fill.price == 100.0
    assert fill.qty == 10.0
    assert fill.symbol == "AAPL"
    assert fill.timestamp == TS


def test_lmt_buy_touched_fills_at_limit() -> None:
    order = _order(OrderSide.BUY, OrderType.LMT, limit=97.0)
    fill = match_bar(order, _bars(100.0, 105.0, 95.0))
    assert fill is not None
    assert fill.price == 97.0  # min(limit, open)


def test_lmt_buy_gap_through_fills_better_at_open() -> None:
    # Bar gapped through the limit in our favour: fill at the open (better).
    order = _order(OrderSide.BUY, OrderType.LMT, limit=90.0)
    fill = match_bar(order, _bars(80.0, 82.0, 75.0))
    assert fill is not None
    assert fill.price == 80.0  # min(90, 80): better than the limit


def test_lmt_buy_missed_is_none() -> None:
    order = _order(OrderSide.BUY, OrderType.LMT, limit=90.0)
    assert match_bar(order, _bars(100.0, 105.0, 95.0)) is None


def test_lmt_sell_touched_fills_at_limit() -> None:
    order = _order(OrderSide.SELL, OrderType.LMT, limit=103.0)
    fill = match_bar(order, _bars(100.0, 105.0, 95.0))
    assert fill is not None
    assert fill.price == 103.0  # max(limit, open)


def test_lmt_sell_gap_through_fills_better_at_open() -> None:
    order = _order(OrderSide.SELL, OrderType.LMT, limit=110.0)
    fill = match_bar(order, _bars(120.0, 125.0, 118.0))
    assert fill is not None
    assert fill.price == 120.0  # max(110, 120): better than the limit


def test_lmt_sell_missed_is_none() -> None:
    order = _order(OrderSide.SELL, OrderType.LMT, limit=110.0)
    assert match_bar(order, _bars(100.0, 105.0, 95.0)) is None


def test_lmt_without_limit_never_fills() -> None:
    order = _order(OrderSide.BUY, OrderType.LMT, limit=None)
    assert match_bar(order, _bars(100.0, 105.0, 95.0)) is None


def test_empty_bars_is_none() -> None:
    order = _order(OrderSide.BUY, OrderType.MKT)
    assert match_bar(order, pd.DataFrame()) is None


@pytest.mark.parametrize(
    ("order_type", "side", "limit", "open_", "high", "low", "expected"),
    [
        (OrderType.MKT, OrderSide.BUY, None, 100.0, 105.0, 95.0, 100.0),
        (OrderType.LMT, OrderSide.BUY, 97.0, 100.0, 105.0, 95.0, 97.0),
        (OrderType.LMT, OrderSide.BUY, 98.0, 100.0, 105.0, 99.0, None),
        # Exact touch (low == limit) is a fill: the matcher is non-strict.
        (OrderType.LMT, OrderSide.BUY, 95.0, 100.0, 105.0, 95.0, 95.0),
        # An LMT with no limit price never fills.
        (OrderType.LMT, OrderSide.BUY, None, 100.0, 105.0, 95.0, None),
        (OrderType.LMT, OrderSide.SELL, 103.0, 100.0, 105.0, 95.0, 103.0),
        # Exact touch (high == limit) is a fill: the matcher is non-strict.
        (OrderType.LMT, OrderSide.SELL, 105.0, 100.0, 105.0, 95.0, 105.0),
        (OrderType.LMT, OrderSide.SELL, 106.0, 100.0, 105.0, 95.0, None),
    ],
)
def test_match_price_table(order_type, side, limit, open_, high, low, expected) -> None:
    assert (
        match_price(_order(side, order_type, limit), open_=open_, high=high, low=low)
        == expected
    )
