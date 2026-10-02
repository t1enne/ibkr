"""Tests for the simulated broker edge (fill parity with the backtest core)."""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from src.bt.execution.pure import execute_signal
from src.bt.portfolio.pure import apply_fill
from src.bt.state import (
    ActionType,
    Candle,
    ExecutionParams,
    PortfolioState,
)
from src.live.broker import OrderResult, SimulatedBroker, intent_to_signal
from src.live.result import Ok, Result
from src.live.types import FeedError, OrderIntent

TS = cast("pd.Timestamp", pd.Timestamp("2024-06-03"))

PlaceResult = Result[OrderResult, FeedError]


def _book() -> PortfolioState:
    return PortfolioState(
        cash=100_000.0,
        positions={},
        trades=(),
        equity_curve=(),
        initial_capital=100_000.0,
    )


def _broker(book: PortfolioState | None = None) -> SimulatedBroker:
    seed = _book() if book is None else book
    return SimulatedBroker(seed, ExecutionParams(), lambda _m: None)


def _open_intent(
    action: ActionType = ActionType.long, qty: float = 10.0
) -> OrderIntent:
    return OrderIntent(
        symbol="AAPL",
        action=action,
        qty=qty,
        ref_price=100.0,
        reason="open long (flat->long)",
    )


def _ref_candle() -> Candle:
    return Candle(
        timestamp=TS,
        symbol="AAPL",
        open=100.0,
        high=100.0,
        low=100.0,
        close=100.0,
        volume=0.0,
        interval=None,
    )


@pytest.mark.asyncio
async def test_place_matches_execute_signal() -> None:
    broker = _broker()
    intent = _open_intent()
    result: PlaceResult = await broker.place(intent)
    assert isinstance(result, Ok)
    order = cast(OrderResult, result.value)
    assert order.ok
    assert order.fill is not None

    expected = execute_signal(
        intent_to_signal(intent, TS), _ref_candle(), ExecutionParams()
    )
    assert order.fill.executed_price == expected.executed_price
    assert order.fill.slippage == expected.slippage
    assert order.fill.commission == expected.commission


@pytest.mark.asyncio
async def test_place_settles_book_like_backtest() -> None:
    broker = _broker()
    result: PlaceResult = await broker.place(_open_intent(qty=10.0))
    assert isinstance(result, Ok)
    order = cast(OrderResult, result.value)
    fill = order.fill
    assert fill is not None

    fresh = apply_fill(_book(), fill)
    assert fresh.cash == 100_000.0 - (10.0 * fill.executed_price + fill.commission)
    lots = fresh.positions["AAPL"]
    assert len(lots) == 1
    assert lots[0].type is ActionType.long
    assert lots[0].qty == 10.0
    # The broker's held book tracks the same settlement.
    assert broker.portfolio() == fresh


@pytest.mark.asyncio
async def test_open_exceeding_cash_is_rejected_book_unchanged() -> None:
    book = PortfolioState(
        cash=100.0,
        positions={},
        trades=(),
        equity_curve=(),
        initial_capital=100.0,
    )
    broker = _broker(book)
    # 10 * 100 = 1000 notional >> 100 cash: the shared cash guard drops it.
    result: PlaceResult = await broker.place(_open_intent(qty=10.0))
    assert isinstance(result, Ok)
    order = cast(OrderResult, result.value)
    assert order.ok is False
    assert order.fill is None
    assert "open rejected" in order.message
    assert broker.portfolio() == book  # held book untouched


@pytest.mark.asyncio
async def test_close_without_position_id_is_rejected_not_raised() -> None:
    broker = _broker()
    intent = OrderIntent(
        symbol="AAPL",
        action=ActionType.close,
        qty=5.0,
        ref_price=100.0,
        reason="close lot None",
        position_id=None,
    )
    result: PlaceResult = await broker.place(intent)
    assert isinstance(result, Ok)
    order = cast(OrderResult, result.value)
    assert order.ok is False
    assert order.fill is None
    assert "position_id" in order.message
    assert broker.portfolio() == _book()


@pytest.mark.asyncio
async def test_seed_replaces_book() -> None:
    broker = _broker()
    replacement = PortfolioState(
        cash=1.0,
        positions={},
        trades=(),
        equity_curve=(),
        initial_capital=1.0,
    )
    broker.seed(replacement)
    assert broker.portfolio() is replacement
    closed = await broker.close()
    assert isinstance(closed, Ok)
    assert closed.value is None


@pytest.mark.asyncio
async def test_open_assigns_synthetic_position_id() -> None:
    broker = _broker()
    result: PlaceResult = await broker.place(_open_intent())
    assert isinstance(result, Ok)
    order = cast(OrderResult, result.value)
    pid = order.position_id
    assert pid is not None
    assert pid.startswith("AAPL_")


def test_intent_to_signal_prices_at_ref() -> None:
    signal = intent_to_signal(_open_intent(), TS)
    assert signal.fill_at_next_open is False
    assert signal.price == 100.0
    assert signal.qty == 10.0
    assert signal.action is ActionType.long


def test_order_result_shape() -> None:
    order = OrderResult(intent=_open_intent(), fill=None, ok=False, message="x")
    assert order.position_id is None
    assert not order.ok
