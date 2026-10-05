"""Execution replay: net/VWAP, foreign exclusion, open/closed — all pure."""

from __future__ import annotations

import pandas as pd
import pytest
from typing import cast

from src.exec.types import OrderSide
from src.live.adapters.ibkr.trades import Execution, ReplayedLot, replay

PREFIX = "abc12345"


def _exec(
    order_id: str,
    *,
    side: OrderSide,
    qty: float,
    price: float,
    ref: str = f"{PREFIX}-deadbeef-20240101T0900-000",
    ts: str = "2024-01-02T09:30:00Z",
    commission: float = 1.0,
    symbol: str = "AAPL",
) -> Execution:
    return Execution(
        execution_id=f"exec-{order_id}-{ts}",
        order_id=order_id,
        order_ref=ref,
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        commission=commission,
        ts=cast("pd.Timestamp", pd.Timestamp(ts)),
    )


def test_single_buy_is_an_open_lot() -> None:
    book = replay(
        (_exec("o1", side=OrderSide.BUY, qty=10, price=100),), ref_prefix=PREFIX
    )
    assert len(book.lots) == 1
    lot = book.lots[0]
    assert isinstance(lot, ReplayedLot)
    assert lot.status == "open"
    assert lot.side is OrderSide.BUY
    assert lot.qty == 10
    assert lot.entry_price == 100
    assert lot.position_id == "o1"


def test_entry_price_is_vwap_of_opening_executions_only() -> None:
    # Open 10@100 then 10@120, partial close 5@130. Entry VWAP ignores the close.
    book = replay(
        (
            _exec(
                "o1", side=OrderSide.BUY, qty=10, price=100, ts="2024-01-02T09:30:00Z"
            ),
            _exec(
                "o1", side=OrderSide.BUY, qty=10, price=120, ts="2024-01-02T09:40:00Z"
            ),
            _exec(
                "o1", side=OrderSide.SELL, qty=5, price=130, ts="2024-01-02T10:00:00Z"
            ),
        ),
        ref_prefix=PREFIX,
    )
    lot = book.lots[0]
    assert lot.qty == 15  # net
    assert lot.entry_price == pytest.approx(110.0)  # (10*100 + 10*120) / 20
    assert lot.status == "open"


def test_scaled_out_to_zero_is_closed() -> None:
    book = replay(
        (
            _exec(
                "o1", side=OrderSide.BUY, qty=10, price=100, ts="2024-01-02T09:30:00Z"
            ),
            _exec(
                "o1", side=OrderSide.SELL, qty=10, price=110, ts="2024-01-02T15:00:00Z"
            ),
        ),
        ref_prefix=PREFIX,
    )
    lot = book.lots[0]
    assert lot.status == "closed"
    assert lot.qty == 0
    assert lot.entry_price == 100  # the opening (buy) executions


def test_short_position_is_negative_net() -> None:
    book = replay(
        (_exec("s1", side=OrderSide.SELL, qty=25, price=50),), ref_prefix=PREFIX
    )
    lot = book.lots[0]
    assert lot.side is OrderSide.SELL
    assert lot.qty == 25
    assert lot.status == "open"


def test_one_lot_per_order_id() -> None:
    book = replay(
        (
            _exec("o1", side=OrderSide.BUY, qty=10, price=100),
            _exec("o2", side=OrderSide.BUY, qty=3, price=90),
        ),
        ref_prefix=PREFIX,
    )
    assert {lot.order_id for lot in book.lots} == {"o1", "o2"}


def test_foreign_orders_are_excluded_and_reported() -> None:
    book = replay(
        (
            _exec("mine", side=OrderSide.BUY, qty=10, price=100),
            _exec(
                "theirs",
                side=OrderSide.BUY,
                qty=99,
                price=1,
                ref="otherstr-corrupt-20240101T0900-000",
            ),
        ),
        ref_prefix=PREFIX,
    )
    assert [lot.order_id for lot in book.lots] == ["mine"]
    assert [e.order_id for e in book.foreign] == ["theirs"]
    # A foreign execution never reaches the strategy book.
    assert all(lot.order_id != "theirs" for lot in book.lots)


def test_commission_is_summed_per_order() -> None:
    book = replay(
        (
            _exec("o1", side=OrderSide.BUY, qty=10, price=100, commission=1.5),
            _exec("o1", side=OrderSide.SELL, qty=10, price=110, commission=2.5),
        ),
        ref_prefix=PREFIX,
    )
    assert book.lots[0].commission == 4.0


def test_input_order_does_not_matter() -> None:
    first = _exec(
        "o1", side=OrderSide.BUY, qty=10, price=100, ts="2024-01-02T09:30:00Z"
    )
    second = _exec(
        "o1", side=OrderSide.SELL, qty=4, price=130, ts="2024-01-02T11:00:00Z"
    )
    assert (
        replay((first, second), ref_prefix=PREFIX).lots
        == replay((second, first), ref_prefix=PREFIX).lots
    )


def test_empty_executions_yield_empty_book() -> None:
    book = replay((), ref_prefix=PREFIX)
    assert book.lots == () and book.foreign == () and book.warnings == ()


def test_empty_ref_prefix_is_rejected() -> None:
    with pytest.raises(ValueError, match="ref_prefix"):
        replay((), ref_prefix="")
