"""Table tests for the pure IBKR order mappings (plan phase 3)."""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from src.bt.state import ActionType
from src.exec.refs import order_ref
from src.exec.types import OrderSide, OrderState, OrderType
from src.live.adapters.ibkr.orders import (
    UnknownCloseLot,
    UnsupportedOrderType,
    build_ticket,
    classify_reply,
    is_fully_filled,
    is_terminal,
    order_side,
    order_state,
    sequence,
    status_to_fill,
)
from src.live.types import OrderIntent

CYCLE_TS = cast("pd.Timestamp", pd.Timestamp("2024-06-03T14:30:00Z"))
STRATEGY = "abcdef1234567890"


def _intent(
    symbol: str = "AAPL",
    action: ActionType = ActionType.long,
    *,
    position_id: str | None = None,
    order_type: OrderType = OrderType.MKT,
    qty: float = 10.0,
) -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        action=action,
        qty=qty,
        ref_price=100.0,
        reason="test",
        position_id=position_id,
        order_type=order_type,
    )


# --- side resolution --------------------------------------------------------


@pytest.mark.parametrize(
    ("action", "position_side", "expected"),
    [
        (ActionType.long, None, OrderSide.BUY),
        (ActionType.short, None, OrderSide.SELL),
        (ActionType.close, ActionType.long, OrderSide.SELL),
        (ActionType.close, ActionType.short, OrderSide.BUY),
    ],
)
def test_order_side_table(
    action: ActionType, position_side: ActionType | None, expected: OrderSide
) -> None:
    assert order_side(_intent(action=action), position_side) is expected


@pytest.mark.parametrize("position_side", [None, ActionType.close])
def test_close_without_a_lot_side_is_an_error(
    position_side: ActionType | None,
) -> None:
    with pytest.raises(UnknownCloseLot, match="lot not found"):
        order_side(_intent(action=ActionType.close, position_id="99"), position_side)


# --- ticket body ------------------------------------------------------------


def test_build_ticket_mkt_body_and_deterministic_coid() -> None:
    ticket = build_ticket(
        _intent(),
        conid=265598,
        side=OrderSide.BUY,
        strategy_id=STRATEGY,
        cycle_ts=CYCLE_TS,
        seq=0,
    )
    assert ticket.body == {
        "conid": 265598,
        "side": "BUY",
        "quantity": 10.0,
        "orderType": "MKT",
        "tif": "DAY",
        "cOID": order_ref(STRATEGY, CYCLE_TS, 0),
    }
    assert ticket.side is OrderSide.BUY
    # Deterministic: the same inputs mint the same cOID, so IBKR dedupes a re-send.
    again = build_ticket(
        _intent(),
        conid=265598,
        side=OrderSide.BUY,
        strategy_id=STRATEGY,
        cycle_ts=CYCLE_TS,
        seq=0,
    )
    assert again.order_ref == ticket.order_ref
    # A different seq (a different intent in the cycle) does not.
    other = build_ticket(
        _intent(),
        conid=265598,
        side=OrderSide.BUY,
        strategy_id=STRATEGY,
        cycle_ts=CYCLE_TS,
        seq=1,
    )
    assert other.order_ref != ticket.order_ref


def test_build_ticket_close_uses_the_lot_side() -> None:
    intent = _intent(action=ActionType.close, position_id="55")
    side = order_side(intent, ActionType.long)
    ticket = build_ticket(
        intent,
        conid=1,
        side=side,
        strategy_id=STRATEGY,
        cycle_ts=CYCLE_TS,
        seq=0,
    )
    assert ticket.body["side"] == "SELL"


def test_build_ticket_rejects_lmt() -> None:
    with pytest.raises(UnsupportedOrderType, match="phase 4"):
        build_ticket(
            _intent(order_type=OrderType.LMT),
            conid=1,
            side=OrderSide.BUY,
            strategy_id=STRATEGY,
            cycle_ts=CYCLE_TS,
            seq=0,
        )


# --- deterministic sequencing ----------------------------------------------


def test_sequence_puts_closes_first_and_is_stable() -> None:
    opens = (_intent("AAPL"), _intent("MSFT"))
    closes = (
        _intent("MSFT", ActionType.close, position_id="2"),
        _intent("AAPL", ActionType.close, position_id="1"),
    )
    # reconcile emits closes-then-opens; sequence must not disturb either group.
    pairs = sequence(closes + opens)
    assert [intent.symbol for _seq, intent in pairs] == ["MSFT", "AAPL", "AAPL", "MSFT"]
    assert [seq for seq, _intent_ in pairs] == [0, 1, 2, 3]
    assert [intent.action for _seq, intent in pairs][:2] == [
        ActionType.close,
        ActionType.close,
    ]
    # Re-running the same cycle yields the identical mapping (same cOIDs).
    assert sequence(closes + opens) == pairs


# --- reply classification ---------------------------------------------------


def test_classify_reply_success() -> None:
    outcome = classify_reply(
        [
            {
                "order_id": "97932.0",
                "order_status": "PreSubmitted",
                "encrypt_message": "1",
            }
        ]
    )
    assert outcome.kind == "success"
    assert outcome.order_id == "97932"  # canonicalised str(int(...))


def test_classify_reply_confirmation() -> None:
    outcome = classify_reply(
        [
            {
                "id": "99097238-9824-4830-84ef-46979aa22593",
                "isSuppressed": False,
                "message": ["Are you sure you want to submit this order?"],
                "messageIds": ["o354"],
            }
        ]
    )
    assert outcome.kind == "confirm"
    assert outcome.reply_id == "99097238-9824-4830-84ef-46979aa22593"
    assert "Are you sure" in outcome.message


def test_classify_reply_error_aborts_verbatim() -> None:
    outcome = classify_reply({"error": "Order not confirmed "})
    assert outcome.kind == "abort"
    assert outcome.message == "Order not confirmed "


def test_classify_reply_advanced_reject_aborts_with_its_text() -> None:
    outcome = classify_reply(
        {
            "orderId": 123456789,
            "text": "price band exceeded",
            "options": ["Use on this order", "Do not use"],
            "prompt": True,
            "type": "M",
        }
    )
    assert outcome.kind == "abort"
    assert outcome.message == "price band exceeded"


@pytest.mark.parametrize("payload", [[], {}, "nonsense", None])
def test_classify_reply_unknown_shape_aborts(payload: object) -> None:
    assert classify_reply(payload).kind == "abort"


def test_classify_reply_id_without_message_aborts() -> None:
    # An id we cannot read is not a confirmation we are willing to give.
    assert classify_reply({"id": "uuid-1"}).kind == "abort"


# --- status -> fill / terminal ---------------------------------------------


@pytest.mark.parametrize(
    ("status", "terminal"),
    [
        ("Filled", True),
        ("Cancelled", True),
        ("Inactive", True),
        ("Submitted", False),
        ("PreSubmitted", False),
        ("PendingSubmit", False),
        ("PendingCancel", False),
        ("WarnState", False),
        ("SomethingNew", False),
        ("", False),
    ],
)
def test_is_terminal_table(status: str, terminal: bool) -> None:
    assert is_terminal({"order_status": status}) is terminal


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("Filled", OrderState.FILLED),
        ("Cancelled", OrderState.CANCELLED),
        ("Inactive", OrderState.REJECTED),
        ("Submitted", OrderState.PENDING),
        ("WarnState", OrderState.PENDING),  # unknown/transient never reads as loaded
        ("Whatever", OrderState.PENDING),
    ],
)
def test_order_state_table(status: str, expected: OrderState) -> None:
    assert order_state({"order_status": status}) is expected


def test_status_to_fill_reads_cum_fill_and_average_price() -> None:
    fill = status_to_fill(
        {"order_status": "Filled", "cum_fill": "3", "average_price": "99.5"},
        order_ref="ref-1",
        symbol="AAPL",
        side=OrderSide.BUY,
    )
    assert fill is not None
    assert (fill.qty, fill.price, fill.order_ref, fill.symbol) == (
        3.0,
        99.5,
        "ref-1",
        "AAPL",
    )


def test_status_to_fill_none_when_nothing_filled() -> None:
    assert (
        status_to_fill(
            {"order_status": "Submitted", "cum_fill": "0"},
            order_ref="ref-1",
            symbol="AAPL",
            side=OrderSide.BUY,
        )
        is None
    )


def test_is_fully_filled_requires_a_full_terminal_fill() -> None:
    full = {"order_status": "Filled", "cum_fill": "10", "total_size": "10"}
    assert is_fully_filled(full, 10.0)
    # Terminal but short of the requested size is NOT a success.
    assert not is_fully_filled(
        {"order_status": "Filled", "cum_fill": "4", "total_size": "10"}, 4.0
    )
    assert not is_fully_filled({"order_status": "Cancelled"}, 0.0)
    # A body with no total_size cannot contradict a terminal Filled.
    assert is_fully_filled({"order_status": "Filled", "cum_fill": "10"}, 10.0)
