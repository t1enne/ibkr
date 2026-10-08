"""Table tests for the pure IBKR order mappings (plan phase 3)."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from src.bt.exchange import execute_signal
from src.bt.state import ActionType, ExecutionParams
from src.exec.types import (
    FixedCommission,
    OrderSide,
    OrderState,
    OrderType,
)
from src.live.adapters.ibkr.orders import (
    OrderMappingError,
    UnknownCloseLot,
    UnsupportedOrderType,
    UnsupportedStopOrder,
    build_ticket,
    classify_reply,
    is_fully_filled,
    match_working,
    order_side,
    order_state,
    parse_working_order,
    placement_order,
    scale_open_cohort,
    status_to_fill,
)
from src.live.pure import intent_to_signal, ref_candle
from src.live.identity import (
    WorkingOrder,
    intent_key,
    order_ref,
    ref_is_ours,
    ref_matches_key,
    ref_prefix,
)
from src.live.types import OrderIntent

CYCLE_TS = cast("pd.Timestamp", pd.Timestamp("2024-06-03T14:30:00Z"))
SCOPE = "momentum"
_FIXTURES = Path(__file__).parent / "fixtures"


def _captured_working_orders() -> list[dict[str, object]]:
    """The orders captured from a real Paper gateway (build 10.50.1a).

    Row 0 is OURS (carries ``order_ref``); rows 1+ are foreign (NO ``order_ref``).
    The field set is the gateway's, so a test can no longer invent a field — the
    hand-built ``cOID`` mocks hid the parser reading the wrong key.
    """
    raw = json.loads((_FIXTURES / "gateway_working_orders.json").read_text())
    return cast("list[dict[str, object]]", raw["orders"])


def replace_intent(intent: OrderIntent, **changes: object) -> OrderIntent:
    """A frozen-intent copy with *changes* applied (test convenience)."""
    return replace(intent, **changes)


#: Zero-friction, zero-commission params: the simple scale cases (requested at
#: the reference price, no reserve) reduce to the pre-fix arithmetic.
_FLAT_PARAMS = ExecutionParams(
    spread_bps=0.0, slippage_bps=0.0, commission_model=FixedCommission(0.0)
)


def _intent(
    symbol: str = "AAPL",
    action: ActionType = ActionType.long,
    *,
    position_id: str | None = None,
    order_type: OrderType = OrderType.MKT,
    qty: float = 10.0,
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        action=action,
        qty=qty,
        ref_price=100.0,
        reason="test",
        position_id=position_id,
        order_type=order_type,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )


# --- side resolution --------------------------------------------------------


def test_close_without_a_lot_side_is_an_error() -> None:
    with pytest.raises(UnknownCloseLot, match="lot not found"):
        order_side(_intent(action=ActionType.close, position_id="99"), None)


# --- ticket body ------------------------------------------------------------


def test_ref_is_bar_free_and_keyed_on_the_intent() -> None:
    # INV-2: the cOID is a pure function of (scope, symbol, action, position_id).
    # It never embeds the decision bar, so a re-run on a LATER bar re-mints the
    # SAME ref and the working-order pre-flight adopts it instead of duplicating.
    early = _intent()
    late = replace_intent(_intent(), decision_ts=CYCLE_TS)
    assert intent_key(SCOPE, early) == intent_key(SCOPE, late)
    assert order_ref(intent_key(SCOPE, early), 0) == order_ref(
        intent_key(SCOPE, late), 0
    )
    # A different intent (different symbol) mints a different ref.
    assert order_ref(intent_key(SCOPE, _intent("MSFT")), 0) != order_ref(
        intent_key(SCOPE, early), 0
    )
    # A different attempt on the SAME key mints a different ref (so IBKR's own
    # dedupe cannot swallow a legitimate re-send).
    key = intent_key(SCOPE, early)
    assert order_ref(key, 0) != order_ref(key, 1)
    assert ref_prefix(key) == order_ref(key, 0)[:-2]


def test_build_ticket_close_floors_the_quantity() -> None:
    # A reducing order must never round UP: a 1.6-share close sends 1 share, not
    # 2 (2 would flip the 1.6 long into a 0.4 short — finding 2).
    intent = _intent(action=ActionType.close, position_id="55", qty=1.6)
    ref = order_ref(intent_key(SCOPE, intent), 0)
    ticket = build_ticket(
        intent, conid=1, side=order_side(intent, ActionType.long), order_ref=ref
    )
    assert ticket.body["quantity"] == 1.0
    assert ticket.rounded


def test_build_ticket_close_that_floors_to_zero_is_refused() -> None:
    # A sub-share close floors to 0: refuse it, never send 1 (which would flip).
    intent = _intent(action=ActionType.close, position_id="55", qty=0.6)
    with pytest.raises(OrderMappingError, match="non-positive whole-share"):
        build_ticket(
            intent,
            conid=1,
            side=order_side(intent, ActionType.long),
            order_ref=order_ref(intent_key(SCOPE, intent), 0),
        )


def test_build_ticket_rejects_lmt() -> None:
    with pytest.raises(UnsupportedOrderType, match="phase 4"):
        build_ticket(
            _intent(order_type=OrderType.LMT),
            conid=1,
            side=OrderSide.BUY,
            order_ref=order_ref(intent_key(SCOPE, _intent()), 0),
        )


@pytest.mark.parametrize(
    ("stop_loss", "take_profit"),
    [(95.0, None), (None, 110.0), (95.0, 110.0)],
)
def test_build_ticket_refuses_an_intent_carrying_a_stop(
    stop_loss: float | None, take_profit: float | None
) -> None:
    # The adapter places no resting stop: refusing beats silently sending a naked
    # order whose risk levels the report would then imply were honoured.
    with pytest.raises(UnsupportedStopOrder, match="naked order"):
        build_ticket(
            _intent(stop_loss=stop_loss, take_profit=take_profit),
            conid=1,
            side=OrderSide.BUY,
            order_ref=order_ref(intent_key(SCOPE, _intent()), 0),
        )


# --- placement order + working-order matching -------------------------------


def test_placement_order_puts_closes_first_deterministically() -> None:
    opens = (_intent("AAPL"), _intent("MSFT"))
    closes = (
        _intent("MSFT", ActionType.close, position_id="2"),
        _intent("AAPL", ActionType.close, position_id="1"),
    )
    assert [i.symbol for i in placement_order(closes + opens)] == [
        "MSFT",
        "AAPL",
        "AAPL",
        "MSFT",
    ]
    assert placement_order(closes + opens) == placement_order(closes + opens)


def test_parse_working_order_reads_the_captured_order_ref_and_order_id() -> None:
    # Measured live: the gateway echoes the client order id in ``order_ref`` (NOT
    # ``cOID``). Driven by the captured row so a mock cannot invent the field.
    parsed = parse_working_order(_captured_working_orders()[0])
    assert parsed is not None
    assert parsed.order_ref == "probe-9b9e2e"
    assert parsed.order_id == "453346548"  # canonicalised str(int(...))
    assert parsed.symbol == "AAPL"
    assert parsed.conid == 265598
    assert parsed.status == "PreSubmitted"


def test_captured_ours_row_is_matched_and_foreign_rows_are_never_adopted() -> None:
    key = intent_key(SCOPE, _intent("AAPL"))
    prefix = ref_prefix(key)
    rows = _captured_working_orders()
    ours = dict(rows[0])
    ours["order_ref"] = order_ref(key, 0)
    parsed: list[WorkingOrder] = []
    for row in (ours, *rows[1:]):
        row_parsed = parse_working_order(row)
        if row_parsed is not None:
            parsed.append(row_parsed)
    assert [p.order_ref for p in parsed] == [order_ref(key, 0)]  # foreign rows dropped
    match = match_working(parsed, prefix)
    assert match is not None and match.order_ref == order_ref(key, 0)


# --- reply classification ---------------------------------------------------


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


# --- status -> fill / terminal ---------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("Inactive", OrderState.REJECTED),
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


def test_is_fully_filled_when_filled_reaches_total_before_the_status_flips() -> None:
    # cum_fill == total_size while the order still reads Submitted: a fill cannot
    # exceed the size, so it is a complete fill regardless of the status string
    # (finding M7b).
    assert is_fully_filled(
        {"order_status": "Submitted", "cum_fill": "10", "total_size": "10"}, 10.0
    )
    # A non-terminal body short of total_size is NOT yet a full fill.
    assert not is_fully_filled(
        {"order_status": "Submitted", "cum_fill": "4", "total_size": "10"}, 4.0
    )


def test_is_fully_filled_false_when_a_terminal_filled_has_no_readable_fill() -> None:
    # A terminal Filled with no readable cum_fill cannot be proven complete; the
    # caller reports it UNKNOWN rather than as a full fill (finding M7a).
    assert not is_fully_filled({"order_status": "Filled", "total_size": "10"}, 0.0)


# --- cohort cash scaling (backtest parity) ----------------------------------


def _open(symbol: str, qty: float, price: float, bound: float) -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        action=ActionType.long,
        qty=qty,
        ref_price=price,
        reason="open",
        cash_bound=bound,
    )


def test_lone_open_is_never_scaled() -> None:
    plan = scale_open_cohort((_open("AAPL", 100.0, 100.0, 5000.0),), _FLAT_PARAMS)
    assert plan.scale is None and not plan.dropped
    assert plan.intents[0].qty == 100.0


def test_multi_open_cohort_scales_to_the_shared_bound() -> None:
    plan = scale_open_cohort(
        (_open("AAPL", 100.0, 100.0, 15000.0), _open("MSFT", 100.0, 100.0, 15000.0)),
        _FLAT_PARAMS,
    )
    assert plan.scale is not None
    assert plan.scale.scale == pytest.approx(0.75)
    assert sorted(i.qty for i in plan.intents) == [75.0, 75.0]
    assert plan.scale.members == ("AAPL", "MSFT")


def test_cohort_that_already_fits_is_not_scaled() -> None:
    plan = scale_open_cohort(
        (_open("AAPL", 10.0, 100.0, 15000.0), _open("MSFT", 10.0, 100.0, 15000.0)),
        _FLAT_PARAMS,
    )
    assert plan.scale is None and not plan.dropped


def test_scaled_open_that_floors_to_zero_is_dropped() -> None:
    plan = scale_open_cohort(
        (_open("AAPL", 100.0, 100.0, 5000.0), _open("MSFT", 1.0, 100.0, 5000.0)),
        _FLAT_PARAMS,
    )
    assert [d.intent.symbol for d in plan.dropped] == ["MSFT"]
    assert sorted(i.qty for i in plan.intents) == [49.0]


def _probe_fills(
    symbols: tuple[str, ...], qty: float, cash: float, params: ExecutionParams
):
    """The backtest's fills, priced exactly as the live probe derives them."""
    return tuple(
        execute_signal(
            intent_to_signal(_open(sym, qty, 100.0, cash), CYCLE_TS, None),
            ref_candle(100.0, sym, CYCLE_TS),
            params,
        )
        for sym in symbols
    )


def _working(order_ref: str, order_id: str, filled_qty: float = 0.0) -> WorkingOrder:
    return WorkingOrder(
        order_ref=order_ref,
        order_id=order_id,
        conid=265598,
        symbol="AAPL",
        side="BUY",
        status="Submitted",
        filled_qty=filled_qty,
    )


def test_match_working_prefers_the_stored_ref_then_the_highest_attempt() -> None:
    """D6: with two of our orders live, adoption never regresses to an older attempt.

    Without a stored ref to prefer, the HIGHEST attempt wins (``1`` here, so the
    durable attempt does not regress one -> zero); with the record's own
    ``order_ref`` still working, that exact ref wins so the ticket is built from
    the same order.
    """
    key = intent_key(SCOPE, _intent())
    prefix = ref_prefix(key)
    stale = _working(order_ref(key, 0), order_id="0")
    live_new = _working(order_ref(key, 1), order_id="1")
    by_attempt = match_working((stale, live_new), prefix)
    assert by_attempt is not None and by_attempt.order_id == "1"
    by_prefer = match_working((stale, live_new), prefix, prefer=stale.order_ref)
    assert by_prefer is not None and by_prefer.order_id == "0"


def test_ref_is_ours_and_ref_matches_key_attribute_the_bar_free_ref() -> None:
    """Ownership helpers: the scope segment and the key token, never a bar."""
    key = intent_key(SCOPE, _intent("AAPL"))
    ref = order_ref(key, 3)
    assert ref_is_ours(SCOPE, ref)
    assert ref_matches_key(SCOPE, key, ref)
    # A different key (same scope) shares the scope segment but not the token.
    other = intent_key(SCOPE, _intent("MSFT"))
    assert ref_is_ours(SCOPE, ref)
    assert not ref_matches_key(SCOPE, other, ref)
    # A foreign scope's ref is not ours even with the same token shape.
    foreign = order_ref(intent_key("other_scope", _intent("AAPL")), 0)
    assert not ref_is_ours(SCOPE, foreign)
    assert not ref_matches_key(SCOPE, key, foreign)
