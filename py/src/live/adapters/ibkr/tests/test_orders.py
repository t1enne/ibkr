"""Table tests for the pure IBKR order mappings (plan phase 3)."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import cast

import pandas as pd
import pytest

from src.bt.exchange import execute_signal
from src.bt.portfolio.pure import _scale_opens
from src.bt.state import ActionType, ExecutionParams, PortfolioState
from src.exec.types import (
    FixedCommission,
    OrderSide,
    OrderState,
    OrderType,
    PerShareCommission,
)
from src.live.adapters.ibkr.orders import (
    OrderMappingError,
    UnknownCloseLot,
    UnsupportedOrderType,
    UnsupportedStopOrder,
    build_ticket,
    classify_reply,
    is_fully_filled,
    is_terminal,
    match_working,
    order_side,
    order_state,
    parse_working_order,
    placement_order,
    scale_open_cohort,
    status_to_fill,
    whole_quantity,
)
from src.live.broker import intent_to_signal, ref_candle
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


def test_build_ticket_mkt_body_carries_the_supplied_coid() -> None:
    key = intent_key(SCOPE, _intent())
    ref = order_ref(key, 0)
    ticket = build_ticket(_intent(), conid=265598, side=OrderSide.BUY, order_ref=ref)
    assert ticket.body == {
        "conid": 265598,
        "side": "BUY",
        "quantity": 10.0,
        "orderType": "MKT",
        "tif": "DAY",
        "cOID": ref,
    }
    assert ticket.side is OrderSide.BUY
    assert ticket.order_ref == ref


def test_build_ticket_close_uses_the_lot_side() -> None:
    intent = _intent(action=ActionType.close, position_id="55")
    side = order_side(intent, ActionType.long)
    ticket = build_ticket(
        intent, conid=1, side=side, order_ref=order_ref(intent_key(SCOPE, intent), 0)
    )
    assert ticket.body["side"] == "SELL"


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


def test_build_ticket_open_still_rounds_to_nearest() -> None:
    # Only reducing orders floor; an open keeps round-to-nearest.
    ticket = build_ticket(
        _intent(qty=1.6),
        conid=1,
        side=OrderSide.BUY,
        order_ref=order_ref(intent_key(SCOPE, _intent()), 0),
    )
    assert ticket.body["quantity"] == 2.0


def test_whole_quantity_is_the_ticket_quantity() -> None:
    assert whole_quantity(_intent(qty=1.6)) == 2  # open rounds to nearest
    assert (
        whole_quantity(_intent(qty=1.6, action=ActionType.close, position_id="x")) == 1
    )


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


def test_parse_working_order_reads_the_coid_and_order_id() -> None:
    parsed = parse_working_order(
        {
            "cOID": "momentum-1a2b3c4d-00",
            "orderId": 979320001.0,
            "conid": "265598",
            "ticker": "AAPL",
            "side": "BUY",
            "order_status": "Submitted",
            "filledQuantity": "0",
        }
    )
    assert parsed is not None
    assert parsed.order_ref == "momentum-1a2b3c4d-00"
    assert parsed.order_id == "979320001"  # canonicalised str(int(...))
    assert parsed.symbol == "AAPL"
    assert parsed.conid == 265598


@pytest.mark.parametrize("entry", [None, {}, {"orderId": 1}, {"cOID": ""}])
def test_parse_working_order_skips_unattributable_rows(entry: object) -> None:
    assert parse_working_order(entry) is None


def test_match_working_matches_on_the_exact_scope_token_prefix() -> None:
    from src.live.identity import WorkingOrder

    ours = WorkingOrder("momentum-1a2b3c4d-00", "1", 1, "AAPL", "BUY", "Submitted", 0.0)
    foreign = WorkingOrder("mom-99999999-00", "2", 1, "AAPL", "BUY", "Submitted", 0.0)
    assert match_working((foreign, ours), "momentum-1a2b3c4d-") is ours
    # Never a symbol+side match: a foreign order on the same symbol is not ours.
    assert match_working((foreign,), "momentum-1a2b3c4d-") is None


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


@pytest.mark.parametrize("payload", [None, {}])
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
        ("Submitted", False),  # a real pre-fill status is non-terminal
        ("", False),  # an unknown status is never read as terminal
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
        ("Whatever", OrderState.PENDING),  # unknown statuses fall back to PENDING
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


def test_opens_disagreeing_on_their_bound_fail_closed() -> None:
    plan = scale_open_cohort(
        (_open("AAPL", 100.0, 100.0, 15000.0), _open("MSFT", 100.0, 100.0, 9000.0)),
        _FLAT_PARAMS,
    )
    assert plan.scale is None
    assert sorted(d.intent.symbol for d in plan.dropped) == ["AAPL", "MSFT"]
    assert plan.intents == ()


def test_commission_reserve_that_does_not_fit_drops_the_opens() -> None:
    # B3: when the commission reserve alone exhausts the budget the cohort used
    # to fit, the opens are DROPPED (rejected), never sent on cash the fee needs.
    huge_fee = ExecutionParams(
        spread_bps=0.0, slippage_bps=0.0, commission_model=FixedCommission(20000.0)
    )
    plan = scale_open_cohort(
        (_open("AAPL", 10.0, 100.0, 1000.0), _open("MSFT", 10.0, 100.0, 1000.0)),
        huge_fee,
    )
    assert plan.scale is not None and plan.scale.scale == 0.0
    assert sorted(d.intent.symbol for d in plan.dropped) == ["AAPL", "MSFT"]
    assert plan.intents == ()


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


def test_live_scale_matches_the_backtest_with_commission_and_friction() -> None:
    """The edge applies the SAME shared factor ``_scale_opens`` derives, reserving
    commission and requesting at the friction-adjusted price (B3).

    The old formula used the raw reference notional and no reserve, so live could
    only ever deploy MORE than the backtest. With a real per-share commission and
    non-zero friction the live factor must EQUAL the backtest's (parity) and be
    strictly BELOW the naive ``cash / ref-notional`` scale the old code produced.
    """
    cash = 15000.0
    params = ExecutionParams(
        spread_bps=5.0, slippage_bps=2.0, commission_model=PerShareCommission(0.01)
    )
    fills = _probe_fills(("AAPL", "MSFT"), 100.0, cash, params)
    portfolio = PortfolioState(
        cash=cash, positions={}, trades=(), equity_curve=(), initial_capital=cash
    )
    scaled, record = _scale_opens(portfolio, fills, params.commission_model)
    assert record is not None

    opens = (_open("AAPL", 100.0, 100.0, cash), _open("MSFT", 100.0, 100.0, cash))
    plan = scale_open_cohort(opens, params)
    assert plan.scale is not None
    # Parity with the backtest's shared scale.
    assert plan.scale.scale == pytest.approx(record.scale)
    # Direction: the live scale is strictly below the naive no-reserve factor, so
    # live never deploys more than the backtest (the bug this fixes).
    naive = cash / sum(o.qty * o.ref_price for o in opens)
    assert plan.scale.scale < naive
    # Same factor, but live orders are whole shares: each is the backtest's scaled
    # qty FLOORED (never rounded up past the budget).
    assert [i.qty for i in plan.intents] == [
        float(math.floor(f.signal.qty)) for f in scaled
    ]


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
