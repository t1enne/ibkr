"""Table tests for the pure IBKR order mappings (plan phase 3)."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
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
)
from src.live.adapters.ibkr.orders import (
    OrderMappingError,
    UnknownCloseLot,
    UnsupportedOrderType,
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
    validate_order,
)
from src.live.pure import intent_to_signal, ref_candle
from src.live.identity import (
    WorkingOrder,
    intent_key,
    order_ref,
    ref_is_ours,
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


def test_ref_is_pure_and_keyed_on_the_intent() -> None:
    # INV-2: the cOID is a pure function of (scope, symbol, action, position_id)
    # with NO attempt tail. It never embeds the decision bar, so a re-run on a
    # LATER bar mints the SAME ref; the working-order pre-flight adopts it and
    # IBKR's own cOID dedupe absorbs a duplicate send instead of duplicating.
    early = _intent()
    late = replace_intent(_intent(), decision_ts=CYCLE_TS)
    assert intent_key(SCOPE, early) == intent_key(SCOPE, late)
    assert order_ref(intent_key(SCOPE, early)) == order_ref(intent_key(SCOPE, late))
    # A different intent (different symbol) mints a different ref.
    assert order_ref(intent_key(SCOPE, _intent("MSFT"))) != order_ref(
        intent_key(SCOPE, early)
    )
    # The ref is the whole attribution: prefix and ref coincide (no attempt tail).
    key = intent_key(SCOPE, early)
    assert ref_prefix(key) == order_ref(key)


def test_build_ticket_close_floors_the_quantity() -> None:
    # A reducing order must never round UP: a 1.6-share close sends 1 share, not
    # 2 (2 would flip the 1.6 long into a 0.4 short — finding 2).
    intent = _intent(action=ActionType.close, position_id="55", qty=1.6)
    ref = order_ref(intent_key(SCOPE, intent))
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
            order_ref=order_ref(intent_key(SCOPE, intent)),
        )


def test_build_ticket_rejects_lmt() -> None:
    with pytest.raises(UnsupportedOrderType, match="phase 4"):
        build_ticket(
            _intent(order_type=OrderType.LMT),
            conid=1,
            side=OrderSide.BUY,
            order_ref=order_ref(intent_key(SCOPE, _intent())),
        )


@pytest.mark.parametrize(
    ("stop_loss", "take_profit"),
    [(95.0, None), (None, 110.0), (95.0, 110.0)],
)
def test_entry_ticket_carrying_a_level_is_admitted_but_never_carries_it(
    stop_loss: float | None, take_profit: float | None
) -> None:
    # Step 3 (spec): an entry carrying a level is ALLOWED — the levels rest as
    # protective CHILDREN after the entry is confirmed filled, never on the
    # entry ticket. The ticket stays MKT/DAY and body carries no ``price``.
    ticket = build_ticket(
        _intent(stop_loss=stop_loss, take_profit=take_profit),
        conid=1,
        side=OrderSide.BUY,
        order_ref=order_ref(intent_key(SCOPE, _intent())),
    )
    assert ticket.body["orderType"] == "MKT"
    assert ticket.body["tif"] == "DAY"
    assert "price" not in ticket.body


def test_protective_stp_ticket_is_gtc_and_carries_the_trigger_on_price() -> None:
    # The wire contract (spec §0): an STP trigger is ``price`` (never ``auxPrice``)
    # and the protective child rests GTC.
    child = _intent(
        action=ActionType.close,
        position_id="1",
        order_type=OrderType.STP,
        stop_loss=92.0,
    )
    child = replace_intent(child, protective=True, position_side=ActionType.long)
    ticket = build_ticket(
        child,
        conid=1,
        side=OrderSide.SELL,
        order_ref=order_ref(intent_key(SCOPE, child)),
    )
    assert ticket.body["orderType"] == "STP"
    assert ticket.body["tif"] == "GTC"
    assert ticket.body["price"] == 92.0


def test_protective_lmt_ticket_is_gtc_and_carries_the_limit_on_price() -> None:
    child = _intent(
        action=ActionType.close,
        position_id="1",
        order_type=OrderType.LMT,
        take_profit=108.0,
    )
    child = replace_intent(child, protective=True, position_side=ActionType.long)
    ticket = build_ticket(
        child,
        conid=1,
        side=OrderSide.SELL,
        order_ref=order_ref(intent_key(SCOPE, child)),
    )
    assert ticket.body["orderType"] == "LMT"
    assert ticket.body["tif"] == "GTC"
    assert ticket.body["price"] == 108.0


def test_protective_stp_requires_a_finite_positive_stop_level() -> None:
    child = _intent(action=ActionType.close, position_id="1", order_type=OrderType.STP)
    child = replace_intent(child, protective=True, position_side=ActionType.long)
    with pytest.raises(OrderMappingError, match="stop_loss"):
        validate_order(child)


def test_protective_order_requires_a_proven_lot_side() -> None:
    child = _intent(
        action=ActionType.close,
        position_id="1",
        order_type=OrderType.STP,
        stop_loss=92.0,
    )
    child = replace_intent(child, protective=True)
    with pytest.raises(OrderMappingError, match="lot side"):
        validate_order(child)


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
    ours["order_ref"] = order_ref(key)
    parsed: list[WorkingOrder] = []
    for row in (ours, *rows[1:]):
        row_parsed = parse_working_order(row)
        if row_parsed is not None:
            parsed.append(row_parsed)
    assert [p.order_ref for p in parsed] == [order_ref(key)]  # foreign rows dropped
    match = match_working(parsed, prefix)
    assert match is not None and match.order_ref == order_ref(key)


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


def _bound_portfolio(cash: float) -> PortfolioState:
    """A flat book holding exactly the cohort's shared cash bound."""
    return PortfolioState(
        cash=cash,
        positions={},
        trades=(),
        equity_curve=(),
        initial_capital=cash,
    )


def test_live_scale_matches_the_backtest_rule() -> None:
    """Parity: the live edge and ``_scale_opens`` take ONE scale decision.

    Same cohort, same cash bound, same priced fills — the live factor, the
    requested notional and the budget must equal the backtest's, so a live
    over-cash cohort can never be scaled more (or less) than the same cohort in
    a backtest.
    """
    symbols = ("AAPL", "MSFT")
    bound = 15000.0
    fills = _probe_fills(symbols, 100.0, bound, _FLAT_PARAMS)
    _, record = _scale_opens(
        _bound_portfolio(bound), fills, _FLAT_PARAMS.commission_model
    )
    plan = scale_open_cohort(
        tuple(_open(s, 100.0, 100.0, bound) for s in symbols), _FLAT_PARAMS
    )
    assert record is not None and plan.scale is not None
    assert plan.scale.scale == pytest.approx(record.scale)
    assert plan.scale.requested == pytest.approx(record.requested)
    assert plan.scale.budget == pytest.approx(record.budget)


def test_under_funded_cohort_is_unscaled_on_both_paths() -> None:
    """A budget at/below the commission reserve scales NOTHING on either side.

    The backtest leaves the cohort at plan size (each open then refuses on
    genuine cash exhaustion); the live edge must match, not rescale every qty
    to zero and drop the cohort as ``unfunded`` — an unsafe classification for
    what is really a re-mintable rejection.
    """
    params = ExecutionParams(
        spread_bps=0.0, slippage_bps=0.0, commission_model=FixedCommission(1.0)
    )
    intents = (_open("AAPL", 100.0, 100.0, 1.0), _open("MSFT", 100.0, 100.0, 1.0))
    fills = _probe_fills(("AAPL", "MSFT"), 100.0, 1.0, params)
    scaled, record = _scale_opens(_bound_portfolio(1.0), fills, params.commission_model)
    plan = scale_open_cohort(intents, params)
    assert record is None and scaled == fills
    assert plan.scale is None and plan.dropped == ()
    assert plan.intents == intents


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


def test_match_working_prefers_the_exact_ref_else_the_first_match() -> None:
    """``prefer`` (the exact pure ref) wins over a longer legacy ref sharing the prefix.

    The pure ref IS the prefix, so a key names at most one working order in a
    steady state. An order carrying a legacy attempt tail (a ref minted by the
    OLD scheme, ``prefix-attempt``) still shares the prefix, so ``prefer`` is what
    lets the caller pick the exact pure ref; without it the first match is
    returned deterministically.
    """
    key = intent_key(SCOPE, _intent())
    prefix = ref_prefix(key)
    legacy = _working(f"{prefix}-0", order_id="0")  # an older attempt-tail ref
    exact = _working(order_ref(key), order_id="1")
    by_prefer = match_working((legacy, exact), prefix, prefer=order_ref(key))
    assert by_prefer is not None and by_prefer.order_id == "1"
    plain = match_working((legacy, exact), prefix)
    assert plain is not None and plain.order_id == "0"  # first entry wins


def test_ref_is_ours_attributes_only_this_scope_s_pure_ref() -> None:
    """Ownership: the whole scope tag, never a bar, and never another scope's ref."""
    key = intent_key(SCOPE, _intent("AAPL"))
    ref = order_ref(key)
    assert ref_is_ours(SCOPE, ref)
    # A different key (same scope) shares the scope segment but not the token.
    other = intent_key(SCOPE, _intent("MSFT"))
    assert ref_is_ours(SCOPE, ref)
    assert not ref.rsplit("-", 1)[-1] == order_ref(other).rsplit("-", 1)[-1]
    # A foreign scope's ref is not ours even with the same token shape.
    foreign = order_ref(intent_key("other_scope", _intent("AAPL")))
    assert not ref_is_ours(SCOPE, foreign)
