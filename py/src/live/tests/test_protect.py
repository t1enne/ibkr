"""Table tests for the pure bracket layer (spec §Step 3) — no I/O.

The wire contract is the spec's (SPEC_LIVE_STOPS §0): STP trigger on ``price``,
``isSingleGroup``/``tif GTC`` on both legs, ``stop_price``/``price`` on the
working row, ``orderType`` as the SL-vs-TP discriminator. The broker is the
single source of truth for resting levels — nothing here persists a level.

These are REGRESSION tests: each pins a behaviour that must not weaken (a
ref that moves on a re-send, a sibling side flip, an over-placed pair are all
money bugs).
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pandas as pd

from src.bt.state import ActionType, PortfolioState, Position, TradeSignal
from src.exec.refs import scope_tag
from src.exec.types import OrderType
from src.live.identity import OrderOutcome, WorkingOrder
from src.live.protect import (
    ProtectionTarget,
    bracket_children,
    kind_of,
    level_bracket_intents,
    protective_ref,
    resting_bracket,
    resting_legs,
    unprotected_results,
    unprotected_targets,
)
from src.live.types import OrderIntent

TS = cast("pd.Timestamp", pd.Timestamp("2024-06-03T14:30:00Z"))
SCOPE = "sim_x_momentum"
TAG = scope_tag(SCOPE)
CONID = 265598


def _entry(
    *,
    sl: float | None = 92.0,
    tp: float | None = 108.0,
    action: ActionType = ActionType.long,
    qty: float = 10.0,
    decision_ts: pd.Timestamp | None = TS,
) -> OrderIntent:
    return OrderIntent(
        symbol="AAPL",
        action=action,
        qty=qty,
        ref_price=100.0,
        reason="open long (flat->long)",
        stop_loss=sl,
        take_profit=tp,
        decision_ts=decision_ts,
    )


# --- bracket_children -------------------------------------------------------


def test_bracket_children_carry_the_lot_side_and_the_confirmed_qty() -> None:
    children = bracket_children(_entry(), conid=CONID, filled_qty=8.0)
    assert len(children) == 2
    stop, take = children
    assert stop.order_type is OrderType.STP and take.order_type is OrderType.LMT
    assert stop.protective and take.protective
    # The child carries the LOT side (the ONE meaning of position_side, the same
    # one order_side gives it) — never an inverted reduce side; both legs agree.
    assert stop.position_side is ActionType.long
    assert take.position_side is ActionType.long
    # Qty is the CONFIRMED fill (entry planned 10, filled 8) — never the ask.
    assert stop.qty == 8.0 and take.qty == 8.0
    # The child carries its own level, so the resting trigger is on the ticket.
    assert stop.stop_loss == 92.0 and take.take_profit == 108.0


def test_bracket_children_empty_without_a_level() -> None:
    assert bracket_children(_entry(sl=None, tp=None), conid=CONID, filled_qty=8.0) == ()


def test_bracket_children_sl_only_mints_one_stop_leg() -> None:
    children = bracket_children(_entry(tp=None), conid=CONID, filled_qty=8.0)
    assert len(children) == 1
    assert children[0].order_type is OrderType.STP


def test_bracket_children_short_entry_carries_the_short_lot_side() -> None:
    stop, take = bracket_children(
        _entry(action=ActionType.short), conid=CONID, filled_qty=8.0
    )
    assert stop.position_side is ActionType.short
    assert take.position_side is ActionType.short


# --- protective_ref ---------------------------------------------------------


def test_protective_ref_distinct_per_level_stable_per_resend() -> None:
    # Same (lot, level) mints the same ref; a tightened level mints a NEW ref so
    # a REPLACE is never swallowed by the broker's cOID dedupe (F11).
    a = protective_ref(
        TAG, symbol="AAPL", conid=CONID, anchor=TS, kind="stop", level=92.0
    )
    a2 = protective_ref(
        TAG, symbol="AAPL", conid=CONID, anchor=TS, kind="stop", level=92.0
    )
    b = protective_ref(
        TAG, symbol="AAPL", conid=CONID, anchor=TS, kind="stop", level=90.0
    )
    tp = protective_ref(
        TAG, symbol="AAPL", conid=CONID, anchor=TS, kind="take_profit", level=108.0
    )
    assert a == a2  # re-send of the same level: stable
    assert a != b  # tightened stop: distinct
    assert a != tp  # distinct per kind
    assert a.startswith(f"{TAG}-")
    assert a.rsplit("-", 1)[0] == TAG  # dashless token keeps the tag exact


def test_protective_ref_is_lot_anchored_not_clock_or_bar() -> None:
    # A different lot (its own fill time) must mint a different ref; the wall
    # clock and the decision bar must NEVER enter the identity.
    other_lot = cast("pd.Timestamp", TS + pd.Timedelta(minutes=5))
    a = protective_ref(
        TAG, symbol="AAPL", conid=CONID, anchor=TS, kind="stop", level=92.0
    )
    b = protective_ref(
        TAG, symbol="AAPL", conid=CONID, anchor=other_lot, kind="stop", level=92.0
    )
    assert a != b
    # Same lot, re-sent an hour later (a re-send after a crash): still the same ref.
    resend = protective_ref(
        TAG, symbol="AAPL", conid=CONID, anchor=TS, kind="stop", level=92.0
    )
    assert a == resend


# --- resting_bracket --------------------------------------------------------


def _pop_str(kwargs: dict[str, object], key: str, default: str) -> str:
    return str(kwargs.pop(key, default))


def _pop_float(kwargs: dict[str, object], key: str, default: float) -> float:
    value = kwargs.pop(key, None)
    return float(cast("float", value)) if value is not None else default


def _working(ref: str, *args: object, **kwargs: object) -> WorkingOrder:
    k: dict[str, object] = dict(kwargs)
    return WorkingOrder(
        order_ref=str(ref),
        order_id=_pop_str(k, "order_id", "1"),
        conid=int(_pop_str(k, "conid", str(CONID))),
        symbol=_pop_str(k, "symbol", "AAPL"),
        side=_pop_str(k, "side", "SELL"),
        status=_pop_str(k, "status", "Submitted"),
        filled_qty=_pop_float(k, "filled_qty", 0.0),
        order_type=_pop_str(k, "order_type", ""),
        level=cast("float | None", k.pop("level", None)),
    )


def _real_bracket() -> tuple[WorkingOrder, WorkingOrder]:
    """The two legs of the captured ``gateway_bracket_working.json``."""
    sl = _working(
        f"{TAG}-11111111", order_id="463681097", order_type="Stop", level=92.10
    )
    tp = _working(
        f"{TAG}-22222222", order_id="463681098", order_type="Limit", level=108.0
    )
    return sl, tp


def test_resting_bracket_reads_our_pair_off_the_wire() -> None:
    pair = resting_bracket(_real_bracket(), scope_tag=TAG, conid=CONID)
    assert pair is not None
    stop, take = pair
    assert kind_of(stop) == "stop" and kind_of(take) == "take_profit"
    # The broker is the single source of truth for the resting LEVELS.
    assert stop.level == 92.10 and take.level == 108.0


def test_resting_bracket_none_when_either_leg_is_missing() -> None:
    sl, tp = _real_bracket()
    assert resting_bracket((sl,), scope_tag=TAG, conid=CONID) is None


def test_resting_bracket_ignores_a_foreign_scope_pair() -> None:
    sl, tp = _real_bracket()
    foreign = scope_tag("other_scope")
    pair = resting_bracket((sl, tp), scope_tag=foreign, conid=CONID)
    assert pair is None


def test_resting_bracket_none_when_leg_sides_disagree() -> None:
    # Two orders that share our prefix+conid but reduce DIFFERENT lots are not a
    # bracket: OCA membership is not re-readable, so side agreement is part of
    # the reconstructed identity (invariant 5).
    sl, tp = _real_bracket()
    tp = replace(tp, side="BUY")
    assert resting_bracket((sl, tp), scope_tag=TAG, conid=CONID) is None


def test_resting_bracket_none_for_an_inactive_leg() -> None:
    # The spec's §0 failed-confirm row reads ``Inactive``/``Pending Submit`` — it
    # is NOT resting, so a bracket containing it is not protection.
    sl, tp = _real_bracket()
    tp = replace(tp, status="Inactive")
    assert resting_bracket((sl, tp), scope_tag=TAG, conid=CONID) is None


# --- resting_legs (item 4: a lone leg IS protection) ------------------------


def test_resting_legs_lone_sl_is_protection() -> None:
    # REGRESSION: a stop-only strategy (vwatr_div_dsl) rests ONE leg. The pair
    # reader refuses it, but the leg-count-agnostic reader must NOT — otherwise
    # the lot reports NAKED forever and a replace re-places a duplicate.
    sl, _ = _real_bracket()
    legs = resting_legs((sl,), scope_tag=TAG, conid=CONID)
    assert [kind_of(leg) for leg in legs] == ["stop"]


def test_resting_legs_lone_take_is_protection() -> None:
    _, tp = _real_bracket()
    legs = resting_legs((tp,), scope_tag=TAG, conid=CONID)
    assert [kind_of(leg) for leg in legs] == ["take_profit"]


def test_resting_legs_reads_both_legs_of_a_pair() -> None:
    legs = resting_legs(_real_bracket(), scope_tag=TAG, conid=CONID)
    assert {kind_of(leg) for leg in legs} == {"stop", "take_profit"}


def test_resting_legs_empty_for_a_foreign_scope_or_conid() -> None:
    # Ownership is by exact prefix + conid: another scope's leg (or a leg on
    # another conid) is NEVER our protection.
    legs = _real_bracket()
    assert resting_legs(legs, scope_tag=scope_tag("other_scope"), conid=CONID) == ()
    assert resting_legs(legs, scope_tag=TAG, conid=CONID + 1) == ()


def test_resting_legs_empty_when_not_submitted() -> None:
    # The §0 failed-confirm row (``Inactive``/``Pending Submit``) is NOT resting.
    sl, tp = _real_bracket()
    inactive = (replace(sl, status="Inactive"), replace(tp, status="Pending Submit"))
    assert resting_legs(inactive, scope_tag=TAG, conid=CONID) == ()


def test_resting_legs_empty_when_sides_disagree() -> None:
    # Two legs reducing opposite directions are two strangers sharing a prefix,
    # so NOTHING is claimed resting for the conid (fail closed, invariant 5).
    sl, tp = _real_bracket()
    assert resting_legs((sl, replace(tp, side="BUY")), scope_tag=TAG, conid=CONID) == ()


def test_resting_legs_ignores_a_row_with_no_readable_kind() -> None:
    # A row whose ``orderType`` names neither a stop nor a limit is not a
    # protective leg — it must not be counted as cover.
    sl, _ = _real_bracket()
    unknown = replace(sl, order_type="TRAIL")
    assert resting_legs((unknown,), scope_tag=TAG, conid=CONID) == ()


# --- unprotected_targets / unprotected_results (item 2 backstop) ------------


def test_backstop_reports_only_the_lots_resting_no_leg() -> None:
    # REGRESSION: this is the missing caller. A lot resting our leg is proven
    # covered and is NOT reported; a lot resting nothing is naked.
    sl, _ = _real_bracket()
    covered = ProtectionTarget(
        conid=CONID, symbol="AAPL", position_id="265598", qty=8.0
    )
    naked = ProtectionTarget(conid=CONID + 1, symbol="MSFT", position_id="999", qty=3.0)
    missing = unprotected_targets((sl,), scope_tag=TAG, targets=(covered, naked))
    assert missing == (naked,)


def test_backstop_does_not_flag_a_lot_resting_a_lone_sl() -> None:
    # The strategy-agnostic requirement: the pass must be a no-op for the
    # current strategies, whose entries carry a stop and rest a lone ``sl``.
    sl, _ = _real_bracket()
    target = ProtectionTarget(conid=CONID, symbol="AAPL", position_id="265598", qty=8.0)
    assert unprotected_targets((sl,), scope_tag=TAG, targets=(target,)) == ()


def test_backstop_reports_every_target_when_nothing_rests() -> None:
    targets = (
        ProtectionTarget(conid=CONID, symbol="AAPL", position_id="265598", qty=8.0),
        ProtectionTarget(conid=CONID + 1, symbol="MSFT", position_id="999", qty=3.0),
    )
    assert unprotected_targets((), scope_tag=TAG, targets=targets) == targets


def test_unprotected_results_are_unsafe_and_never_claimed_placed() -> None:
    target = ProtectionTarget(conid=CONID, symbol="AAPL", position_id="265598", qty=8.0)
    (row,) = unprotected_results((target,), reason="no leg rests")
    assert row.ok is False
    assert row.outcome is OrderOutcome.UNPROTECTED
    assert row.error_kind == "unprotected"
    assert row.intent.position_id == "265598"
    # The report alone must make a cycle unsafe (exit 3) — see report.is_unsafe.
    from src.live.report import _UNSAFE_OUTCOMES

    assert row.outcome in _UNSAFE_OUTCOMES


# --- level_bracket_intents --------------------------------------------------


def _lot(*, position_id: str = "265598") -> Position:
    return Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=TS,
        stop_loss=None,
        take_profit=None,
        last_price=100.0,
        type=ActionType.long,
        position_id=position_id,
    )


def _update(sl: float | None, tp: float | None) -> TradeSignal:
    return TradeSignal(
        action=ActionType.stop_update,
        symbol="AAPL",
        timestamp=TS,
        price=100.0,
        qty=0.0,
        stop_loss=sl,
        take_profit=tp,
        position_id="265598",
    )


def _view(*lots: Position) -> PortfolioState:
    return PortfolioState(
        cash=100_000.0,
        positions={"AAPL": lots},
        trades=(),
        equity_curve=(),
        initial_capital=100_000.0,
    )


def test_level_intents_arm_a_newly_levelled_lot() -> None:
    intents = level_bracket_intents("cfg.json", _view(_lot()), (_update(92.0, 108.0),))
    assert len(intents) == 2
    assert all(i.protective for i in intents)
    assert {i.order_type for i in intents} == {OrderType.STP, OrderType.LMT}


def test_level_intents_empty_for_a_flat_symbol() -> None:
    assert level_bracket_intents("cfg.json", _view(), (_update(92.0, 108.0),)) == ()


def test_level_intents_refuse_a_widening_level_ratchet_never_widens() -> None:
    lot = _lot()
    lot = Position(
        symbol=lot.symbol,
        qty=lot.qty,
        entry_price=100.0,
        entry_time=TS,
        stop_loss=92.0,
        take_profit=108.0,
        last_price=100.0,
        type=ActionType.long,
        position_id=lot.position_id,
    )
    # A LONG's stop may only move UP (tighter); 90.0 is a widening -> silent.
    assert level_bracket_intents("cfg.json", _view(lot), (_update(90.0, 108.0),)) == ()


def test_level_intents_emit_a_tightening_level() -> None:
    lot = _lot()
    lot = Position(
        symbol=lot.symbol,
        qty=lot.qty,
        entry_price=100.0,
        entry_time=TS,
        stop_loss=92.0,
        take_profit=108.0,
        last_price=100.0,
        type=ActionType.long,
        position_id=lot.position_id,
    )
    intents = level_bracket_intents("cfg.json", _view(lot), (_update(95.0, 108.0),))
    assert intents
    assert all(i.stop_loss == 95.0 or i.take_profit == 108.0 for i in intents)
