"""Pure ``OrderIntent`` ⇄ IBKR tickets (plan phase 3, MKT first).

Three pure mappings live here, all table-tested and free of I/O:

- ``build_ticket`` — an ``OrderIntent`` plus the conid we resolved for it becomes
  the ``singleOrderSubmissionRequest`` body IBKR wants (``conid``, ``side``,
  ``quantity``, ``orderType``, ``tif``, ``cOID``). The ``cOID`` is now supplied
  by the CALLER (``src.live.identity.order_ref``) — the intent's bar-free token
  plus an attempt counter — so this function mints no identity of its own and
  the same key always maps to the same prefix regardless of the decision bar.
- ``placement_order`` — the deterministic placement order (closes first, then
  opens in the order reconcile emitted). It no longer affects any ref: identity
  is keyed on the intent, never the batch position.
- ``scale_open_cohort`` — the backtest's ONE shared cash scale applied to a live
  over-cash OPEN cohort (whole-share floor). It reserves the cohort's estimated
  commission and requests at the friction-adjusted (executed) price, mirroring
  ``src.bt.portfolio.pure._scale_opens`` exactly so the live scale never exceeds
  the backtest's.
- ``classify_reply`` — the reply-confirmation vocabulary: an order ticket's
  response is either a success carrying an ``order_id``, a reply message that must
  be *confirmed* through ``POST /iserver/reply/{replyId}``, or a refusal to abort
  on. Only an ordinary confirm-this-order reply is auto-confirmable; everything
  else (a reject prompt, a price-band prompt, an ``error``, an unknown shape)
  aborts with the broker's verbatim text.
- ``status_to_fill`` / ``order_state`` / ``is_terminal`` — ``orderStatus`` →
  the shared ``Fill`` and the terminal classifier.

**MKT only, and no resting stops.** ``build_ticket`` refuses an ``LMT`` intent
outright: an unfilled live limit order has to be carried across cycles (or
repriced), which is phase 4. It also refuses any intent that carries a
``stop_loss``/``take_profit``: this adapter places no resting/bracket order, so
honouring the intent is impossible and sending a naked order would silently drop
the strategy's risk levels. The refusal is fail-closed — the caller reports a
rejected order naming the levels rather than trading without a stop.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, cast

import pandas as pd

from src.bt.exchange import execute_signal
from src.bt.portfolio.pure import ScaleRecord, estimate_open_commission
from src.bt.state import ActionType, ExecutionParams, FillEvent
from src.exec.types import Fill, OrderSide, OrderState, OrderType
from src.live.adapters.ibkr.mapping import canonical_order_id, num, opt_str
from src.live.broker import intent_to_signal, ref_candle
from src.live.identity import WorkingOrder, attempt_of
from src.live.types import OrderIntent

#: Deterministic timestamp for the synthetic cohort-sizing probes. Only the
#: friction/commission figures depend on it (price, not time), so a constant
#: keeps ``scale_open_cohort`` pure and its scale reproducible.
_PROBE_TS = cast("pd.Timestamp", pd.Timestamp(0))

#: IBKR's ``orderStatus.order_status`` values. ``Filled``/``Cancelled``/
#: ``Inactive`` end an order's life; everything else (including an unknown string
#: the spec does not list, and ``WarnState``) keeps the poll alive so a status we
#: do not recognise can never be mistaken for a fill.
_FILLED = "Filled"
_TERMINAL_STATUSES = frozenset({_FILLED, "Cancelled", "Inactive"})

#: ``order_status`` -> the shared ``OrderState``. Absent keys are ``PENDING``.
_STATUS_TO_STATE: Mapping[str, OrderState] = {
    _FILLED: OrderState.FILLED,
    "Cancelled": OrderState.CANCELLED,
    "Inactive": OrderState.REJECTED,
}


class OrderMappingError(ValueError):
    """An intent cannot be turned into a submittable ticket (or a reply is unusable)."""


class UnsupportedOrderType(OrderMappingError):
    """A non-MKT order reached the live adapter; carry-over policy is phase 4."""


class UnsupportedStopOrder(OrderMappingError):
    """An intent carrying SL/TP reached the adapter; no resting order exists."""


class UnknownCloseLot(OrderMappingError):
    """A close whose lot is not in the seeded book — a side we cannot prove."""


@dataclass(frozen=True)
class Ticket:
    """A submission-ready order: the ``cOID`` it carries, its side, and its body."""

    order_ref: str
    side: OrderSide
    #: The resolved conid the order targets — the book's lot handle, carried so a
    #: filled ``OrderResult`` can report the lot the fill will land on.
    conid: int
    body: dict[str, object]
    #: True when the intent's fractional quantity was rounded to whole shares.
    rounded: bool = False


@dataclass(frozen=True)
class ReplyOutcome:
    """What a submission/reply response asks of us next."""

    kind: Literal["success", "confirm", "abort"]
    order_id: str = ""
    reply_id: str = ""
    message: str = ""


def order_side(
    intent: OrderIntent, position_side: ActionType | None = None
) -> OrderSide:
    """``BUY``/``SELL`` for *intent*; a close needs the targeted lot's side.

    An open's side is the intent's own action (``long`` buys, ``short`` sells). A
    close is the opposite trade of the lot it targets, and an ``OrderIntent``
    carries only the lot id — so the caller resolves the lot's side from the
    (replayed) book and passes it in. A close whose side cannot be proved raises
    ``UnknownCloseLot``: never guess a direction for a real order.
    """
    if intent.action is ActionType.long:
        return OrderSide.BUY
    if intent.action is ActionType.short:
        return OrderSide.SELL
    if position_side is ActionType.long:
        return OrderSide.SELL
    if position_side is ActionType.short:
        return OrderSide.BUY
    raise UnknownCloseLot(
        f"close {intent.symbol} lot {intent.position_id!r}: lot not found in the "
        f"seeded book, so its side (and thus the order's BUY/SELL) is unknown"
    )


def whole_quantity(intent: OrderIntent) -> int:
    """The whole-share quantity the ticket will actually carry.

    A REDUCING order (``close``) FLOORS so it can never overshoot the position
    it reduces; an opening order rounds to nearest. The ONE definition of the
    rounding rule: ``build_ticket`` and the edge's cash guard both call it, so
    the guard bounds the quantity that is really sent (never the pre-round one).
    """
    whole = (
        math.floor(intent.qty)
        if intent.action is ActionType.close
        else int(round(intent.qty))
    )
    if whole <= 0:
        raise OrderMappingError(
            f"{intent.symbol}: quantity {intent.qty!r} rounds to {whole} shares; "
            f"refusing to place a non-positive whole-share order"
        )
    return whole


def validate_order(intent: OrderIntent) -> None:
    """Raise when *intent* cannot become a ticket (type, stops, whole quantity).

    Split from ``build_ticket`` so a caller can refuse an unsupported intent
    BEFORE any network round trip (the pre-flight working-orders read), and so
    the edge's cash guard bounds a quantity that has already been proven
    positive-whole. Raising ``OrderMappingError`` keeps the refusal a value at
    the edge.
    """
    if intent.order_type is not OrderType.MKT:
        raise UnsupportedOrderType(
            f"{intent.order_type.value} order for {intent.symbol} is not supported "
            f"live yet: LMT carry-over lands in phase 4 (MKT only in phase 3)"
        )
    if intent.stop_loss is not None or intent.take_profit is not None:
        raise UnsupportedStopOrder(
            f"{intent.symbol}: intent carries stop_loss={intent.stop_loss!r} "
            f"take_profit={intent.take_profit!r} but the IBKR adapter places no "
            f"resting stop (phase 4); refusing to place a naked order that would "
            f"drop the risk levels"
        )
    whole_quantity(intent)


def build_ticket(
    intent: OrderIntent,
    *,
    conid: int,
    side: OrderSide,
    order_ref: str,
) -> Ticket:
    """Pure: intent + resolved conid + resolved side -> the IBKR ticket body.

    ``order_ref`` is the ``cOID`` the CALLER minted (``identity.order_ref``): a
    bar-free key token plus an attempt, so the ticket never embeds the decision
    bar. ``tif`` is always ``DAY``; an MKT order has no meaningful resting life
    in this phase. Quantities are whole shares: a fractional ``qty`` is rounded
    and flagged, never sent as-is (``whole_quantity``). The side comes from
    ``order_side`` (the caller resolves a close's lot side from the replayed
    book before it gets here), so this function never guesses a direction.
    """
    validate_order(intent)
    whole = whole_quantity(intent)
    return Ticket(
        order_ref=order_ref,
        side=side,
        conid=conid,
        rounded=abs(intent.qty - whole) > 1e-9,
        body={
            "conid": conid,
            "side": side.value,
            "quantity": float(whole),
            "orderType": intent.order_type.value,
            "tif": "DAY",
            "cOID": order_ref,
        },
    )


def placement_order(intents: Sequence[OrderIntent]) -> tuple[OrderIntent, ...]:
    """Deterministic placement order: closes first, then opens in input order.

    A close frees the lot/cash an open needs, so it is placed first. The order
    no longer affects any ref (identity is keyed on the intent), only the wire
    sequence.
    """
    ranked = sorted(
        enumerate(intents),
        key=lambda pair: (0 if pair[1].action is ActionType.close else 1, pair[0]),
    )
    return tuple(intent for _, intent in ranked)


def parse_working_order(entry: object) -> WorkingOrder | None:
    """Parse one open-orders entry into a ``WorkingOrder`` (``None`` if not ours).

    The gateway echoes our client order id in ``order_ref`` (measured live; a
    foreign order placed through another client or the UI carries NO ``order_ref``
    key at all). So a row without an ``order_ref`` can never be ours and is
    dropped explicitly — a scope prefix can never match a row we did not place by
    construction. Every other field is best-effort: only the ref and the order id
    drive adoption, the rest is diagnostic.
    """
    if not isinstance(entry, Mapping):
        return None
    body = cast("Mapping[str, object]", entry)
    order_ref = opt_str(body.get("order_ref")).strip()
    if not order_ref:
        return None
    return WorkingOrder(
        order_ref=order_ref,
        order_id=canonical_order_id(body.get("orderId")),
        conid=int(num(body.get("conid"))),
        symbol=opt_str(body.get("ticker") or body.get("symbol")),
        side=opt_str(body.get("side")),
        status=opt_str(body.get("order_status") or body.get("status")),
        filled_qty=num(
            body.get("filledQuantity")
            or body.get("filled_quantity")
            or body.get("cum_fill")
        ),
    )


def match_working(
    orders: Sequence[WorkingOrder], prefix: str, *, prefer: str | None = None
) -> WorkingOrder | None:
    """The one working order whose ref starts with *prefix*, chosen deterministically.

    *prefix* is ``scope_tag-token-``: an exact scope+key prefix, never a
    symbol/side match, so a shared account's foreign orders are never adopted.
    With two of our orders live for one key (a stale attempt plus a newer one),
    returning the first match could regress the durable attempt; instead the
    record's stored ``order_ref`` wins when still working, else the HIGHEST
    attempt does — so adoption never regresses to an older order's ref.
    """
    matches = [o for o in orders if o.order_ref.startswith(prefix)]
    if not matches:
        return None
    if prefer is not None:
        for order in matches:
            if order.order_ref == prefer:
                return order
    return max(matches, key=_attempt_key)


def _attempt_key(order: WorkingOrder) -> int:
    """Sort key ordering a working order by its attempt (unreadable tail lowest)."""
    attempt = attempt_of(order.order_ref)
    return attempt if attempt is not None else -1


# -- cohort cash scaling (backtest parity) -----------------------------------

#: The 4 dp floor ``src.bt.portfolio.pure._scale_opens`` applies to a scaled qty.
#: Matched here so live and backtest agree BEFORE the live-only whole-share step.
_QTY_DP = 1e4


@dataclass(frozen=True)
class OpenDrop:
    """A scaled OPEN the edge drops instead of sending a 0-share order."""

    intent: OrderIntent
    reason: str


@dataclass(frozen=True)
class CohortPlan:
    """The edge's cohort decision: the (possibly rescaled) intents, the scale
    report for the log, and the opens a scale-to-zero dropped."""

    intents: tuple[OrderIntent, ...]
    scale: ScaleRecord | None
    dropped: tuple[OpenDrop, ...]


def scale_open_cohort(
    intents: tuple[OrderIntent, ...], params: ExecutionParams
) -> CohortPlan:
    """Mimic the backtest's ONE shared cash scale on a live OPEN cohort.

    A lone open is full-size-or-reject and is NEVER scaled (the edge's per-intent
    ``_open_cash_guard`` refuses it) — the legacy path, bit-identical to before.
    Two or more opens were all sized against the ONE ``view.cash`` reconcile
    carried to the edge as their shared ``cash_bound``; if their combined
    cost exceeds that budget they compete for it and every qty is scaled by
    ``min(1, budget / requested)``, mirroring ``src.bt.portfolio.pure._scale_opens``
    exactly: ``requested`` is the friction-adjusted (executed-price) notional and
    ``budget`` is the shared cash MINUS the cohort's estimated commission under
    the config's model. Subtracting the reserve is what keeps the live scale from
    exceeding the backtest's and stops the cohort deploying cash the commission
    still needs. A budget at or below the reserve scales to zero, so the opens
    are dropped (rejected) rather than sent. Two live-only consequences of orders
    being whole shares: a scaled qty is FLOORED to whole shares (round-to-nearest
    could resize it back UP past the budget), and one that floors to 0 shares is
    DROPPED rather than sent as a 0-share order. An unprovable budget (opens
    disagreeing on their bound) fails closed — every open is dropped, none sent.
    """
    opens = tuple(i for i in intents if i.action in (ActionType.long, ActionType.short))
    if len(opens) <= 1:
        return CohortPlan(intents=intents, scale=None, dropped=())
    cash = _shared_cash_bound(opens)
    if cash is None:
        return _refuse_opens(
            intents,
            opens,
            "cohort opens disagree on their cash bound; cannot prove a shared budget",
        )
    commission, requested = _cohort_cost(opens, params)
    budget = cash - commission
    if requested <= 0.0:
        return CohortPlan(intents=intents, scale=None, dropped=())
    scale = max(0.0, min(1.0, budget / requested))
    if scale >= 1.0:
        return CohortPlan(intents=intents, scale=None, dropped=())
    scaled, dropped, members = _apply_scale(opens, scale)
    if not members:
        return CohortPlan(intents=intents, scale=None, dropped=())
    dropped_ids = {id(d.intent) for d in dropped}
    new_intents = tuple(scaled.get(i, i) for i in intents if id(i) not in dropped_ids)
    return CohortPlan(
        intents=new_intents,
        scale=ScaleRecord(
            scale=scale, requested=requested, budget=budget, members=members
        ),
        dropped=tuple(dropped),
    )


def _cohort_cost(
    opens: tuple[OrderIntent, ...], params: ExecutionParams
) -> tuple[float, float]:
    """The cohort's estimated commission and its friction-adjusted requested notional.

    Priced through the SAME ``execute_signal`` the broker fills with, so the
    figures match ``_scale_opens``' (executed price, model commission) exactly
    rather than approximating them from the reference price.
    """
    fills: tuple[FillEvent, ...] = tuple(
        execute_signal(
            intent_to_signal(intent, _PROBE_TS, None),
            ref_candle(intent.ref_price, intent.symbol, _PROBE_TS),
            params,
        )
        for intent in opens
    )
    commission = estimate_open_commission(fills, params.commission_model)
    requested = sum(max(f.signal.qty, 0.0) * f.executed_price for f in fills)
    return commission, requested


def _shared_cash_bound(opens: tuple[OrderIntent, ...]) -> float | None:
    """The one post-close cash the opens compete for, or ``None`` if unprovable."""
    bounds = {i.cash_bound for i in opens}
    if len(bounds) != 1:
        return None
    (bound,) = bounds
    if bound is None or not math.isfinite(bound):
        return None
    return bound


def _refuse_opens(
    intents: tuple[OrderIntent, ...], opens: tuple[OrderIntent, ...], reason: str
) -> CohortPlan:
    """Fail closed: drop every open (keep non-opens) when the budget is unprovable."""
    dropped_ids = {id(i) for i in opens}
    kept = tuple(i for i in intents if id(i) not in dropped_ids)
    return CohortPlan(
        intents=kept,
        scale=None,
        dropped=tuple(OpenDrop(i, f"refused open {i.symbol}: {reason}") for i in opens),
    )


def _apply_scale(
    opens: tuple[OrderIntent, ...], scale: float
) -> tuple[dict[OrderIntent, OrderIntent], list[OpenDrop], tuple[str, ...]]:
    """Scaled opens (whole shares), the drops, and the affected symbols."""
    scaled: dict[OrderIntent, OrderIntent] = {}
    dropped: list[OpenDrop] = []
    members: list[str] = []
    for intent in opens:
        reduced = math.floor(intent.qty * scale * _QTY_DP) / _QTY_DP
        whole = math.floor(reduced)
        if whole <= 0:
            dropped.append(
                OpenDrop(
                    intent,
                    f"dropped open {intent.symbol}: scaled qty {reduced:g} floors "
                    f"to 0 shares",
                )
            )
            members.append(intent.symbol)
        elif float(whole) != intent.qty:
            scaled[intent] = replace(intent, qty=float(whole))
            members.append(intent.symbol)
    return scaled, dropped, tuple(members)


# -- reply-confirmation vocabulary -------------------------------------------


def _first(payload: object) -> Mapping[str, object] | None:
    """The response object: IBKR wraps tickets in an array; a bare object is fine."""
    entry = payload[0] if isinstance(payload, list) and payload else payload
    return cast("Mapping[str, object]", entry) if isinstance(entry, Mapping) else None


def _strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, list):
        return tuple(str(v) for v in value if isinstance(v, (str, int, float)))
    return ()


def classify_reply(payload: object) -> ReplyOutcome:
    """Classify a submit/reply response into success / confirm / abort (pure).

    The refusal checks come FIRST because the array element shapes overlap:
    ``advancedOrderReject`` carries ``text``/``options``/``orderId`` and an
    ``error`` body carries ``error``. Only a plain order reply — an ``id``
    (the replyId) plus its ``message``/``messageIds`` — is auto-confirmable.
    """
    body = _first(payload)
    if body is None:
        return ReplyOutcome(kind="abort", message="empty order submission response")
    error = body.get("error")
    if error is not None:
        return ReplyOutcome(kind="abort", message=opt_str(error))
    text = body.get("text")
    if isinstance(text, str) and text:
        return ReplyOutcome(kind="abort", message=text)
    if body.get("prompt") is True or body.get("orderId") is not None:
        return ReplyOutcome(kind="abort", message=_verbatim(body))
    order_id = body.get("order_id")
    if order_id is not None:
        return ReplyOutcome(kind="success", order_id=canonical_order_id(order_id))
    reply_id = body.get("id")
    if isinstance(reply_id, str) and reply_id:
        messages = _strings(body.get("message")) + _strings(body.get("messageIds"))
        if messages:
            return ReplyOutcome(
                kind="confirm", reply_id=reply_id, message=" | ".join(messages)
            )
        return ReplyOutcome(
            kind="abort",
            message=f"reply {reply_id} carries no message text: {_verbatim(body)}",
        )
    return ReplyOutcome(kind="abort", message=_verbatim(body))


def _verbatim(body: Mapping[str, object]) -> str:
    """The broker's own message, unedited — what a human has to read to debug it."""
    return json.dumps(cast("object", body), default=str, sort_keys=True)


# -- status -> fill / terminal classification --------------------------------


def order_status_of(status: Mapping[str, object]) -> str:
    """The raw ``order_status`` string of an ``orderStatus`` body."""
    return opt_str(status.get("order_status")).strip()


def is_terminal_status(status: str) -> bool:
    """True when a raw status string is one an order never leaves."""
    return status in _TERMINAL_STATUSES


def is_terminal(status: Mapping[str, object]) -> bool:
    """True when the order's status will not change again."""
    return is_terminal_status(order_status_of(status))


def order_state(status: Mapping[str, object]) -> OrderState:
    """The shared ``OrderState`` for an ``orderStatus`` body (unknown -> PENDING)."""
    return _STATUS_TO_STATE.get(order_status_of(status), OrderState.PENDING)


def status_to_fill(
    status: Mapping[str, object],
    *,
    order_ref: str,
    symbol: str,
    side: OrderSide,
) -> Fill | None:
    """``orderStatus`` -> the shared ``Fill`` (``cum_fill`` @ ``average_price``).

    ``orderStatus`` does not echo the ``cOID`` or the symbol, so the caller (who
    submitted the order) supplies them. ``None`` when nothing has filled yet —
    a zero-quantity ``Fill`` would read as a real trade.
    """
    qty = num(status.get("cum_fill"))
    if qty <= 0:
        return None
    return Fill(
        order_ref=order_ref,
        symbol=symbol,
        side=side,
        qty=qty,
        price=num(status.get("average_price")),
    )


def is_fully_filled(status: Mapping[str, object], filled: float) -> bool:
    """True when the order filled its whole requested size.

    A readable ``total_size`` is authoritative: ``filled >= total`` is a complete
    fill REGARDLESS of the status string, because a status can lag a fill
    (``cum_fill == total_size`` while the order still reads ``Submitted``) and a
    fill can never exceed the requested size. Only when ``total_size`` is absent
    (or zeroed) do we fall back to a terminal ``Filled`` status; a body that
    omits it cannot contradict a terminal ``Filled``.
    """
    total = num(status.get("total_size"))
    if total > 0:
        return filled >= total
    return order_state(status) is OrderState.FILLED


__all__ = [
    "OrderMappingError",
    "ReplyOutcome",
    "Ticket",
    "UnknownCloseLot",
    "UnsupportedOrderType",
    "build_ticket",
    "classify_reply",
    "is_fully_filled",
    "is_terminal",
    "is_terminal_status",
    "match_working",
    "order_side",
    "order_state",
    "order_status_of",
    "parse_working_order",
    "placement_order",
    "scale_open_cohort",
    "status_to_fill",
    "validate_order",
    "whole_quantity",
]
