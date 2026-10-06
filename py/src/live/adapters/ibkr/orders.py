"""Pure ``OrderIntent`` ⇄ IBKR tickets (plan phase 3, MKT first).

Three pure mappings live here, all table-tested and free of I/O:

- ``build_ticket`` — an ``OrderIntent`` plus the conid we resolved for it becomes
  the ``singleOrderSubmissionRequest`` body IBKR wants (``conid``, ``side``,
  ``quantity``, ``orderType``, ``tif``, ``cOID``). The ``cOID`` is
  ``refs.order_ref`` — the SAME deterministic scheme the sim side uses — so a
  re-run of a cycle re-sends an identical ref and IBKR dedupes it.
- ``sequence`` — the deterministic ``(seq, intent)`` ordering (closes first, then
  opens in the order reconcile emitted, i.e. ``config.symbols`` order). Re-running
  the same cycle therefore mints identical refs; the next bar gets fresh ones.
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
from dataclasses import dataclass
from typing import Literal, cast

import pandas as pd

from src.bt.state import ActionType
from src.exec.refs import assign_seqs, order_ref as mint_ref
from src.exec.types import Fill, OrderSide, OrderState, OrderType
from src.live.adapters.ibkr.mapping import canonical_order_id, num, opt_str
from src.live.types import OrderIntent

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


def build_ticket(
    intent: OrderIntent,
    *,
    conid: int,
    side: OrderSide,
    scope: str,
    cycle_ts: pd.Timestamp,
    seq: int,
) -> Ticket:
    """Pure: intent + resolved conid + resolved side -> the IBKR ticket body.

    ``cOID`` is ``refs.order_ref(scope, cycle_ts, seq)`` — the deterministic
    identity that makes a re-sent cycle dedupe at IBKR. ``tif`` is always ``DAY``;
    an MKT order has no meaningful resting life in this phase. Quantities are
    whole shares (plan §7.14): a fractional ``qty`` is rounded and flagged, never
    sent as-is. A REDUCING order (``close``) FLOORS its quantity so it can never
    overshoot the position it reduces; an opening order rounds to nearest. The
    side comes from ``order_side`` (the caller resolves a close's lot side from
    the replayed book before it gets here), so this function never guesses a
    direction.
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
    ref = mint_ref(scope, cycle_ts, seq)
    return Ticket(
        order_ref=ref,
        side=side,
        rounded=abs(intent.qty - whole) > 1e-9,
        body={
            "conid": conid,
            "side": side.value,
            "quantity": float(whole),
            "orderType": intent.order_type.value,
            "tif": "DAY",
            "cOID": ref,
        },
    )


def intent_identity(intent: OrderIntent) -> str:
    """The intent's stable identity: symbol + action + targeted lot.

    ``seq`` (and therefore the ``cOID``) is derived from this, NOT the batch
    position, so a close filling and dropping out of the next cycle's batch
    cannot hand the following open the close's already-seen ref (plan §4).
    """
    return f"{intent.symbol}|{intent.action.value}|{intent.position_id or ''}"


def sequence(
    intents: Sequence[OrderIntent],
) -> tuple[tuple[int, OrderIntent], ...]:
    """``(seq, intent)`` pairs: seq keyed on intent identity, closes ordered first.

    Each intent's ``seq`` comes from its stable identity (``intent_identity``),
    so re-running a cycle whose membership changed still mints the intended refs.
    Returned in closes-first order (a deterministic placement order), which no
    longer affects any ref.
    """
    ranked = sorted(
        enumerate(intents),
        key=lambda pair: (0 if pair[1].action is ActionType.close else 1, pair[0]),
    )
    seqs = assign_seqs([intent_identity(intent) for _, intent in ranked])
    return tuple((seq, intent) for seq, (_, intent) in zip(seqs, ranked, strict=True))


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


def is_terminal(status: Mapping[str, object]) -> bool:
    """True when the order's status will not change again."""
    return order_status_of(status) in _TERMINAL_STATUSES


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
    """True when a terminal ``Filled`` order really filled its whole size.

    ``total_size`` is what the ticket asked for; a body that omits it (or zeroes
    it) cannot contradict a terminal ``Filled``, so the fill stands.
    """
    if order_state(status) is not OrderState.FILLED:
        return False
    total = num(status.get("total_size"))
    return total <= 0 or filled >= total


__all__ = [
    "OrderMappingError",
    "ReplyOutcome",
    "Ticket",
    "UnknownCloseLot",
    "UnsupportedOrderType",
    "build_ticket",
    "classify_reply",
    "intent_identity",
    "is_fully_filled",
    "is_terminal",
    "order_side",
    "order_state",
    "order_status_of",
    "sequence",
    "status_to_fill",
]
