"""Protective OCA bracket placement over the gateway — the Step-5 mixin.

The spec's Steps 2–5 land in the basket: a confirmed entry fill is followed
WITHIN THE SAME CYCLE by an ``STP`` + ``LMT`` pair (one OCA group, ``GTC``,
``isSingleGroup``), confirmed through the o354 challenge and RE-READ as
``Submitted`` before anything is reported resting (§4.4). REPLACE on a tighten
cancels BOTH legs, confirms their absence, then submits the new pair — never
one leg. A close of a protected lot cancels-and-confirms the pair first
(invariant 6). Anything not PROVEN resting reports ``UNPROTECTED`` (exit 3),
never "placed" (F2).

Refs are minted from the LOT's own open/fill time (the entry fill's timestamp
for a same-cycle pair, the lot's ``entry_time`` for a re-arm) — never the wall
clock or the decision bar — so a re-send of the same ``(lot, level)`` reuses the
ref while a tightened level mints a new one (F11, spec §4.2/§Design).

Mixed into ``IbkrBroker``: it borrows the broker's client, account, log,
``_working_index`` and reply machinery, so this module stays a small mixin and
the broker class does not grow past its size budget.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import pandas as pd

from src.data.ibkr.client import IbkrClient, IbkrError
from src.bt.state import ActionType
from src.exec.refs import scope_tag
from src.exec.types import OrderSide, OrderType
from src.live.adapters.ibkr.orders import (
    build_ticket,
    classify_reply,
    parse_working_order,
)
from src.live.identity import OrderOutcome, WorkingOrder
from src.live.pure import OrderResult
from src.live.protect import (
    bracket_children,
    child_ref,
    kind_of,
    resting_bracket,
)
from src.live.result import Err, Ok, Result
from src.live.types import FeedError, OrderIntent, feed_error

#: How many post-confirm re-reads prove a pair rests before we call it
#: UNPROTECTED. The o354 loop bound mirrors the broker's ``MAX_REPLIES`` (this
#: module must not import the broker — the broker imports the mixin).
_PAIR_MAX_REPLIES = 5

#: How many post-confirm re-reads prove a pair rests before we call it
#: UNPROTECTED. A just-submitted pair occasionally takes a beat to flip
#: ``Submitted`` on the working read.
_BRACKET_VERIFY_RETRIES = 3

#: The wire's "resting" status (spec §0) — the ONLY proof a level rests.
_SUBMITTED = "SUBMITTED"


@dataclass(frozen=True)
class TicketLite:
    """One submission ticket: its ref, side, and the wrapped ``orders`` body."""

    order_ref: str
    side: OrderSide
    body: dict[str, object]


class _BracketHost(Protocol):
    """The ``IbkrBroker`` surface this mixin BORROWS (never re-implements).

    A mixin cannot declare the host its methods run against, so the borrowed
    surface is stated once here (methods + the attributes the broker sets in
    ``__init__``). ``BracketProtectionMixin`` inherits it, so ``self`` inside a
    mixin method is typed as the host — no ``Any``/``cast`` escape hatch, and a
    host that drops one of these surfaces fails the type check.
    """

    _client: IbkrClient
    _scope: str
    _account: str | None
    _poll_interval_s: float
    _sleep: Callable[[float], Awaitable[None]]
    _log: Callable[[str], None]

    async def _account_id(self) -> str: ...

    async def _working_index(self) -> Result[dict[str, WorkingOrder], FeedError]: ...


class BracketProtectionMixin(_BracketHost):
    """Protective bracket placement/replace/cancel over ``IbkrBroker`` internals.

    Every method is fail-closed (spec invariant 1): no POST happens after a
    failed working-orders read, and nothing is claimed resting without a read
    showing ``Submitted``. Every failure is reported ``UNPROTECTED`` — never
    "not placed", never a silent exit 0.
    """

    async def _place_bracket(
        self,
        entry: OrderIntent,
        *,
        filled_qty: float,
        conid: int,
        fill_ts: pd.Timestamp,
    ) -> Result[tuple[OrderResult, ...], FeedError]:
        """Place and CONFIRM the OCA pair for a confirmed entry fill (§4.4).

        Submit both legs in ONE ``{"orders": [...]}`` (``isSingleGroup`` on
        each), run the o354 confirm loop, then RE-READ the working orders until
        both legs are ``Submitted``. Exhaustion ⇒ ``UNPROTECTED``. A pair that
        already rests (read by our refs) is re-reported, never re-sent. Refs are
        anchored on the lot's OWN fill time (``fill_ts``), never the clock.
        """
        children = bracket_children(
            entry, conid=conid, filled_qty=filled_qty, tif="GTC"
        )
        if not children:
            return Ok(())
        return await self._submit_children(
            entry.symbol, children, conid, anchor=fill_ts
        )

    async def _submit_children(
        self,
        symbol: str,
        children: tuple[OrderIntent, ...],
        conid: int,
        *,
        anchor: pd.Timestamp,
    ) -> Result[tuple[OrderResult, ...], FeedError]:
        """Submit + CONFIRM an already-minted OCA pair and prove it rests (§4.4).

        The shared placement core for BOTH the entry fill (``_place_bracket``
        mints the children) and a re-arm (``level_bracket_intents``' children).
        Submit both legs in ONE ``{"orders": [...]}`` (``isSingleGroup`` on
        each), run the o354 confirm loop, then RE-READ until both legs read
        ``Submitted``. Exhaustion ⇒ ``UNPROTECTED``. A pair that already rests
        (by our refs) is re-reported, never re-sent. Refs anchor on ``anchor``
        — the lot's OWN open/fill time, never the clock.
        """
        working = await self._working_index()
        if isinstance(working, Err):
            return Err(
                self._unprotected(
                    f"protect {symbol}: pre-flight working-orders read failed; "
                    f"no POST, level not proven resting"
                )
            )
        resting = resting_bracket(
            tuple(working.value.values()),
            scope_tag=scope_tag(self._scope),
            conid=conid,
        )
        if resting is not None:
            stop, take = resting
            self._log(
                f"protect {symbol}: pair already resting "
                f"({stop.order_id}/{take.order_id}); re-reported, never re-sent"
            )
            return Ok(self._resting_results(children, resting))
        tickets = self._pair_tickets(symbol, children, conid, anchor)
        submitted = await self._submit_pair_confirmed(symbol, tickets)
        if isinstance(submitted, Err):
            return Err(cast("FeedError", submitted.error))
        verify = await self._re_read_resting(
            symbol, conid, tuple(t.order_ref for t in tickets)
        )
        if verify is None:
            return Err(
                self._unprotected(
                    f"protect {symbol}: pair submitted but never re-read "
                    f"as Submitted; lot is NAKED"
                )
            )
        self._log(
            f"protect {symbol} pair resting ({verify[0].order_id}/{verify[1].order_id})"
        )
        return Ok(self._resting_results(children, verify))

    async def _replace_bracket(
        self, symbol: str, children: tuple[OrderIntent, ...], *, conid: int
    ) -> Result[tuple[OrderResult, ...], FeedError]:
        """REPLACE a tightened pair: cancel BOTH legs, confirm absent, submit new.

        The new children carry NEW refs (a tightened level mints a new ref), so
        the replacement is never swallowed by the broker's cOID dedupe (F11).
        An unconfirmable cancel means NO new pair is submitted (a live pair plus
        a replacement would be over-cover) — ``UNPROTECTED`` (invariant 6).
        """
        working = await self._working_index()
        if isinstance(working, Err):
            return Err(
                self._unprotected(
                    f"replace {symbol}: pre-flight working-orders read failed; "
                    f"no POST, level not proven resting"
                )
            )
        resting = resting_bracket(
            tuple(working.value.values()),
            scope_tag=scope_tag(self._scope),
            conid=conid,
        )
        # The children are ALREADY minted (a tightened or re-armed level's pair) —
        # submit them directly, never re-mint through ``bracket_children``: that
        # asserts an ENTRY action and would crash on a reduce child.
        anchor = _lot_anchor(children[0])
        if resting is None:
            # Nothing rests: this is an ARM, not a replace (the ratchet diff
            # already refused a widening; an unlevelled lot takes the pair).
            return await self._submit_children(symbol, children, conid, anchor=anchor)
        cancelled = await self._cancel_pair(symbol, resting)
        if isinstance(cancelled, Err):
            return Err(cast("FeedError", cancelled.error))
        # The old pair is confirmed absent; only now is the new pair submitted.
        return await self._submit_children(symbol, children, conid, anchor=anchor)

    async def _cancel_pair(
        self, symbol: str, pair: tuple[WorkingOrder, WorkingOrder]
    ) -> Result[None, FeedError]:
        """Cancel BOTH legs and CONFIRM their absence by a follow-up read (§4.3).

        A ``DELETE`` reply alone is never the confirmation (spec §0/F8): the
        confirming evidence is the working read showing both ``orderId``s gone.
        An unconfirmable cancel refuses (``UNPROTECTED``) — callers must not
        submit a replacement or close the lot on top of a live pair.
        """
        account = await self._account_id()
        for leg in pair:
            try:
                await self._client.cancel_order(account, leg.order_id)
            except IbkrError as exc:
                return Err(
                    self._unprotected(
                        f"cancel {symbol} leg {leg.order_ref} ({leg.order_id}) "
                        f"errored ({exc.kind}): {exc}; do not submit or close on "
                        "top of a live pair"
                    )
                )
        working = await self._working_index()
        if isinstance(working, Err):
            return Err(
                self._unprotected(
                    f"cancel {symbol}: absence re-read failed; whether the legs "
                    f"are gone is unknown"
                )
            )
        still = [leg for leg in pair if leg.order_id in working.value]
        if still:
            return Err(
                self._unprotected(
                    f"cancel {symbol}: legs "
                    f"{[leg.order_id for leg in still]} still on the working "
                    f"read; cannot prove the pair is gone"
                )
            )
        self._log(f"cancel {symbol} pair confirmed absent from the working read")
        return Ok(None)

    async def _verify_resting(
        self, working: Sequence[object], *, conid: int
    ) -> Result[tuple[OrderResult, ...], FeedError]:
        """Report ``UNPROTECTED`` when our pair on *conid* is not read as resting.

        The reporter half of §4.2: the working read is the ONE source of truth —
        this never places, it only attests. A pair that reads ``Submitted`` on
        both legs returns an empty tuple; anything else (missing, ``Inactive``/
        ``Pending Submit``, one leg only) reports ``UNPROTECTED``. The caller
        decides WHICH conids must be protected.
        """
        parsed = tuple(
            w for w in (parse_working_order(row) for row in working) if w is not None
        )
        resting = resting_bracket(parsed, scope_tag=scope_tag(self._scope), conid=conid)
        if resting is not None:
            return Ok(())
        return Err(
            self._unprotected(
                f"verify {conid}: no Submitted pair on the working read; the lot "
                f"is NAKED"
            )
        )

    # -- internals ----------------------------------------------------------

    def _pair_tickets(
        self,
        symbol: str,
        children: tuple[OrderIntent, ...],
        conid: int,
        fill_ts: pd.Timestamp,
    ) -> tuple[TicketLite, ...]:
        """The submission tickets, each carrying ``isSingleGroup`` and its own
        ``(lot, level)`` ref (anchored on the lot's own time).

        The wire side is the child's own REDUCE side (Step-3 contract: a
        long-lot child carries ``position_side=short`` and trades a SELL; ``order_side``
        interprets ``position_side`` as the LOT side, so it is NOT used here).
        """
        tickets: list[TicketLite] = []
        for child in children:
            side = _wire_side(child)
            ref = child_ref(scope_tag(self._scope), child, conid=conid, anchor=fill_ts)
            ticket = build_ticket(child, conid=conid, side=side, order_ref=ref)
            body = dict(ticket.body)
            body["isSingleGroup"] = True
            tickets.append(TicketLite(order_ref=ref, side=side, body=body))
        return tuple(tickets)

    async def _submit_pair_confirmed(
        self, symbol: str, tickets: tuple[TicketLite, ...]
    ) -> Result[str, FeedError]:
        """POST the pair and run the o354 loop to a confirmed ``order_id``.

        ``classify_reply`` (the shared vocabulary) drives the loop: an ordinary
        confirm-this-order reply is confirmed through the client; anything else
        (a reject prompt, an error, an unknown shape) aborts as UNPROTECTED.
        Bounded by ``_PAIR_MAX_REPLIES``.
        """
        account = await self._account_id()
        endpoint = f"iserver/account/{account}/orders"
        try:
            response: object = await self._client.post(
                endpoint, json={"orders": [t.body for t in tickets]}
            )
        except IbkrError as exc:
            return Err(
                self._unprotected(
                    f"protect {symbol}: pair submit errored ({exc.kind}): "
                    f"{exc}; level not proven resting"
                )
            )
        confirmations = 0
        while True:
            outcome = classify_reply(response)
            if outcome.kind == "success":
                return Ok(outcome.order_id)
            if outcome.kind != "confirm":
                return Err(
                    self._unprotected(
                        f"protect {symbol}: pair submit aborted "
                        f"({outcome.message}); level not proven resting"
                    )
                )
            confirmations += 1
            if confirmations > _PAIR_MAX_REPLIES:
                return Err(
                    self._unprotected(
                        f"protect {symbol}: still asking for confirmation after "
                        f"{_PAIR_MAX_REPLIES} replies; level not proven resting"
                    )
                )
            guard = await self._working_index()
            if isinstance(guard, Err):
                return Err(
                    self._unprotected(
                        f"protect {symbol}: pre-confirm working-orders read "
                        f"failed; not confirming on an unreadable broker"
                    )
                )
            try:
                response = await self._client.confirm(outcome.reply_id)
            except IbkrError as exc:
                return Err(
                    self._unprotected(
                        f"protect {symbol}: confirm {outcome.reply_id} errored "
                        f"({exc.kind}): {exc}; level not proven resting"
                    )
                )

    async def _re_read_resting(
        self, symbol: str, conid: int, refs: tuple[str, ...]
    ) -> tuple[WorkingOrder, WorkingOrder] | None:
        """Re-read the working orders until BOTH refs read ``Submitted``.

        Bounded retry (``_BRACKET_VERIFY_RETRIES``) with the broker's poll
        interval; ``None`` on exhaustion — the caller reports ``UNPROTECTED``.
        A row whose status is not ``Submitted`` never counts (a failed-confirm
        ``Inactive`` row is not protection, spec §0).
        """
        for _ in range(_BRACKET_VERIFY_RETRIES):
            index = await self._working_index()
            if isinstance(index, Err):
                return None
            rows = tuple(index.value.values())
            ours = [
                o
                for o in rows
                if o.order_ref in refs and o.status.strip().upper() == _SUBMITTED
            ]
            if len(ours) == len(refs):
                return (ours[0], ours[1])
            await self._sleep(self._poll_interval_s)
        self._log(
            f"protect {symbol}: pair not re-read as Submitted after "
            f"{_BRACKET_VERIFY_RETRIES} reads"
        )
        return None

    def _resting_results(
        self,
        children: tuple[OrderIntent, ...],
        pair: tuple[WorkingOrder, WorkingOrder],
    ) -> tuple[OrderResult, ...]:
        """The PLACED results for a resting pair, one per CHILD (kind-matched).

        ``resting_bracket`` always returns (stop, take); a lone ``sl`` child maps
        to the stop leg only, never a strict zip off-by-one.
        """
        by_kind = {kind_of(leg): leg for leg in pair}
        return tuple(
            OrderResult(
                intent=child,
                fill=None,
                ok=True,
                message=(
                    f"protect {child.symbol} {child_kind(child)} level="
                    f"{_leg_level(child):g} resting GTC cOID={leg.order_ref} "
                    f"order_id={leg.order_id}"
                ),
                position_id=children[0].position_id,
                outcome=OrderOutcome.PLACED,
                filled_qty=0.0,
            )
            for child, leg in ((c, by_kind[child_kind(c)]) for c in children)
        )

    def _unprotected(self, message: str) -> FeedError:
        """The ONE UNPROTECTED failure shape (unsafe, exit 3 — F2)."""
        return feed_error("unprotected", message)


def child_kind(child: OrderIntent) -> str:
    """The wire kind name of a protective child (``stop``/``take_profit``)."""
    return "stop" if child.order_type is OrderType.STP else "take_profit"


def _leg_level(child: OrderIntent) -> float:
    """The resting trigger/limit a protective child carries (asserted present)."""
    level = child.stop_loss if child.order_type is OrderType.STP else child.take_profit
    assert level is not None
    return level


def _wire_side(child: OrderIntent) -> OrderSide:
    """The wire BUY/SELL of a reduce child, from its own reduce side.

    A child carries the REDUCE side (``position_side`` = the side it trades): a
    long-lot child is ``short`` and SELLs, a short-lot child is ``long`` and
    BUYs. Mirrors the sim broker's ``wire_side`` rule (parity, never forked).
    """
    assert child.position_side is not None, "a protective child needs its reduce side"
    return OrderSide.BUY if child.position_side is ActionType.long else OrderSide.SELL


def _lot_anchor(child: OrderIntent) -> pd.Timestamp:
    """The child's own lot-anchored time (its ``decision_ts`` = the lot's open)."""
    return (
        child.decision_ts
        if child.decision_ts is not None
        else cast("pd.Timestamp", pd.Timestamp(0))
    )


__all__ = [
    "BracketProtectionMixin",
    "TicketLite",
    "child_kind",
    "resting_bracket",
]
