"""Protective OCA bracket placement over the gateway — the Step-5 mixin.

The spec's Steps 2–5 land in the basket: a confirmed entry fill is followed
WITHIN THE SAME CYCLE by an ``STP`` + ``LMT`` pair (one OCA group, ``GTC``,
``isSingleGroup``), confirmed through the o354 challenge and RE-READ as
``Submitted`` before anything is reported resting (§4.4). REPLACE on a tighten
cancels WHAT RESTS (a full pair, or the lone leg a stop-only strategy armed),
confirms their absence, then submits the new pair — never one leg. A close of a
protected lot cancels-and-confirms first (invariant 6). Anything not PROVEN
resting reports ``UNPROTECTED`` (exit 3), never "placed" (F2).

:meth:`BracketProtectionMixin._attest_protection` is the per-cycle BACKSTOP
(§4.2): it reads the working orders once, reports every lot resting no
protective leg of ours, and never places — a bracket lost AFTER its entry cycle
still exits 3 instead of passing as clean.

Refs are minted from the LOT's own open/fill time (the entry fill's timestamp
for a same-cycle pair, the lot's ``entry_time`` for a re-arm) — never the wall
clock or the decision bar — so a re-send of the same ``(lot, level)`` reuses the
ref while a tightened level mints a new one (F11, spec §4.2/§Design).

Mixed into ``IbkrBroker``: it borrows the broker's client, account, log,
``_working_index`` and reply machinery, so this module stays a small mixin and
the broker class does not grow past its size budget.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import pandas as pd

from src.data.ibkr.client import IbkrClient, IbkrError
from src.bt.state import ActionType
from src.exec.refs import scope_tag
from src.exec.types import OrderSide, OrderType
from src.live.adapters.ibkr.orders import build_ticket, classify_reply
from src.live.identity import OrderOutcome, WorkingOrder
from src.live.pure import OrderResult
from src.live.protect import (
    ProtectionTarget,
    bracket_children,
    child_ref,
    is_resting_status,
    kind_of,
    resting_legs,
    unprotected_results,
    unprotected_targets,
)
from src.live.result import Err, Ok, Result
from src.live.types import FeedError, OrderIntent, feed_error

#: How many order-reply messages we will confirm for one pair before giving up.
#: The o354 loop bound mirrors the broker's ``MAX_REPLIES`` (this module must not
#: import the broker — the broker imports the mixin).
_PAIR_MAX_REPLIES = 5

#: How many post-confirm re-reads prove a pair rests before we call it
#: UNPROTECTED. A just-submitted pair occasionally takes a beat to flip onto
#: the working read. The resting RULE itself lives in ``src.live.protect``
#: (``is_resting_status``) so this edge and the pure backstop cannot drift.
_BRACKET_VERIFY_RETRIES = 3


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
        Submit every leg in ONE ``{"orders": [...]}`` (``isSingleGroup`` on
        each), run the o354 confirm loop, then RE-READ until every leg reads
        ``Submitted``. Exhaustion ⇒ ``UNPROTECTED``. A conid that already rests
        our legs for EVERY child kind is re-reported, never re-sent; a conid
        resting only SOME is refused (never a duplicate POST on top of a live
        leg). Refs anchor on ``anchor`` — the lot's OWN open/fill time, never
        the clock.
        """
        working = await self._working_index()
        if isinstance(working, Err):
            return Err(
                self._unprotected(
                    f"protect {symbol}: pre-flight working-orders read failed; "
                    f"no POST, level not proven resting"
                )
            )
        resting = resting_legs(
            tuple(working.value.values()),
            scope_tag=scope_tag(self._scope),
            conid=conid,
        )
        if resting:
            covered = {kind_of(leg) for leg in resting}
            if all(child_kind(child) in covered for child in children):
                self._log(
                    f"protect {symbol}: legs already resting "
                    f"({', '.join(leg.order_id for leg in resting)}); "
                    f"re-reported, never re-sent"
                )
                return Ok(self._resting_results(children, resting))
            return Err(
                self._unprotected(
                    f"protect {symbol}: {len(resting)} protective leg(s) already "
                    f"rest on conid {conid} but do not cover every leg to place; "
                    f"refusing to duplicate on top of a live leg"
                )
            )
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
            f"protect {symbol} bracket resting: "
            + ", ".join(f"{kind_of(v)}={v.order_id}" for v in verify)
        )
        return Ok(self._resting_results(children, verify))

    async def _replace_bracket(
        self, symbol: str, children: tuple[OrderIntent, ...], *, conid: int
    ) -> Result[tuple[OrderResult, ...], FeedError]:
        """REPLACE a tightened bracket: cancel WHAT RESTS, confirm absent, submit new.

        The new children carry NEW refs (a tightened level mints a new ref), so
        the replacement is never swallowed by the broker's cOID dedupe (F11).
        Whatever rests is cancelled — a full pair, or the lone leg a stop-only
        strategy armed (item 4). An unconfirmable cancel means NO new pair is
        submitted (a live pair plus a replacement would be over-cover) —
        ``UNPROTECTED`` (invariant 6).
        """
        working = await self._working_index()
        if isinstance(working, Err):
            return Err(
                self._unprotected(
                    f"replace {symbol}: pre-flight working-orders read failed; "
                    f"no POST, level not proven resting"
                )
            )
        resting = resting_legs(
            tuple(working.value.values()),
            scope_tag=scope_tag(self._scope),
            conid=conid,
        )
        # The children are ALREADY minted (a tightened or re-armed level's pair) —
        # submit them directly, never re-mint through ``bracket_children``: that
        # asserts an ENTRY action and would crash on a reduce child.
        anchor = _lot_anchor(children[0])
        if not resting:
            # Nothing rests: this is an ARM, not a replace (the ratchet diff
            # already refused a widening; an unlevelled lot takes the pair).
            return await self._submit_children(symbol, children, conid, anchor=anchor)
        cancelled = await self._cancel_legs(symbol, resting)
        if isinstance(cancelled, Err):
            return Err(cast("FeedError", cancelled.error))
        # The old legs are confirmed absent; only now is the new pair submitted.
        return await self._submit_children(symbol, children, conid, anchor=anchor)

    async def _cancel_legs(
        self, symbol: str, legs: Sequence[WorkingOrder]
    ) -> Result[None, FeedError]:
        """Cancel EVERY given leg and CONFIRM their absence by a follow-up read (§4.3).

        A ``DELETE`` reply alone is never the confirmation (spec §0/F8): the
        confirming evidence is the working read showing every ``orderId`` gone.
        Any count of legs is handled — one, or a full pair (item 4). An
        unconfirmable cancel refuses (``UNPROTECTED``) — callers must not submit
        a replacement or close the lot on top of a live leg.
        """
        assert legs, "cancel needs at least one leg"
        account = await self._account_id()
        for leg in legs:
            try:
                await self._client.cancel_order(account, leg.order_id)
            except IbkrError as exc:
                return Err(
                    self._unprotected(
                        f"cancel {symbol} leg {leg.order_ref} ({leg.order_id}) "
                        f"errored ({exc.kind}): {exc}; do not submit or close on "
                        "top of a live leg"
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
        present = _order_ids(working.value)
        still = [leg for leg in legs if leg.order_id in present]
        if still:
            return Err(
                self._unprotected(
                    f"cancel {symbol}: legs "
                    f"{[leg.order_id for leg in still]} still on the working "
                    f"read; cannot prove they are gone"
                )
            )
        self._log(
            f"cancel {symbol} {len(legs)} leg(s) confirmed absent from the working read"
        )
        return Ok(None)

    async def _attest_protection(
        self, targets: Sequence[ProtectionTarget]
    ) -> tuple[OrderResult, ...]:
        """The per-cycle re-arm BACKSTOP: report every naked lot, place nothing.

        Spec §4.2 "missing ⇒ naked, loudly" and plan D1. Reads the working orders
        ONCE and, for each target, confirms a ``Submitted`` protective leg of
        ours rests (any count — a lone ``sl`` is protection). A target resting
        none is reported ``UNPROTECTED``, so a bracket that was rejected,
        cancelled externally, or fired-partially AFTER the entry cycle still
        reaches the cycle's unsafe set (exit 3) instead of a clean exit 0 — the
        gap this pass closes.

        ATTEST ONLY: it never submits, cancels or amends. A failed read is
        reported for every target — ambiguity is never "clean" — and no POST
        follows it (invariant 1). Bounded: exactly one read.
        """
        if not targets:
            return ()
        working = await self._working_index()
        if isinstance(working, Err):
            return unprotected_results(
                targets,
                reason=(
                    "working-orders read failed; cannot prove a protective leg "
                    "rests, so nothing is claimed covered"
                ),
            )
        naked = unprotected_targets(
            tuple(working.value.values()),
            scope_tag=scope_tag(self._scope),
            targets=targets,
        )
        for target in naked:
            self._log(
                f"UNPROTECTED {target.symbol} (conid {target.conid}): no "
                f"Submitted protective leg of ours rests; the lot is NAKED"
            )
        return unprotected_results(
            naked,
            reason="no Submitted protective leg of ours rests; the lot is NAKED",
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

        The wire side is derived from the child's own LOT side (Step-3 contract:
        a child protecting a long lot carries ``position_side=long`` and trades a
        SELL). ``order_side`` reads ``position_side`` the same way, so the two
        agree by construction rather than by an accidental inversion.
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
    ) -> tuple[WorkingOrder, ...] | None:
        """Re-read the working orders until BOTH refs read as resting.

        Bounded retry (``_BRACKET_VERIFY_RETRIES``) with the broker's poll
        interval; ``None`` on exhaustion — the caller reports ``UNPROTECTED``.
        A row whose status is not resting never counts (a failed-confirm
        ``Inactive`` row is not protection, spec §0).
        """
        for _ in range(_BRACKET_VERIFY_RETRIES):
            index = await self._working_index()
            if isinstance(index, Err):
                return None
            rows = tuple(index.value.values())
            ours = [
                o for o in rows if o.order_ref in refs and is_resting_status(o.status)
            ]
            if len(ours) == len(refs):
                # One leg or many: return exactly the matched legs, never index
                # a pair position that a lone-leg bracket does not have (M1).
                return tuple(ours)
            await self._sleep(self._poll_interval_s)
        self._log(
            f"protect {symbol}: pair not re-read as resting after "
            f"{_BRACKET_VERIFY_RETRIES} reads"
        )
        return None

    def _resting_results(
        self,
        children: tuple[OrderIntent, ...],
        pair: Sequence[WorkingOrder],
    ) -> tuple[OrderResult, ...]:
        """The PLACED results for the resting legs, one per CHILD (kind-matched).

        Kind-matched, so a lone ``sl`` child maps to the stop leg only and a pair
        maps to both — never a strict zip off-by-one.
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


def _order_ids(index: Mapping[str, WorkingOrder]) -> frozenset[str]:
    """Every broker order id present in a working-orders index.

    The index is keyed by ``order_ref`` (the cOID), NOT by the broker's
    ``orderId`` — so an absence check against the mapping's KEYS would always
    pass and report an unconfirmable cancel as confirmed. Read the ids off the
    values.
    """
    return frozenset(order.order_id for order in index.values())


def _leg_level(child: OrderIntent) -> float:
    """The resting trigger/limit a protective child carries (asserted present)."""
    level = child.stop_loss if child.order_type is OrderType.STP else child.take_profit
    assert level is not None
    return level


def _wire_side(child: OrderIntent) -> OrderSide:
    """The wire BUY/SELL of a reduce child, from the child's LOT side.

    A child carries ``position_side`` = the side of the LOT it protects — the ONE
    meaning the field has (the same one ``order_side`` gives it), never an
    inverted second meaning: a long lot's child SELLs, a short lot's child BUYs.
    Mirrors the sim broker's ``wire_side`` rule (parity, never forked).
    """
    assert child.position_side is not None, "a protective child needs its lot side"
    return OrderSide.SELL if child.position_side is ActionType.long else OrderSide.BUY


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
]
