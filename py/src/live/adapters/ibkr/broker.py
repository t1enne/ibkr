"""``IbkrBroker`` — the real routing edge for a live cycle (plan phase 3, MKT).

Implements ``LiveBroker`` over the Client Portal Gateway. Three things make this
adapter real rather than a wrapper:

1. **The reply-confirmation loop.** ``POST /iserver/account/{acct}/orders`` does
   not always return an ``order_id``: for a warning (an MKT order without market
   data, say) it returns an order *reply message* carrying a ``replyId`` that must
   be confirmed through ``POST /iserver/reply/{replyId}`` before the order is
   actually submitted. We auto-confirm exactly one thing — an ordinary
   confirm-this-order reply — and abort on anything else (a reject prompt, an
   ``error``, an unrecognised shape) with the broker's own verbatim text. The
   loop is bounded: an endless stream of confirmations is a broker bug, not a
   mandate to keep pressing yes.
2. **The fill wait.** A submission returns an order id, not a fill. ``wait_filled``
   polls ``/iserver/account/order/status/{orderId}`` to a terminal status with a
   bounded timeout, and maps the outcome honestly: filled → a fill; rejected →
   ``rejected``; still working (or partial) when the clock runs out → ``unfilled``
   carrying the partial qty — never silently reported as success; no terminal
   status at all → ``timeout``.
3. **No local book.** ``seed`` records the replayed book for ONE purpose —
   resolving the side of a lot a close intent targets — but never settles fills
   through ``apply_fills``. IBKR is the truth: the next cycle's replay picks the
   fills up from ``/iserver/account/trades``.

Defense in depth: an ``IbkrBroker`` constructed (or flagged) as a dry run refuses
to place anything, even if a caller forgets the CLI's guard. Phase 3 places no
cancel/modify, no resting stop, no bracket and no OCA order.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import cast

import pandas as pd

from src.bt.state import ActionType, FillEvent, PortfolioState
from src.data.ibkr.client import IbkrClient, IbkrError
from src.data.ibkr.lookup import lookup
from src.exec.refs import assign_seqs
from src.exec.types import Fill, OrderState
from src.live.adapters.ibkr.mapping import (
    canonical_order_id,
    opt_str,
    parse_positions,
)
from src.live.adapters.ibkr.orders import (
    OrderMappingError,
    Ticket,
    build_ticket,
    classify_reply,
    intent_identity,
    is_fully_filled,
    is_terminal,
    order_side,
    order_state,
    order_status_of,
    sequence,
    status_to_fill,
)
from src.live.broker import OrderResult, position_side_of, trade_signal
from src.live.result import Err, Ok, Result
from src.live.types import FeedError, OrderIntent, PortfolioView, feed_error

#: How many order-reply messages we will confirm for one ticket before giving up.
#: A real confirmation needs one round trip; more than a couple is a broker that
#: keeps asking — pressing "yes" forever is how an unintended order lands.
MAX_REPLIES = 5

#: Fill-wait defaults. A DAY MKT order either fills at once or does not; 30s of
#: polling is generous and keeps a cron cycle from hanging on a dead session.
DEFAULT_POLL_INTERVAL_S = 1.0
DEFAULT_TIMEOUT_S = 30.0

#: Absolute account net below which the book is flat (the close safety guard).
_FLAT_EPS = 1e-9

ConidLookup = Callable[[str], Awaitable[int]]
Sleeper = Callable[[float], Awaitable[None]]


async def _default_conid_lookup(ticker: str) -> int:
    """Resolve a ticker to its IBKR conid via the secdef search endpoint."""
    conid = (await lookup(ticker)).conid
    if not isinstance(conid, str) or not conid.strip():
        raise ValueError(f"secdef search for {ticker} returned no conid")
    return int(conid)


def _cycle_ts(intent: OrderIntent, fallback: pd.Timestamp) -> pd.Timestamp:
    """The cOID's timestamp anchor: the intent's decision bar, else *fallback*.

    A DETERMINISTIC anchor (the bar the order was decided on) makes a re-run on
    the same data re-mint an identical ref, so IBKR dedupes it and the
    working-order pre-flight can adopt it. A wall-clock anchor minted a fresh
    ref every cycle, so an open whose ``wait_filled`` timed out became a second
    live order on the next cycle (double exposure). ``fallback`` (the wall clock)
    is used only when the intent never carried a bar — a direct/synthetic caller.
    """
    return intent.decision_ts if intent.decision_ts is not None else fallback


class IbkrBroker:
    """``LiveBroker`` over the gateway: MKT orders, reply loop, bounded fill wait.

    ``dry_run`` is the defence-in-depth flag: a broker built for a dry run places
    nothing, so a missing CLI guard cannot turn into a real order.
    """

    def __init__(
        self,
        client: IbkrClient,
        *,
        scope: str,
        account: str | None = None,
        dry_run: bool = False,
        conid_lookup: ConidLookup | None = None,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        sleep: Sleeper | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], pd.Timestamp] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self._client = client
        self._scope = scope
        self._account = account
        self._dry_run = dry_run
        self._conid_lookup = (
            conid_lookup if conid_lookup is not None else _default_conid_lookup
        )
        self._poll_interval_s = poll_interval_s
        self._timeout_s = timeout_s
        self._sleep = sleep if sleep is not None else asyncio.sleep
        self._monotonic = monotonic
        self._now = now if now is not None else lambda: pd.Timestamp.now(tz="UTC")
        self._log = log if log is not None else (lambda _message: None)
        self._book: PortfolioView | None = None

    # -- LiveBroker --------------------------------------------------------

    def seed(self, portfolio: PortfolioState) -> None:
        """Record the replayed book — metadata only, never a settle target.

        The book is consulted for one thing: the side of the lot a close targets.
        It is NOT settled through ``apply_fills`` — IBKR is the truth, and the
        next cycle's execution replay is what turns this cycle's fills into lots.
        """
        self._book = portfolio

    async def place(
        self, intent: OrderIntent, *, seq: int | None = None
    ) -> Result[OrderResult, FeedError]:
        """Submit ONE MKT order and wait for its fill.

        Failure is a value at every step: a refused mapping (LMT, an unknown close
        lot), a transport failure, a rejected/unfilled/timed-out order. Nothing is
        retried blindly — a duplicate live order is worse than a failed cycle.
        """
        if self._dry_run:
            return Err(
                FeedError(
                    kind="auth",
                    message=(
                        f"dry-run IbkrBroker refuses to place {intent.symbol}: "
                        f"no order was submitted"
                    ),
                    symbol=intent.symbol,
                )
            )
        account = self._account or self._client.account
        if not account:
            return Err(
                FeedError(
                    kind="auth",
                    message="no account configured (set IBKR_ACCOUNT)",
                    symbol=intent.symbol,
                )
            )
        try:
            # Resolve the side FIRST: a close whose lot is not in the seeded book
            # is an error, and it must fail before any network round trip.
            side = order_side(intent, position_side_of(self._book, intent))
            conid = await self._conid(intent.symbol)
            resolved_seq = (
                seq if seq is not None else assign_seqs([intent_identity(intent)])[0]
            )
            ticket = build_ticket(
                intent,
                conid=conid,
                side=side,
                scope=self._scope,
                cycle_ts=_cycle_ts(intent, self._now()),
                seq=resolved_seq,
            )
        except (OrderMappingError, ValueError, IbkrError) as exc:
            kind = exc.kind if isinstance(exc, IbkrError) else "rejected"
            return Err(
                feed_error(kind, f"{intent.symbol}: {exc}", symbol=intent.symbol)
            )

        # Safety guard (plan §6 phase 3.5): never send a REDUCING order for a
        # conid the account is actually flat on, and never send one when the
        # account net cannot be READ. On a shared account the net is the sum of
        # all scopes plus a human's, so a zero net is a hard "nothing to
        # reduce"; an unreadable net fails CLOSED (a reducing order against an
        # unknown net is how a close flips into an open).
        if intent.action is ActionType.close:
            net = await self._account_net(conid)
            if net is None:
                return Err(
                    feed_error(
                        "rejected",
                        f"refused close {intent.symbol} (conid {conid}): account net "
                        f"unreadable, cannot prove there is anything to reduce",
                        symbol=intent.symbol,
                    )
                )
            if abs(net) <= _FLAT_EPS:
                return Err(
                    feed_error(
                        "rejected",
                        f"refused close {intent.symbol} (conid {conid}): account is "
                        f"flat, nothing to reduce",
                        symbol=intent.symbol,
                    )
                )

        if ticket.rounded:
            self._log(
                f"rounded {intent.symbol} qty {intent.qty:g} -> "
                f"{ticket.body['quantity']:g} (whole shares)"
            )

        submitted = await self._submit(account, ticket)
        if isinstance(submitted, Err):
            return Err(cast("FeedError", submitted.error))
        order_id = submitted.value
        self._log(
            f"submitted {intent.action.value} {intent.symbol} qty={intent.qty:g} "
            f"cOID={ticket.order_ref} order_id={order_id}"
        )
        return await self.wait_filled(order_id, intent, ticket)

    async def place_cohort(
        self, intents: tuple[OrderIntent, ...]
    ) -> Result[tuple[OrderResult, ...], FeedError]:
        """A deterministic loop of ``place`` — one order per intent, in order.

        Unlike the simulated broker this settles nothing atomically (IBKR does the
        settling), so a single failed order is reported as a failed ``OrderResult``
        and the loop continues: one bad symbol must not abandon the rest of the
        cycle. The cohort therefore never returns a cohort-level ``Err``.
        """
        results: list[OrderResult] = []
        for seq, intent in sequence(intents):
            placed = await self.place(intent, seq=seq)
            if isinstance(placed, Err):
                error = cast("FeedError", placed.error)
                message = f"{error.kind}: {error.message}"
                self._log(f"order failed {intent.symbol}: {message}")
                results.append(
                    OrderResult(intent=intent, fill=None, ok=False, message=message)
                )
            else:
                results.append(placed.value)
        return Ok(tuple(results))

    async def close(self) -> Result[None, FeedError]:
        """Close the HTTP client (its pool outlives nothing once the cycle ends)."""
        await self._client.aclose()
        return Ok(None)

    async def wait_filled(
        self, order_id: str, intent: OrderIntent, ticket: Ticket
    ) -> Result[OrderResult, FeedError]:
        """Poll the order to a terminal status; map it to a fill or a typed failure.

        Outcomes (all reported, none silently accepted):

        - terminal ``Filled`` for the whole size -> ``Ok`` with the fill;
        - a complete fill whose status has not yet flipped (``cum_fill`` reaches
          ``total_size``) -> ``Ok``, because a fill cannot exceed the size;
        - terminal non-filled status (``Cancelled``/``Inactive``) -> ``rejected``;
        - a partial fill on a dead order, or still working at the deadline ->
          ``unfilled``, carrying the qty that DID fill;
        - a terminal ``Filled`` whose body omits/garbles ``cum_fill`` ->
          ``unresolved`` (it filled; its size is unknown) — never "nothing filled";
        - nothing terminal within the timeout -> ``timeout``.
        """
        deadline = self._monotonic() + self._timeout_s
        status: dict[str, object] = {}
        while True:
            fetched = await self._status(order_id)
            if isinstance(fetched, Err):
                return Err(cast("FeedError", fetched.error))
            status = fetched.value
            if is_terminal(status) or is_fully_filled(
                status, self._filled_qty(ticket, intent, status)
            ):
                return self._terminal_result(order_id, intent, ticket, status)
            if self._monotonic() >= deadline:
                return self._unresolved_result(order_id, intent, ticket, status)
            await self._sleep(self._poll_interval_s)

    # -- internals ---------------------------------------------------------

    async def _conid(self, symbol: str) -> int:
        """The IBKR conid for *symbol* (an edge failure is a typed ``Err``)."""
        return await self._conid_lookup(symbol)

    async def _account_net(self, conid: int) -> float | None:
        """The account's net quantity for *conid*, or ``None`` if it cannot be read.

        The ONE read of account state in the live edge: it backs the close safety
        guard. ``None`` (a failed read) means "unknown", which must not be treated
        as flat — the order proceeds and IBKR itself refuses a bogus reduction.
        """
        account = self._account or self._client.account
        try:
            raw = await self._client.positions_all(account)
        except IbkrError:
            return None
        positions, _ = parse_positions(raw)
        return sum(p.qty for p in positions if p.conid == conid)

    async def _working_order_id(self, order_ref: str) -> str | None:
        """A working order's id for *order_ref* from ``/iserver/account/orders``.

        Read after an ambiguous submit so a working order is SEEN rather than
        re-sent (plan §6 phase 3.5 placement hygiene).
        """
        try:
            orders = await self._client.open_orders()
        except IbkrError:
            return None
        for entry in orders:
            if not isinstance(entry, Mapping):
                continue
            body = cast("Mapping[str, object]", entry)
            if opt_str(body.get("cOID")) == order_ref:
                return canonical_order_id(body.get("orderId")) or None
        return None

    async def _submit(self, account: str, ticket: Ticket) -> Result[str, FeedError]:
        """Adopt an already-working order, else POST and run the reply loop.

        Before every submit we ask the account which orders are working and ADOPT
        one carrying our cOID. Because the cOID is anchored on the decision bar,
        a still-live order from a prior cycle (its ``wait_filled`` timed out) is
        found here and never re-sent — the fix for cross-cycle double exposure.
        """
        endpoint = f"iserver/account/{account}/orders"
        working = await self._working_order_id(ticket.order_ref)
        if working is not None:
            self._log(
                f"adopting already-working order {working} for {ticket.order_ref} "
                f"via /iserver/account/orders"
            )
            return Ok(working)
        try:
            # The gateway rejects a bare order (400 "Missing orders"); it wants
            # the order(s) wrapped under an ``orders`` array.
            response: object = await self._client.post(
                endpoint, json={"orders": [ticket.body]}
            )
        except IbkrError as exc:
            # Ambiguous submit: before reporting failure, ask the account which
            # orders are working. An order carrying our cOID already exists, so
            # we adopt it rather than risk a duplicate.
            working = await self._working_order_id(ticket.order_ref)
            if working is not None:
                self._log(
                    f"submit {ticket.order_ref} errored ({exc}); found working "
                    f"order {working} via /iserver/account/orders"
                )
                return Ok(working)
            return Err(feed_error(exc.kind, f"submit {ticket.order_ref}: {exc}"))
        confirmations = 0
        while True:
            outcome = classify_reply(response)
            if outcome.kind == "success":
                return Ok(outcome.order_id)
            if outcome.kind == "abort":
                if confirmations == 0:
                    return Err(
                        FeedError(
                            kind="rejected",
                            message=(
                                f"order {ticket.order_ref} not placed: "
                                f"{outcome.message}"
                            ),
                        )
                    )
                return await self._resolve_after_confirms(
                    ticket,
                    f"aborted after {confirmations} confirmations: {outcome.message}",
                )
            confirmations += 1
            if confirmations > MAX_REPLIES:
                return await self._resolve_after_confirms(
                    ticket,
                    f"still asking for confirmation after {MAX_REPLIES} replies; "
                    f"last message: {outcome.message}",
                )
            self._log(
                f"confirming {ticket.order_ref} reply {outcome.reply_id}: "
                f"{outcome.message}"
            )
            try:
                response = await self._client.post(
                    f"iserver/reply/{outcome.reply_id}", json={"confirmed": True}
                )
            except IbkrError as exc:
                return Err(feed_error(exc.kind, f"confirm {ticket.order_ref}: {exc}"))

    async def _resolve_after_confirms(
        self, ticket: Ticket, reason: str
    ) -> Result[str, FeedError]:
        """A refusal AFTER we pressed yes: the order may be live, so never claim otherwise.

        Every confirmation we sent was a ``{"confirmed": true}`` POST, which can
        submit the order. So a refusal past the first confirmation cannot be
        reported as "not placed". Ask the account by cOID: a working order is
        ADOPTED (the caller then waits on its real status); when none is found the
        state is genuinely UNKNOWN, reported as ``unresolved`` — never a rejection
        a human might answer by placing the order a second time.
        """
        working = await self._working_order_id(ticket.order_ref)
        if working is not None:
            self._log(
                f"{reason}; adopting working order {working} for {ticket.order_ref}"
            )
            return Ok(working)
        return Err(
            FeedError(
                kind="unresolved",
                message=(
                    f"order {ticket.order_ref} {reason}; whether it was placed is "
                    f"unknown (no working order carries the cOID) — do not assume "
                    f"it was not placed"
                ),
            )
        )

    async def _status(self, order_id: str) -> Result[dict[str, object], FeedError]:
        """One ``orderStatus`` read, as a typed value."""
        try:
            body = await self._client.get(f"iserver/account/order/status/{order_id}")
        except IbkrError as exc:
            return Err(feed_error(exc.kind, f"status {order_id}: {exc}"))
        if not isinstance(body, dict):
            return Err(
                FeedError(
                    kind="transport",
                    message=f"status {order_id}: unreadable body {body!r}",
                )
            )
        return Ok(dict(body))

    def _filled(
        self, ticket: Ticket, intent: OrderIntent, status: dict[str, object]
    ) -> Fill | None:
        """The shared ``Fill`` for a status body, or ``None`` when nothing filled."""
        return status_to_fill(
            status, order_ref=ticket.order_ref, symbol=intent.symbol, side=ticket.side
        )

    def _filled_qty(
        self, ticket: Ticket, intent: OrderIntent, status: dict[str, object]
    ) -> float:
        """The readable filled quantity (0.0 when the body omits/garbles ``cum_fill``)."""
        fill = self._filled(ticket, intent, status)
        return fill.qty if fill is not None else 0.0

    def _terminal_result(
        self,
        order_id: str,
        intent: OrderIntent,
        ticket: Ticket,
        status: dict[str, object],
    ) -> Result[OrderResult, FeedError]:
        """A terminal (or completed) status -> a filled ``OrderResult`` or a typed failure."""
        raw = order_status_of(status)
        fill = self._filled(ticket, intent, status)
        if fill is None:
            return self._no_readable_fill(order_id, intent, ticket, raw, status)
        if not is_fully_filled(status, fill.qty):
            return Err(
                FeedError(
                    kind="unfilled",
                    message=(
                        f"order {ticket.order_ref} ({order_id}) status {raw} with "
                        f"only {fill.qty:g} of {intent.qty:g} filled"
                    ),
                    symbol=intent.symbol,
                )
            )
        position_id = intent.position_id if intent.position_id else order_id
        message = (
            f"{intent.action.value} {intent.symbol} qty={fill.qty:g} "
            f"@ {fill.price:.4f} cOID={ticket.order_ref} order_id={order_id}"
        )
        self._log(message)
        return Ok(
            OrderResult(
                intent=intent,
                fill=self._fill_event(intent, ticket, fill),
                ok=True,
                message=message,
                position_id=position_id,
            )
        )

    def _no_readable_fill(
        self,
        order_id: str,
        intent: OrderIntent,
        ticket: Ticket,
        raw: str,
        status: dict[str, object],
    ) -> Result[OrderResult, FeedError]:
        """A terminal body with no readable fill quantity: unknown, not "nothing filled".

        A terminal ``Filled`` with an omitted/garbled ``cum_fill`` is NOT a
        zero-fill rejection — the order DID fill; only its size is unreadable, so
        it is reported ``unresolved``. Only a terminal non-filled status
        (``Cancelled``/``Inactive``) is a genuine nothing-filled rejection.
        """
        if order_state(status) is OrderState.FILLED:
            return Err(
                FeedError(
                    kind="unresolved",
                    message=(
                        f"order {ticket.order_ref} ({order_id}) status {raw}: "
                        f"terminal Filled but cum_fill unreadable — the order filled; "
                        f"its quantity is unknown"
                    ),
                    symbol=intent.symbol,
                )
            )
        return Err(
            FeedError(
                kind="rejected",
                message=(
                    f"order {ticket.order_ref} ({order_id}) status {raw}: "
                    f"nothing filled"
                ),
                symbol=intent.symbol,
            )
        )

    def _unresolved_result(
        self,
        order_id: str,
        intent: OrderIntent,
        ticket: Ticket,
        status: dict[str, object],
    ) -> Result[OrderResult, FeedError]:
        """No terminal status inside the deadline: partial -> unfilled, else timeout."""
        raw = order_status_of(status)
        fill = self._filled(ticket, intent, status)
        filled_qty = fill.qty if fill is not None else 0.0
        if filled_qty > 0:
            return Err(
                FeedError(
                    kind="unfilled",
                    message=(
                        f"order {ticket.order_ref} ({order_id}) still {raw} after "
                        f"{self._timeout_s:g}s with {filled_qty:g} of "
                        f"{intent.qty:g} filled"
                    ),
                    symbol=intent.symbol,
                )
            )
        return Err(
            FeedError(
                kind="timeout",
                message=(
                    f"order {ticket.order_ref} ({order_id}) still {raw} after "
                    f"{self._timeout_s:g}s: no terminal status"
                ),
                symbol=intent.symbol,
            )
        )

    def _fill_event(self, intent: OrderIntent, ticket: Ticket, fill: Fill) -> FillEvent:
        """The shared ``Fill`` (exec vocabulary) -> the cycle's ``FillEvent``.

        Commission stays ``0.0`` on purpose: ``orderStatus`` does not report it,
        and the exact number comes from the trade replay (plan §7.3), not from a
        guess made at submission time.
        """
        signal = trade_signal(
            symbol=intent.symbol,
            action=intent.action,
            price=fill.price,
            qty=fill.qty,
            ts=self._now(),
            reason=intent.reason,
            position_id=intent.position_id,
            stop_loss=intent.stop_loss,
            take_profit=intent.take_profit,
            tag=intent.tag,
        )
        return FillEvent(
            signal=signal,
            filled_qty=fill.qty,
            executed_price=fill.price,
            commission=fill.commission,
            slippage=fill.slippage,
            timestamp=fill.timestamp
            if fill.timestamp is not None
            else signal.timestamp,
            spread=fill.spread,
        )


__all__ = ["DEFAULT_POLL_INTERVAL_S", "DEFAULT_TIMEOUT_S", "MAX_REPLIES", "IbkrBroker"]
