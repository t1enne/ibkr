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
from collections.abc import Awaitable, Callable
from typing import cast

import pandas as pd

from src.bt.state import FillEvent, PortfolioState
from src.data.ibkr.client import IbkrClient, IbkrError
from src.data.ibkr.lookup import lookup
from src.exec.types import Fill
from src.live.adapters.ibkr.orders import (
    OrderMappingError,
    Ticket,
    build_ticket,
    classify_reply,
    is_fully_filled,
    is_terminal,
    order_side,
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

ConidLookup = Callable[[str], Awaitable[int]]
Sleeper = Callable[[float], Awaitable[None]]


async def _default_conid_lookup(ticker: str) -> int:
    """Resolve a ticker to its IBKR conid via the secdef search endpoint."""
    conid = (await lookup(ticker)).conid
    if not isinstance(conid, str) or not conid.strip():
        raise ValueError(f"secdef search for {ticker} returned no conid")
    return int(conid)


class IbkrBroker:
    """``LiveBroker`` over the gateway: MKT orders, reply loop, bounded fill wait.

    ``dry_run`` is the defence-in-depth flag: a broker built for a dry run places
    nothing, so a missing CLI guard cannot turn into a real order.
    """

    def __init__(
        self,
        client: IbkrClient,
        *,
        strategy_id: str,
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
        self._strategy_id = strategy_id
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
        self, intent: OrderIntent, *, seq: int = 0
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
            ticket = build_ticket(
                intent,
                conid=await self._conid(intent.symbol),
                side=side,
                strategy_id=self._strategy_id,
                cycle_ts=self._now(),
                seq=seq,
            )
        except (OrderMappingError, ValueError, IbkrError) as exc:
            kind = exc.kind if isinstance(exc, IbkrError) else "rejected"
            return Err(
                feed_error(kind, f"{intent.symbol}: {exc}", symbol=intent.symbol)
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
        - terminal non-filled status (``Cancelled``/``Inactive``) -> ``rejected``;
        - a partial fill on a dead order, or still working at the deadline ->
          ``unfilled``, carrying the qty that DID fill;
        - nothing terminal within the timeout -> ``timeout``.
        """
        deadline = self._monotonic() + self._timeout_s
        status: dict[str, object] = {}
        while True:
            fetched = await self._status(order_id)
            if isinstance(fetched, Err):
                return Err(cast("FeedError", fetched.error))
            status = fetched.value
            if is_terminal(status):
                return self._terminal_result(order_id, intent, ticket, status)
            if self._monotonic() >= deadline:
                return self._unresolved_result(order_id, intent, ticket, status)
            await self._sleep(self._poll_interval_s)

    # -- internals ---------------------------------------------------------

    async def _conid(self, symbol: str) -> int:
        """The IBKR conid for *symbol* (an edge failure is a typed ``Err``)."""
        return await self._conid_lookup(symbol)

    async def _submit(self, account: str, ticket: Ticket) -> Result[str, FeedError]:
        """POST the ticket and run the (bounded) reply-confirmation loop."""
        endpoint = f"iserver/account/{account}/orders"
        try:
            # The gateway rejects a bare order (400 "Missing orders"); it wants
            # the order(s) wrapped under an ``orders`` array.
            response: object = await self._client.post(
                endpoint, json={"orders": [ticket.body]}
            )
        except IbkrError as exc:
            return Err(feed_error(exc.kind, f"submit {ticket.order_ref}: {exc}"))
        confirmations = 0
        while True:
            outcome = classify_reply(response)
            if outcome.kind == "success":
                return Ok(outcome.order_id)
            if outcome.kind == "abort":
                return Err(
                    FeedError(
                        kind="rejected",
                        message=f"order {ticket.order_ref} not placed: {outcome.message}",
                    )
                )
            confirmations += 1
            if confirmations > MAX_REPLIES:
                return Err(
                    FeedError(
                        kind="rejected",
                        message=(
                            f"order {ticket.order_ref} still asking for "
                            f"confirmation after {MAX_REPLIES} replies; "
                            f"last message: {outcome.message}"
                        ),
                    )
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

    def _terminal_result(
        self,
        order_id: str,
        intent: OrderIntent,
        ticket: Ticket,
        status: dict[str, object],
    ) -> Result[OrderResult, FeedError]:
        """A terminal status -> a filled ``OrderResult`` or a typed failure."""
        raw = order_status_of(status)
        fill = self._filled(ticket, intent, status)
        filled_qty = fill.qty if fill is not None else 0.0
        if is_fully_filled(status, filled_qty):
            assert fill is not None  # a full fill always carries qty > 0
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
        if filled_qty > 0:
            return Err(
                FeedError(
                    kind="unfilled",
                    message=(
                        f"order {ticket.order_ref} ({order_id}) status {raw} with "
                        f"only {filled_qty:g} of {intent.qty:g} filled"
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
