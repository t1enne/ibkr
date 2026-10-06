"""``IbkrBroker`` — the real routing edge for a live cycle (plan phase 3, MKT).

Implements ``LiveBroker`` over the Client Portal Gateway. Four things make this
adapter real rather than a wrapper:

1. **Identity + the pending-intent table.** A ``cOID`` is ``slug(scope)-token-
   attempt`` built by ``src.live.identity`` — a bar-free key token plus an
   attempt counter — never the decision bar. The durable ``PendingIntents``
   table (the ONLY owner of OPEN state) records every unresolved intent, so a
   reported timeout leaves a durable record and the next cycle re-checks it
   instead of re-minting a fresh duplicate.
2. **The reply-confirmation loop.** ``POST /iserver/account/{acct}/orders`` does
   not always return an ``order_id``: for a warning it returns an order *reply
   message* carrying a ``replyId`` that must be confirmed through
   ``POST /iserver/reply/{replyId}``. We auto-confirm exactly one thing — an
   ordinary confirm-this-order reply — and abort on anything else.
3. **The fill wait.** A submission returns an order id, not a fill.
   ``wait_filled`` polls ``/iserver/account/order/status/{orderId}`` honestly:
   filled → a fill; rejected → ``rejected``; still working/partial at the
   deadline → ``unfilled``/``timeout`` — never silently success.
4. **No local book.** ``seed`` records the replayed book only to resolve a
   close's lot side; IBKR is the truth and the next cycle's execution replay
   advances the book.

**Fail-closed on identity.** Before any POST — submit or reply-confirm — an
``open_orders`` read must SUCCEED. A failed read is an ``Err`` and no POST
follows: a duplicate order is unbounded exposure, while a skipped cycle is
recoverable on the next bar because the intent is still desired. Any POST that
can submit (a submit OR a confirm) is settled by a SUCCESSFUL working-orders
read; absence in a FAILED read is ``unresolved``, never "not placed".
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from typing import cast

import pandas as pd
from ib_rest_api_client.models import SecdefSearchResponseItem

from src.bt.state import ActionType, ExecutionParams, FillEvent, PortfolioState
from src.data.ibkr.client import IbkrClient, IbkrError
from src.data.ibkr.lookup import search_contracts
from src.exec.types import Fill, OrderSide, OrderState
from src.live.adapters.ibkr.mapping import parse_positions
from src.live.adapters.ibkr.orders import (
    OrderMappingError,
    ReplyOutcome,
    Ticket,
    build_ticket,
    classify_reply,
    is_fully_filled,
    is_terminal,
    match_working,
    order_side,
    order_state,
    order_status_of,
    parse_working_order,
    placement_order,
    scale_open_cohort,
    status_to_fill,
    validate_order,
    whole_quantity,
)
from src.live.broker import OrderResult, position_side_of, trade_signal
from src.live.identity import (
    OPEN_STATES,
    IntentKey,
    IntentRecord,
    IntentState,
    OrderOutcome,
    PendingIntents,
    Resolution,
    WorkingOrder,
    intent_key,
    order_ref,
    ref_prefix,
)
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

#: Slack allowed on an OPEN's notional over its decision-time ``cash_bound`` before
#: refusing: absorbs a normal reference-price gap between the decision bar and the
#: fill. A material over-deployment (an explicit-qty strategy, or a stale intent)
#: still trips it.
_OPEN_CASH_TOLERANCE = 0.02

#: A settle-with-no-terminal maps back to a durable intent state. ``rejected`` and
#: ``unfilled`` are terminal; anything else (timeout, a transport blip on the
#: status read) keeps the intent WORKING so the next cycle re-checks it.
_STATE_FOR_KIND: Mapping[str, IntentState] = {
    "rejected": IntentState.REJECTED,
    "unfilled": IntentState.UNFILLED,
}

ConidLookup = Callable[[str], Awaitable[int]]
Sleeper = Callable[[float], Awaitable[None]]


def _names_the_ticker(candidate: SecdefSearchResponseItem, ticker: str) -> bool:
    """Whether *candidate*'s underlying symbol is exactly *ticker* (case-folded)."""
    symbol = candidate.symbol
    return isinstance(symbol, str) and symbol.strip().upper() == ticker.strip().upper()


async def _default_conid_lookup(ticker: str) -> int:
    """Resolve *ticker* to its IBKR conid, VERIFYING it is the intended contract.

    The search is by ticker, so an unverified first hit can put a real order on
    the wrong instrument. Refuses (``ValueError`` -> a typed ``FeedError``) when
    no candidate names the ticker, when the candidates name more than one conid
    (ambiguous), or when the chosen contract is ``restricted``.
    """
    candidates = await search_contracts(ticker)
    matches = tuple(c for c in candidates if _names_the_ticker(c, ticker))
    if not matches:
        raise ValueError(
            f"secdef search for {ticker} returned no contract naming {ticker}"
        )
    conids = {str(c.conid) for c in matches}
    if len(conids) > 1:
        raise ValueError(
            f"ambiguous contract for {ticker}: candidate conids {sorted(conids)}"
        )
    chosen = matches[0]
    if chosen.restricted is True:
        raise ValueError(f"contract for {ticker} is restricted (not tradable)")
    conid = chosen.conid
    if not isinstance(conid, str) or not conid.strip():
        raise ValueError(f"secdef search for {ticker} returned no conid")
    return int(conid)


def _attempt_of(order_ref_str: str, existing: IntentRecord | None) -> int:
    """The attempt encoded in *order_ref_str*'s dashless hex tail (or the prior one)."""
    try:
        return int(order_ref_str.rsplit("-", 1)[-1], 16)
    except ValueError:
        return existing.attempt if existing is not None else 0


def _day_rolled(decision_ts: pd.Timestamp | None, now: pd.Timestamp) -> bool:
    """Whether *now* is a later calendar day than *decision_ts* (a DAY order expired).

    A DAY order that is no longer working after its bar's day rolled over has
    expired unfilled; before that, a missing working order is not proof.
    """
    return decision_ts is not None and now.date() > decision_ts.date()


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
        intents: PendingIntents,
        params: ExecutionParams,
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
        self._intents = intents
        self._params = params
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
        self._conid_cache: dict[str, int] = {}

    # -- LiveBroker --------------------------------------------------------

    def seed(self, portfolio: PortfolioState) -> None:
        """Record the replayed book — metadata only, never a settle target.

        The book is consulted for one thing: the side of the lot a close targets.
        It is NOT settled through ``apply_fills`` — IBKR is the truth, and the
        next cycle's execution replay is what turns this cycle's fills into lots.
        """
        self._book = portfolio

    async def resync(self) -> Result[tuple[OrderResult, ...], FeedError]:
        """Reconcile every OPEN intent record at cycle start (INV-4).

        A working order carrying an OPEN key's prefix is ADOPTED and persisted
        WORKING. Otherwise the record's known ``order_id`` is asked directly:
        terminal Filled -> FILLED, Cancelled/Inactive -> UNFILLED/REJECTED, still
        working -> WORKING, and a DAY order gone with no fill after its day rolled
        -> UNFILLED. A record with no ``order_id`` stays OPEN (an empty working
        read is not proof when the id is unknown) and a failing status read keeps
        it WORKING (retried next cycle). A failed open-orders read returns ``Err``
        and leaves every record untouched.
        """
        records = self._intents.load_open(self._scope)
        if not records:
            return Ok(())
        index = await self._working_index()
        if isinstance(index, Err):
            return Err(cast("FeedError", index.error))
        working = tuple(index.value.values())
        now = self._now()
        adopted: list[OrderResult] = []
        for record in records:
            found = match_working(working, ref_prefix(record.key))
            if found is not None:
                self._persist_working(record.key, found, record.decision_ts)
                adopted.append(_adopted_result(record, found))
                continue
            await self._resync_status(record, now)
        return Ok(tuple(adopted))

    async def place(self, intent: OrderIntent) -> Result[OrderResult, FeedError]:
        """Submit or ADOPT one MKT order and settle its outcome.

        Fail-closed at the pre-flight: an ``open_orders`` read failure is an
        ``Err`` and NO POST follows. Any POST that can submit is settled by a
        SUCCESSFUL working-orders read: absence in a FAILED read is
        ``unresolved``, never "not placed".
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
        prepared = await self._prepare_leg(intent)
        if isinstance(prepared, Err):
            return Err(cast("FeedError", prepared.error))
        side, conid = prepared.value
        key = intent_key(self._scope, intent)
        refused = await self._pre_guard(intent, conid)
        if refused is not None:
            return Err(refused)
        # INV-1: a SUCCESSFUL working-orders read is a precondition for ANY POST.
        index = await self._working_index()
        if isinstance(index, Err):
            return Err(cast("FeedError", index.error))
        decision = self._resolve(intent, key, index.value)
        if decision.kind == "adopt":
            return await self._adopt(intent, key, side, conid, index.value)
        if decision.kind == "skip":
            self._persist_unresolved(key)
            return Err(
                feed_error(
                    "unresolved", f"{intent.symbol}: {decision.reason}", intent.symbol
                )
            )
        return await self._submit_new(account, intent, key, side, conid)

    async def place_cohort(
        self, intents: tuple[OrderIntent, ...]
    ) -> Result[tuple[OrderResult, ...], FeedError]:
        """A deterministic loop of ``place`` — closes first, then opens.

        The cohort's opens are scaled by ONE shared cash factor (``scale_open_cohort``,
        the backtest's rule including its commission reserve at the edge). Opens are
        funded by prospective cash from this cycle's closes, so if ANY close leg
        fails, every open is DROPPED and reported: the shared ``cash_bound`` the
        sizer used assumed those closes settled, and without a live available-funds
        read the only provable-safe rule is to refuse the opens the failed close
        was funding. One bad symbol never abandons the rest of the cycle.
        """
        if not intents:
            return Ok(())
        plan = scale_open_cohort(intents, self._params)
        if plan.scale is not None:
            report = plan.scale
            self._log(
                f"cohort scaled x{report.scale:.4f}: requested "
                f"{report.requested:.2f} > budget {report.budget:.2f}; reduced "
                f"{', '.join(report.members)}"
            )
        scaled = {intent_key(self._scope, i): i for i in plan.intents}
        dropped = {intent_key(self._scope, d.intent): d for d in plan.dropped}
        results: list[OrderResult] = []
        close_failed = False
        for intent in placement_order(intents):
            ident = intent_key(self._scope, intent)
            drop = dropped.get(ident)
            if drop is not None:
                self._log(drop.reason)
                results.append(_failed(intent, drop.reason))
                continue
            if intent.action is not ActionType.close and close_failed:
                message = (
                    f"dropped open {intent.symbol}: its funding close leg did not "
                    f"fill; refusing to deploy prospective cash"
                )
                self._log(message)
                results.append(_failed(intent, message))
                continue
            placed = await self.place(scaled.get(ident, intent))
            if isinstance(placed, Err):
                error = cast("FeedError", placed.error)
                message = f"{error.kind}: {error.message}"
                self._log(f"order failed {intent.symbol}: {message}")
                results.append(_failed(intent, message))
            else:
                results.append(placed.value)
            if intent.action is ActionType.close and not results[-1].ok:
                close_failed = True
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

    # -- internals: pre-flight --------------------------------------------

    async def _prepare_leg(
        self, intent: OrderIntent
    ) -> Result[tuple[OrderSide, int], FeedError]:
        """Validate the intent and resolve its side + conid, before any POST.

        A close whose lot is not in the seeded book, an unsupported order type or
        a stop-carrying intent all fail here — an ``Err`` with no network round
        trip and no ticket.
        """
        try:
            validate_order(intent)
        except OrderMappingError as exc:
            return Err(feed_error("rejected", f"{intent.symbol}: {exc}", intent.symbol))
        try:
            # Resolve the side FIRST: a close whose lot is not in the seeded book
            # is an error, and it must fail before any network round trip.
            side = order_side(intent, position_side_of(self._book, intent))
            conid = await self._conid(intent.symbol)
        except (OrderMappingError, ValueError, IbkrError) as exc:
            kind = exc.kind if isinstance(exc, IbkrError) else "rejected"
            return Err(
                feed_error(kind, f"{intent.symbol}: {exc}", symbol=intent.symbol)
            )
        return Ok((side, conid))

    async def _conid(self, symbol: str) -> int:
        """The IBKR conid for *symbol*, resolved once and cached for the run."""
        cached = self._conid_cache.get(symbol)
        if cached is not None:
            return cached
        conid = await self._conid_lookup(symbol)
        self._conid_cache[symbol] = conid
        return conid

    async def _working_index(self) -> Result[dict[str, WorkingOrder], FeedError]:
        """Every working order keyed by ``cOID``, or ``Err`` when the read fails.

        A failed read is ALWAYS an ``Err`` — never ``None`` or an empty index —
        so the caller fails closed (no POST) rather than mistaking "unreadable"
        for "nothing working" (the fail-open bug this replaces).
        """
        try:
            raw = await self._client.open_orders()
        except IbkrError as exc:
            return Err(feed_error(exc.kind, f"open_orders: {exc}"))
        index: dict[str, WorkingOrder] = {}
        for entry in raw:
            parsed = parse_working_order(entry)
            if parsed is not None:
                index.setdefault(parsed.order_ref, parsed)
        return Ok(index)

    async def _pre_guard(self, intent: OrderIntent, conid: int) -> FeedError | None:
        """The pre-submit safety guards, run before any working-orders read.

        A REDUCING order is refused against a flat or unreadable account net; an
        OPEN is bounded by the decision-time ``cash_bound`` applied to the
        WHOLE-share quantity the ticket will actually carry (never the pre-round
        request, which could round up past the bound).
        """
        if intent.action is ActionType.close:
            net = await self._account_net(conid)
            if net is None:
                return feed_error(
                    "rejected",
                    f"refused close {intent.symbol} (conid {conid}): account net "
                    f"unreadable, cannot prove there is anything to reduce",
                    intent.symbol,
                )
            if abs(net) <= _FLAT_EPS:
                return feed_error(
                    "rejected",
                    f"refused close {intent.symbol} (conid {conid}): account is "
                    f"flat, nothing to reduce",
                    intent.symbol,
                )
            return None
        try:
            whole = whole_quantity(intent)
        except OrderMappingError as exc:
            return feed_error("rejected", f"{intent.symbol}: {exc}", intent.symbol)
        return self._open_cash_guard(intent, whole)

    def _open_cash_guard(self, intent: OrderIntent, whole: int) -> FeedError | None:
        """Refuse an open whose TICKET notional exceeds its funded cash bound.

        The bound is ``intent.cash_bound`` — the EXACT cash reconcile passed to
        the sizer — re-checked against ``whole * ref_price``, the notional of the
        order actually sent. An intent that cannot state a bound, or a non-finite
        / non-positive reference price, is UNDETERMINABLE and fails closed.
        """
        bound = intent.cash_bound
        if bound is None or not math.isfinite(bound):
            return feed_error(
                "rejected",
                f"refused open {intent.symbol}: no decision-time cash bound, cannot "
                f"prove the scope funds the notional",
                intent.symbol,
            )
        price = intent.ref_price
        if not math.isfinite(price) or price <= 0:
            return feed_error(
                "rejected",
                f"refused open {intent.symbol}: no usable reference price "
                f"({price!r}) to bound the notional against the cash",
                intent.symbol,
            )
        notional = whole * price
        if notional > bound * (1.0 + _OPEN_CASH_TOLERANCE):
            return feed_error(
                "rejected",
                f"refused open {intent.symbol}: notional {notional:.2f} exceeds "
                f"funded cash {bound:.2f} by more than {_OPEN_CASH_TOLERANCE:.0%}",
                intent.symbol,
            )
        return None

    def _resolve(
        self, intent: OrderIntent, key: IntentKey, index: Mapping[str, WorkingOrder]
    ) -> Resolution:
        """Decide adopt / submit / skip for *key* given the working-orders index.

        A working order carrying the key's exact prefix is ADOPTED. With none, a
        record in a terminal state submits a fresh attempt; a record that is OPEN
        with a known ``order_id`` is SKIPPED (no re-mint — a vanished working
        order is not proof it is gone); an OPEN record with no id (never
        confirmed placed) submits, because a clean read showing nothing is proof
        enough for an order we have no id for.
        """
        found = match_working(tuple(index.values()), ref_prefix(key))
        if found is not None:
            return Resolution(
                "adopt",
                found.order_id,
                f"working order {found.order_id} carries {found.order_ref}",
            )
        existing = self._intents.load(key)
        if existing is None or existing.state not in OPEN_STATES:
            return Resolution("submit", None, "no open intent for this key")
        if existing.order_id is None:
            return Resolution(
                "submit", None, "open intent has no known order id and no working order"
            )
        return Resolution(
            "skip",
            existing.order_id,
            f"open intent {existing.state.value} holds order {existing.order_id} "
            f"but no working order carries its prefix; not re-minting",
        )

    # -- internals: submit paths ------------------------------------------

    async def _adopt(
        self,
        intent: OrderIntent,
        key: IntentKey,
        side: OrderSide,
        conid: int,
        index: Mapping[str, WorkingOrder],
    ) -> Result[OrderResult, FeedError]:
        """Persist WORKING for a found working order and wait on its real status."""
        found = match_working(tuple(index.values()), ref_prefix(key))
        if found is None:
            self._persist_unresolved(key)
            return Err(
                feed_error(
                    "unresolved",
                    f"{intent.symbol}: pre-flight adopted an order that is no longer "
                    f"working; whether it is live is unknown",
                    intent.symbol,
                )
            )
        self._persist_working(key, found, intent.decision_ts)
        self._log(
            f"adopting already-working order {found.order_id} for {found.order_ref}"
        )
        ticket = build_ticket(intent, conid=conid, side=side, order_ref=found.order_ref)
        result = await self.wait_filled(found.order_id, intent, ticket)
        return self._settle_wait(key, result, found.order_id, adopted=True)

    async def _submit_new(
        self,
        account: str,
        intent: OrderIntent,
        key: IntentKey,
        side: OrderSide,
        conid: int,
    ) -> Result[OrderResult, FeedError]:
        """Mint a fresh attempt, build the ticket, submit, and settle the wait."""
        record = self._intents.open_attempt(key, intent.decision_ts, self._now())
        try:
            ticket = build_ticket(
                intent, conid=conid, side=side, order_ref=order_ref(key, record.attempt)
            )
        except OrderMappingError as exc:
            self._intents.close(key, IntentState.REJECTED, None, self._now())
            return Err(feed_error("rejected", f"{intent.symbol}: {exc}", intent.symbol))
        if ticket.rounded:
            self._log(
                f"rounded {intent.symbol} qty {intent.qty:g} -> "
                f"{ticket.body['quantity']:g} (whole shares)"
            )
        submitted = await self._submit(
            account, ticket, key, record.attempt, intent.decision_ts
        )
        if isinstance(submitted, Err):
            return Err(cast("FeedError", submitted.error))
        order_id = submitted.value
        self._log(
            f"submitted {intent.action.value} {intent.symbol} qty={intent.qty:g} "
            f"cOID={ticket.order_ref} order_id={order_id}"
        )
        result = await self.wait_filled(order_id, intent, ticket)
        return self._settle_wait(key, result, order_id, adopted=False)

    async def _submit(
        self,
        account: str,
        ticket: Ticket,
        key: IntentKey,
        attempt: int,
        decision_ts: pd.Timestamp | None,
    ) -> Result[str, FeedError]:
        """POST the ticket and run the reply loop; an ambiguous submit is settled.

        The pre-flight already read the working orders (INV-1), so this POST is
        only reached with a successful read behind it.
        """
        endpoint = f"iserver/account/{account}/orders"
        try:
            # The gateway rejects a bare order (400 "Missing orders"); it wants
            # the order(s) wrapped under an ``orders`` array.
            response: object = await self._client.post(
                endpoint, json={"orders": [ticket.body]}
            )
        except IbkrError as exc:
            return await self._settle_ambiguous(
                key, f"submit {ticket.order_ref} errored ({exc.kind}): {exc}"
            )
        return await self._reply_loop(response, ticket, key, attempt, decision_ts)

    async def _reply_loop(
        self,
        response: object,
        ticket: Ticket,
        key: IntentKey,
        attempt: int,
        decision_ts: pd.Timestamp | None,
    ) -> Result[str, FeedError]:
        """Classify the submission response, confirming ordinary replies only."""
        confirmations = 0
        while True:
            outcome = classify_reply(response)
            if outcome.kind != "confirm":
                return await self._settle_reply(
                    outcome, ticket, key, attempt, decision_ts, confirmations
                )
            confirmations += 1
            if confirmations > MAX_REPLIES:
                return await self._settle_ambiguous(
                    key,
                    f"still asking for confirmation after {MAX_REPLIES} replies; "
                    f"last message: {outcome.message}",
                )
            # INV-1 before the confirm POST: a SUCCESSFUL read must show no
            # working order for our key (adopt one if it appears; never POST on a
            # failed read).
            guard = await self._working_index()
            if isinstance(guard, Err):
                self._persist_unresolved(key)
                return self._ambiguous_error(
                    f"confirm {ticket.order_ref}: pre-confirm working-orders read failed"
                )
            already = match_working(tuple(guard.value.values()), ref_prefix(key))
            if already is not None:
                self._persist_working(key, already, decision_ts)
                self._log(
                    f"{ticket.order_ref} already working as {already.order_id}; "
                    f"adopting instead of confirming again"
                )
                return Ok(already.order_id)
            self._log(
                f"confirming {ticket.order_ref} reply {outcome.reply_id}: "
                f"{outcome.message}"
            )
            try:
                response = await self._client.post(
                    f"iserver/reply/{outcome.reply_id}", json={"confirmed": True}
                )
            except IbkrError as exc:
                return await self._settle_ambiguous(
                    key, f"confirm {ticket.order_ref} errored ({exc.kind}): {exc}"
                )

    async def _settle_reply(
        self,
        outcome: ReplyOutcome,
        ticket: Ticket,
        key: IntentKey,
        attempt: int,
        decision_ts: pd.Timestamp | None,
        confirmations: int,
    ) -> Result[str, FeedError]:
        """A terminal submission reply: success persists WORKING; a plain abort rejects.

        A refusal BEFORE any confirmation cannot have been submitted, so it is a
        genuine rejection. A refusal AFTER a confirmation goes through
        ``_settle_ambiguous`` — the confirm may have submitted the order.
        """
        if outcome.kind == "success":
            self._intents.save(
                IntentRecord(
                    key=key,
                    state=IntentState.WORKING,
                    attempt=attempt,
                    order_ref=ticket.order_ref,
                    order_id=outcome.order_id,
                    decision_ts=decision_ts,
                )
            )
            return Ok(outcome.order_id)
        if confirmations == 0:
            self._intents.close(key, IntentState.REJECTED, None, self._now())
            return Err(
                FeedError(
                    kind="rejected",
                    message=f"order {ticket.order_ref} not placed: {outcome.message}",
                )
            )
        return await self._settle_ambiguous(
            key, f"aborted after {confirmations} confirmations: {outcome.message}"
        )

    async def _settle_ambiguous(
        self, key: IntentKey, reason: str
    ) -> Result[str, FeedError]:
        """Settle an order that MAY have been submitted: adopt, or mark UNRESOLVED.

        Shared by the ambiguous-submit, ambiguous-confirm and reply-abort/overflow
        paths. Every one of them sent a POST that can submit the order, so absence
        from a SUCCESSFUL read means the state is UNKNOWN — persisted UNRESOLVED
        and reported as such, never as "not placed". A failed read is likewise
        UNRESOLVED, never a rejection.
        """
        index = await self._working_index()
        if isinstance(index, Err):
            self._persist_unresolved(key)
            return self._ambiguous_error(reason)
        found = match_working(tuple(index.value.values()), ref_prefix(key))
        if found is not None:
            self._persist_working(key, found, None)
            self._log(f"{reason}; adopting working order {found.order_id}")
            return Ok(found.order_id)
        self._persist_unresolved(key)
        return self._ambiguous_error(reason)

    def _ambiguous_error(self, reason: str) -> Err[str, FeedError]:
        """The unresolved failure for an order whose submission left state unknown."""
        return Err(
            FeedError(
                kind="unresolved",
                message=(
                    f"{reason}; whether it was placed is unknown (no working order "
                    f"carries the key prefix) — do not assume it was not placed"
                ),
            )
        )

    def _settle_wait(
        self,
        key: IntentKey,
        result: Result[OrderResult, FeedError],
        order_id: str | None,
        *,
        adopted: bool,
    ) -> Result[OrderResult, FeedError]:
        """Persist the durable state a wait settled to, and pass the result through."""
        now = self._now()
        if isinstance(result, Err):
            error = cast("FeedError", result.error)
            state = _STATE_FOR_KIND.get(error.kind, IntentState.WORKING)
            self._intents.close(key, state, order_id, now)
            return Err(error)
        self._intents.close(key, IntentState.FILLED, order_id, now)
        order = result.value
        return Ok(replace(order, outcome=OrderOutcome.ADOPTED) if adopted else order)

    # -- internals: durable intent state ----------------------------------

    def _persist_working(
        self,
        key: IntentKey,
        working: WorkingOrder,
        decision_ts: pd.Timestamp | None,
    ) -> None:
        """Upsert a WORKING record for a found/adopted working order."""
        existing = self._intents.load(key)
        self._intents.save(
            IntentRecord(
                key=key,
                state=IntentState.WORKING,
                attempt=_attempt_of(working.order_ref, existing),
                order_ref=working.order_ref,
                order_id=working.order_id,
                decision_ts=decision_ts
                if decision_ts is not None
                else (existing.decision_ts if existing is not None else None),
            )
        )

    def _persist_unresolved(self, key: IntentKey) -> None:
        """Stamp the record UNRESOLVED: an ambiguous POST left state unknown."""
        if self._intents.load(key) is not None:
            self._intents.close(key, IntentState.UNRESOLVED, None, self._now())

    async def _resync_status(self, record: IntentRecord, now: pd.Timestamp) -> None:
        """Resolve an OPEN record that has no working order but a known order id."""
        if record.order_id is None:
            return
        fetched = await self._status(record.order_id)
        if isinstance(fetched, Err):
            return  # keep WORKING; retried next cycle
        status = fetched.value
        state = order_state(status)
        if state is OrderState.FILLED:
            new_state = IntentState.FILLED
        elif order_status_of(status) == "Cancelled":
            new_state = IntentState.UNFILLED
        elif state is OrderState.REJECTED:
            new_state = IntentState.REJECTED
        elif _day_rolled(record.decision_ts, now):
            new_state = IntentState.UNFILLED
        else:
            new_state = IntentState.WORKING
        self._intents.close(record.key, new_state, record.order_id, now)

    # -- internals: account / status reads --------------------------------

    async def _account_net(self, conid: int) -> float | None:
        """The account's net quantity for *conid*, or ``None`` if it cannot be read.

        ``None`` (a failed read) means "unknown", which must not be treated as
        flat — the caller fails closed rather than reducing a book it cannot see.
        """
        account = self._account or self._client.account
        try:
            raw = await self._client.positions_all(account)
        except IbkrError:
            return None
        positions, _ = parse_positions(raw)
        return sum(p.qty for p in positions if p.conid == conid)

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
        position_id = intent.position_id or str(ticket.conid)
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
        """A terminal body with no readable fill quantity: unknown, not "nothing filled"."""
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


def _failed(intent: OrderIntent, message: str) -> OrderResult:
    """A dropped/refused order as a failed ``OrderResult`` (outcome REJECTED)."""
    return OrderResult(
        intent=intent,
        fill=None,
        ok=False,
        message=message,
        outcome=OrderOutcome.REJECTED,
    )


def _adopted_result(record: IntentRecord, found: WorkingOrder) -> OrderResult:
    """The report row for an order adopted at cycle start (a synthetic intent)."""
    intent = OrderIntent(
        symbol=record.key.symbol,
        action=record.key.action,
        qty=found.filled_qty,
        ref_price=0.0,
        reason=f"adopted cycle-start working order {found.order_id}",
        position_id=record.key.position_id,
    )
    return OrderResult(
        intent=intent,
        fill=None,
        ok=True,
        message=f"adopted {found.order_ref} order_id={found.order_id}",
        position_id=record.key.position_id,
        outcome=OrderOutcome.ADOPTED,
    )


__all__ = ["DEFAULT_POLL_INTERVAL_S", "DEFAULT_TIMEOUT_S", "MAX_REPLIES", "IbkrBroker"]
