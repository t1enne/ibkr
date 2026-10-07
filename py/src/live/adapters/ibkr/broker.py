"""``IbkrBroker`` — the real routing edge for a live cycle (plan phase 3, MKT).

Implements ``LiveBroker`` over the Client Portal Gateway. Four things make this
adapter real rather than a wrapper:

1. **Identity + the pending-intent table.** A ``cOID`` is ``scope_tag-token-
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
from dataclasses import dataclass, replace
from typing import cast

import pandas as pd
from ib_rest_api_client.models import SecdefSearchResponseItem

from src.bt.state import ActionType, ExecutionParams, FillEvent, PortfolioState
from src.data.ibkr.client import IbkrClient, IbkrError
from src.data.ibkr.lookup import search_contracts
from src.exec.types import Fill, OrderSide, OrderState
from src.live.adapters.ibkr.mapping import parse_executions, parse_positions
from src.live.adapters.ibkr.orders import (
    OrderMappingError,
    ReplyOutcome,
    Ticket,
    build_ticket,
    classify_reply,
    is_fully_filled,
    is_terminal,
    is_terminal_status,
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
from src.live.adapters.ibkr.trades import Execution
from src.live.broker import OrderResult, position_side_of, trade_signal
from src.live.identity import (
    DEFAULT_TIF,
    OPEN_STATES,
    WEDGED_CYCLES,
    IntentKey,
    IntentRecord,
    IntentState,
    OrderOutcome,
    PendingIntents,
    Resolution,
    WorkingOrder,
    attempt_of,
    intent_key,
    order_ref,
    ref_matches_key,
    ref_prefix,
)
from src.live.ledger_base import LedgerReadError
from src.live.ports import BookExposure
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

#: Slack allowed between the account net and the ledger's booked net on a conid
#: before an OPEN is refused as a divergence. Our book and the account derive from
#: the SAME fill quantities, so a true mismatch is at least a whole (or a
#: fractional) share; only binary-float representation differs. This is a shares
#: tolerance, kept tiny so a single unbooked share — the duplicate-open danger —
#: cannot hide under it.
_EXPOSURE_TOLERANCE = 1e-6

ConidLookup = Callable[[str], Awaitable[int]]
Sleeper = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class _WaitOutcome:
    """A finished fill wait: the durable state it settles, plus what it reports.

    The WAIT BRANCH chooses ``state`` — a terminal status settles FILLED/
    UNFILLED/REJECTED, a deadline with a partial keeps WORKING — so a live
    partial is never stamped terminal by a kind-to-state guess.
    """

    state: IntentState
    result: Result[OrderResult, FeedError]


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
    """The HIGHEST of the attempt in *order_ref_str*'s tail and the record's own.

    Adoption must never REGRESS the durable attempt: a working order whose ref
    parses to a lower attempt than the record already reached still leaves the
    record's attempt as the floor (``max``), so a later re-send keeps minting
    strictly higher cOIDs.
    """
    prior = existing.attempt if existing is not None else 0
    parsed = attempt_of(order_ref_str)
    return prior if parsed is None else max(parsed, prior)


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
        exposure: BookExposure | None = None,
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
        self._exposure = exposure
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
        working -> WORKING, and a DAY order gone (or its status unreadable) after
        its day rolled -> UNFILLED, so a close that could never be confirmed can
        still settle and the key re-mint. A record with no ``order_id`` stays OPEN
        (an empty working read is not proof when the id is unknown). A record left
        unresolved across ``WEDGED_CYCLES`` resyncs is surfaced as a distinct
        WEDGED row — a failed open-orders read returns ``Err`` and leaves every
        record untouched.
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
        alarm: list[OrderResult] = []
        for record in records:
            found = match_working(
                working, ref_prefix(record.key), prefer=record.order_ref
            )
            if found is not None:
                self._persist_working(record.key, found, record.decision_ts)
                adopted.append(_adopted_result(record, found))
                continue
            if await self._resync_status(record, now):
                continue
            stuck = self._bump_stuck(record)
            if stuck.stuck_cycles >= WEDGED_CYCLES:
                alarm.append(_wedged_result(stuck))
        return Ok(tuple(adopted) + tuple(alarm))

    async def place(self, intent: OrderIntent) -> Result[OrderResult, FeedError]:
        """Submit or ADOPT one MKT order and settle its outcome.

        Fail-closed at the pre-flight: an ``open_orders`` read failure is an
        ``Err`` and NO POST follows. Any POST that can submit is settled by a
        SUCCESSFUL working-orders read: absence in a FAILED read is
        ``unresolved``, never "not placed".

        A lone placement carries no confirmed close of our own this cycle, so the
        exposure guard is given a zero in-cycle delta; a cohort leg threads the
        delta its own already-confirmed closes produced (:meth:`place_cohort`).
        """
        return await self._place(intent, in_cycle_delta=0.0)

    async def _place(
        self, intent: OrderIntent, *, in_cycle_delta: float
    ) -> Result[OrderResult, FeedError]:
        """One placement, excusing ``in_cycle_delta`` of our own confirmed closes.

        ``in_cycle_delta`` is the signed net change this cycle's already-confirmed
        reducing fills made to the ACCOUNT on the guard's conid; it is passed in
        explicitly (never stored) so nothing leaks across cycles.
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
        refused = await self._pre_guard(intent, side, conid, in_cycle_delta)
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
        guarded = await self._guard_lost_predecessor(intent, key)
        if guarded is not None:
            return guarded
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
        in_cycle_delta = 0.0
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
            placed = await self._place(
                scaled.get(ident, intent), in_cycle_delta=in_cycle_delta
            )
            if isinstance(placed, Err):
                error = cast("FeedError", placed.error)
                message = f"{error.kind}: {error.message}"
                self._log(f"order failed {intent.symbol}: {message}")
                results.append(_failed(intent, message, error.kind, error.filled_qty))
            else:
                results.append(placed.value)
                # The close's fill has moved the ACCOUNT net but not our durable
                # book yet; carry its signed change so a same-cohort OPEN is not
                # refused as a divergence our own close fully explains.
                in_cycle_delta += _confirmed_reduce_delta(
                    intent, placed.value, position_side_of(self._book, intent)
                )
            if intent.action is ActionType.close and not results[-1].ok:
                close_failed = True
        return Ok(tuple(results))

    async def close(self) -> Result[None, FeedError]:
        """Close the HTTP client (its pool outlives nothing once the cycle ends)."""
        await self._client.aclose()
        return Ok(None)

    async def wait_filled(
        self, order_id: str, intent: OrderIntent, ticket: Ticket
    ) -> _WaitOutcome:
        """Poll the order to a terminal status; map it to a fill or a typed failure.

        The ``_WaitOutcome.state`` is chosen by the BRANCH, not the error kind: a
        terminal status settles FILLED/UNFILLED/REJECTED, a deadline with a
        partial fill keeps WORKING. Outcomes (all reported, none silently
        accepted):

        - terminal ``Filled`` for the whole size -> ``Ok`` with the fill;
        - a complete fill whose status has not yet flipped (``cum_fill`` reaches
          ``total_size``) -> ``Ok``, because a fill cannot exceed the size;
        - terminal non-filled status (``Cancelled``/``Inactive``) -> ``rejected``;
        - a partial fill on a DEAD order -> ``unfilled``, carrying the qty that
          DID fill;
        - still working at the deadline -> ``timeout`` (a partial is reported in
          the message), settling WORKING so the next cycle re-checks it;
        - a terminal ``Filled`` whose body omits/garbles ``cum_fill`` ->
          ``unresolved`` (it filled; its size is unknown) — never "nothing filled".
        """
        deadline = self._monotonic() + self._timeout_s
        status: dict[str, object] = {}
        while True:
            fetched = await self._status(order_id)
            if isinstance(fetched, Err):
                return _WaitOutcome(
                    IntentState.WORKING, Err(cast("FeedError", fetched.error))
                )
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
        """Every WORKING, joinable order of ours keyed by ``cOID``, or ``Err``.

        A failed read is ALWAYS an ``Err`` — never ``None`` or an empty index —
        so the caller fails closed (no POST) rather than mistaking "unreadable"
        for "nothing working" (the fail-open bug this replaces). A foreign row
        (no ``order_ref``) is dropped by the parser; a row with no readable
        ``order_id`` cannot be joined or waited on and is skipped; and a row the
        gateway lists as already terminal (it lists filled/cancelled orders for
        the session) must NOT be adopted — adopting a dead order would block a
        legitimate re-send.
        """
        try:
            raw = await self._client.open_orders()
        except IbkrError as exc:
            return Err(feed_error(exc.kind, f"open_orders: {exc}"))
        index: dict[str, WorkingOrder] = {}
        for entry in raw:
            parsed = parse_working_order(entry)
            if parsed is None or not parsed.order_id:
                continue
            if is_terminal_status(parsed.status):
                continue
            index.setdefault(parsed.order_ref, parsed)
        return Ok(index)

    async def _pre_guard(
        self,
        intent: OrderIntent,
        side: OrderSide,
        conid: int,
        in_cycle_delta: float,
    ) -> FeedError | None:
        """The pre-submit safety guards, run before any working-orders read.

        A REDUCING order is refused unless the WHOLE-share quantity it will send
        is no larger than the account net AND the net is on the side the order
        reduces (a SELL reduces a long net, a BUY a short one) — so a close on a
        shared account can never flip a smaller foreign net into an open. An
        unreadable net still fails closed. An OPEN is bounded by the decision-time
        ``cash_bound`` applied to the whole-share quantity the ticket will carry
        (never the pre-round request, which could round up past the bound), then
        cross-checked against the account net (:meth:`_open_exposure_guard`) net of
        ``in_cycle_delta`` — the signed change this cohort's own confirmed closes
        already made to the account.
        """
        if intent.action is ActionType.close:
            return await self._close_guard(intent, side, conid)
        try:
            whole = whole_quantity(intent)
        except OrderMappingError as exc:
            return feed_error("rejected", f"{intent.symbol}: {exc}", intent.symbol)
        # The cash bound is LOCAL (no I/O): check it first so an intent that
        # cannot even state its funding is refused without an account round trip.
        refused = self._open_cash_guard(intent, whole)
        if refused is not None:
            return refused
        return await self._open_exposure_guard(intent, conid, in_cycle_delta)

    async def _open_exposure_guard(
        self, intent: OrderIntent, conid: int, in_cycle_delta: float
    ) -> FeedError | None:
        """Refuse an open whose conid the account holds but our book cannot explain.

        An OPEN routes to a fresh cOID whenever the predecessor's record is
        terminal or pruned (see :meth:`_resolve`), so IBKR's own dedupe never
        blocks a re-send. The only thing stopping a second full-size open is our
        book, and the book is derived from the SAME lagging executions feed a
        missed fill would hide in. So before opening we compare two independent
        reads of the same conid:

        - the ACCOUNT net (``positions_all`` — the broker's truth);
        - the BOOKED net the ledger can account for (``BookExposure.net_exposure``,
          summed over ALL scopes sharing this book — our durable rows) plus
          ``in_cycle_delta``.

        ``in_cycle_delta`` is the signed net change THIS cohort's already-confirmed
        reducing fills made to the account (signed to match ``net_exposure``: a
        close of a long lot sells and subtracts, a short cover adds). A same-cycle
        close->open flip places the close first and awaits it to terminal, so by
        the open's guard the account has already applied our close while the
        durable book still shows the pre-close lot; crediting our own confirmed
        closes keeps that legitimate flip from reading as a divergence. It excuses
        ONLY a change our own sends produced: a divergence larger than the delta,
        or one with no confirmed close behind it, is still refused.

        They derive from the same fills, so in steady state they are equal. A
        divergence beyond ``_EXPOSURE_TOLERANCE`` means one of two things, both
        unsafe to open on: the account holds a fill we never booked (a live order
        we cannot see), or our book is ahead of the account (a phantom lot).
        Either way we REFUSE — this is a divergence check, NOT a cap: a legitimate
        open is allowed precisely when the account net matches what every scope
        books, so another scope holding the symbol is accounted for rather than
        blocked.

        This fires on a shared account where a human trades the same conid: that
        is an unexplained net that is nobody's scope, EXACTLY the case to refuse
        on until an operator reconciles it. Pending/open intent records are read
        as part of the book (they are our rows) but are never credited as a
        numeric excuse for a surplus — a fill in flight for a pending intent IS
        the lag this guard exists to catch.

        Fails CLOSED, like the rest of the edge: no oracle, an unreadable account
        net, or an unreadable book all refuse the open. The refusal carries the
        distinct ``divergence`` kind so an operator can tell it apart from a
        broker ``rejected``/``unresolved``.
        """
        if self._exposure is None:
            return feed_error(
                "divergence",
                f"refused open {intent.symbol} (conid {conid}): no exposure oracle, "
                f"cannot cross-check the account against the book",
                intent.symbol,
            )
        net = await self._account_net(conid)
        if net is None:
            return feed_error(
                "divergence",
                f"refused open {intent.symbol} (conid {conid}): account net "
                f"unreadable, cannot prove the conid is flat in the account",
                intent.symbol,
            )
        try:
            booked = self._exposure.net_exposure(conid)
        except LedgerReadError as exc:
            return feed_error(
                "divergence",
                f"refused open {intent.symbol} (conid {conid}): booked exposure "
                f"unreadable ({exc}), cannot cross-check the account",
                intent.symbol,
            )
        if abs(net - (booked + in_cycle_delta)) > _EXPOSURE_TOLERANCE:
            return feed_error(
                "divergence",
                f"refused open {intent.symbol} (conid {conid}): account net {net:g} "
                f"disagrees with the booked net {booked:g} plus our in-cycle closes "
                f"{in_cycle_delta:+g} (tol {_EXPOSURE_TOLERANCE:g}) — a fill we "
                f"cannot see may be live, or our book is ahead of the account",
                intent.symbol,
            )
        return None

    async def _close_guard(
        self, intent: OrderIntent, side: OrderSide, conid: int
    ) -> FeedError | None:
        """Refuse a reducing order the account net cannot safely absorb.

        The account net is the SHARED account's net for the conid, so it can be
        smaller than our book, or held by another scope/human on the opposite
        side. Refusing (never clamping — a clamped close would silently leave the
        lot half-open) when the sent quantity exceeds ``|net|`` or the net is not
        on the reduce side stops a close from opening a position. An unreadable
        net fails closed; the flat check keeps the old fail-closed behaviour for
        a genuinely empty book. ``_FLAT_EPS`` absorbs float noise and a fractional
        net that shrank slightly between cycles.
        """
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
        reducing = net > 0 if side is OrderSide.SELL else net < 0
        if not reducing:
            return feed_error(
                "rejected",
                f"refused close {intent.symbol} (conid {conid}): account net "
                f"{net:g} is on the opposite side to the {side.value} close, which "
                f"would open a position",
                intent.symbol,
            )
        whole = whole_quantity(intent)
        if whole > abs(net) + _FLAT_EPS:
            return feed_error(
                "rejected",
                f"refused close {intent.symbol} (conid {conid}): close qty {whole:g} "
                f"exceeds the account net {abs(net):g}, which would open a position",
                intent.symbol,
            )
        return None

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

        A working order carrying the key's exact prefix is ADOPTED. With none,
        only a record that PROVES the predecessor was never confirmed placed may
        submit: a PENDING record (minted, submit not confirmed) or no record at
        all. An OPEN record in any other state is SKIPPED — ``UNRESOLVED`` means
        an ambiguous POST left 'whether it is live' unknown, so a clean read that
        lists nothing is NOT proof it was never placed (``/iserver/account/orders``
        lists WORKING orders AND any order filled or cancelled in the CURRENT
        session, so absence cannot tell 'never sent' from 'sent and filled', and
        cannot even prove a same-session order has stopped working), and a known
        ``order_id`` with no working order is likewise not proof. A record in a
        terminal state submits a fresh attempt.
        """
        existing = self._intents.load(key)
        found = match_working(
            tuple(index.values()),
            ref_prefix(key),
            prefer=existing.order_ref if existing is not None else None,
        )
        if found is not None:
            return Resolution(
                "adopt",
                found.order_id,
                f"working order {found.order_id} carries {found.order_ref}",
            )
        if existing is None or existing.state not in OPEN_STATES:
            return Resolution("submit", None, "no open intent for this key")
        if existing.state is IntentState.PENDING:
            return Resolution(
                "submit", None, "pending intent was never confirmed placed"
            )
        if not existing.order_id:
            return Resolution(
                "skip",
                None,
                f"open intent {existing.state.value} has no known order id and no "
                f"working order; not re-minting",
            )
        return Resolution(
            "skip",
            existing.order_id,
            f"open intent {existing.state.value} holds order {existing.order_id} "
            f"but no working order carries its prefix; not re-minting",
        )

    # -- internals: lost-predecessor sweep --------------------------------
    #
    # Before minting a NEW attempt the executions window is swept for a fill
    # attributable to this key. The ref is bar-free, so a fill from ANY attempt
    # maps to the same key; finding one means a predecessor filled and a submit
    # would be a duplicate. Swept ONLY when the durable store cannot vouch for
    # the predecessor (no record, or a PENDING row) — a terminal record already
    # accounts for its predecessor, so a legitimate re-send (a close re-closed
    # after a partial) must NOT be blocked. It lives at MINT time, not resync:
    # only there can a failed executions read block the POST (resync failures
    # are non-fatal by design, so a resync-time sweep could not be fail-closed,
    # the INV-1 invariant).

    async def _guard_lost_predecessor(
        self, intent: OrderIntent, key: IntentKey
    ) -> Result[OrderResult, FeedError] | None:
        """Refuse to mint when the executions window proves the key already filled.

        ``None`` means no evidence of a predecessor and the caller may mint; an
        ``Err`` (a failed executions read) fails closed so no POST follows.
        """
        existing = self._intents.load(key)
        if existing is not None and existing.state is not IntentState.PENDING:
            return None
        swept = await self._sweep_key_fill(key)
        if isinstance(swept, Err):
            return Err(cast("FeedError", swept.error))
        execution = swept.value
        if execution is None:
            return None
        self._adopt_execution(key, execution)
        return Err(
            feed_error(
                "unresolved",
                f"{intent.symbol}: executions window already carries "
                f"{execution.order_ref} (order {execution.order_id}) for this key "
                f"— not minting a duplicate; its terminal state is re-checked next "
                f"cycle",
                intent.symbol,
            )
        )

    async def _sweep_key_fill(
        self, key: IntentKey
    ) -> Result[Execution | None, FeedError]:
        """One execution already attributable to *key*, or ``None`` (fail-closed read)."""
        try:
            raw = await self._client.trades()
        except IbkrError as exc:
            return Err(feed_error(exc.kind, f"trades: {exc}", key.symbol))
        executions, _ = parse_executions(raw)
        return Ok(
            next(
                (
                    e
                    for e in executions
                    if ref_matches_key(self._scope, key, e.order_ref)
                ),
                None,
            )
        )

    def _adopt_execution(self, key: IntentKey, execution: Execution) -> None:
        """Persist WORKING for *key*, adopting a found execution's order identity.

        Storing the execution's ``order_id`` lets the NEXT cycle's ``resync``
        settle the true terminal state through ``order_status``; marking it
        WORKING (not FILLED) avoids asserting a full fill a partial would
        contradict.
        """
        existing = self._intents.load(key)
        self._intents.save(
            IntentRecord(
                key=key,
                state=IntentState.WORKING,
                attempt=_attempt_of(execution.order_ref, existing),
                order_ref=execution.order_ref,
                order_id=execution.order_id,
                decision_ts=existing.decision_ts if existing is not None else None,
            )
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
        existing = self._intents.load(key)
        found = match_working(
            tuple(index.values()),
            ref_prefix(key),
            prefer=existing.order_ref if existing is not None else None,
        )
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
                    f"confirm {ticket.order_ref}: pre-confirm working-orders read failed",
                    read_ok=False,
                )
            already = match_working(
                tuple(guard.value.values()), ref_prefix(key), prefer=ticket.order_ref
            )
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
            return self._ambiguous_error(reason, read_ok=False)
        existing = self._intents.load(key)
        found = match_working(
            tuple(index.value.values()),
            ref_prefix(key),
            prefer=existing.order_ref if existing is not None else None,
        )
        if found is not None:
            self._persist_working(key, found, None)
            self._log(f"{reason}; adopting working order {found.order_id}")
            return Ok(found.order_id)
        self._persist_unresolved(key)
        return self._ambiguous_error(reason, read_ok=True)

    def _ambiguous_error(self, reason: str, *, read_ok: bool) -> Err[str, FeedError]:
        """The unresolved failure for an order whose submission left state unknown.

        The working-orders attestation is only claimed when that read actually
        SUCCEEDED: ``read_ok=True`` says "no working order carries the key prefix",
        ``read_ok=False`` (the read FAILED) says "whether a working order exists is
        unknown" — never positive evidence we do not have (D5).
        """
        observation = (
            "no working order carries the key prefix"
            if read_ok
            else "the working-orders read failed, so whether a working order exists "
            "is unknown"
        )
        return Err(
            FeedError(
                kind="unresolved",
                message=(
                    f"{reason}; whether it was placed is unknown ({observation}) — "
                    f"do not assume it was not placed"
                ),
            )
        )

    def _settle_wait(
        self,
        key: IntentKey,
        outcome: _WaitOutcome,
        order_id: str | None,
        *,
        adopted: bool,
    ) -> Result[OrderResult, FeedError]:
        """Persist the durable state the wait branch chose, and pass the result on.

        The ORDER (not an error kind) decides state: a terminal status settles
        FILLED/UNFILLED/REJECTED, a deadline with a partial settles WORKING, so
        a still-live partial is never marked terminal.
        """
        now = self._now()
        result = outcome.result
        if isinstance(result, Err):
            self._intents.close(key, outcome.state, order_id, now)
            return Err(cast("FeedError", result.error))
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
        """Stamp the record UNRESOLVED when an ambiguous POST left state unknown.

        A WORKING record is NEVER downgraded: it already carries the order
        provenance (ref/id) a later cycle needs to decide, and UNRESOLVED would
        erase it.
        """
        existing = self._intents.load(key)
        if existing is None or existing.state is IntentState.WORKING:
            return
        self._intents.close(key, IntentState.UNRESOLVED, None, self._now())

    def _bump_stuck(self, record: IntentRecord) -> IntentRecord:
        """Increment an OPEN record's wedged-key counter (resync could not resolve it)."""
        updated = replace(record, stuck_cycles=record.stuck_cycles + 1)
        self._intents.save(updated)
        return updated

    async def _resync_status(self, record: IntentRecord, now: pd.Timestamp) -> bool:
        """Resolve an OPEN record with no working order; True when settled or known.

        False means the record's true state is STILL UNKNOWN this cycle (a failed
        status read on a session that can no longer see the order, or an id-less
        UNRESOLVED record before its day rolls) — the caller counts it toward the
        wedged-key alarm. A record with a known id is asked directly; when that
        read FAILS, the same DAY-rollover inference the id-less branch applies is
        used, because the status endpoint only covers the current brokerage
        session and a DAY order cannot outlive its session — so once its day has
        rolled and nothing is working it has expired unfilled (re-mintable).
        """
        if not record.order_id:
            if record.state is IntentState.UNRESOLVED and self._day_expired(
                record, now
            ):
                self._intents.close(record.key, IntentState.UNFILLED, None, now)
                return True
            return False
        fetched = await self._status(record.order_id)
        if isinstance(fetched, Err):
            if self._day_expired(record, now):
                self._intents.close(
                    record.key, IntentState.UNFILLED, record.order_id, now
                )
                return True
            return False  # whether it is live is unknown; retried next cycle
        status = fetched.value
        state = order_state(status)
        if state is OrderState.FILLED:
            new_state = IntentState.FILLED
        elif order_status_of(status) == "Cancelled":
            new_state = IntentState.UNFILLED
        elif state is OrderState.REJECTED:
            new_state = IntentState.REJECTED
        elif self._day_expired(record, now):
            new_state = IntentState.UNFILLED
        else:
            new_state = IntentState.WORKING
        self._intents.close(record.key, new_state, record.order_id, now)
        return True

    @staticmethod
    def _day_expired(record: IntentRecord, now: pd.Timestamp) -> bool:
        """Whether the record's DAY order can no longer be working (its day rolled)."""
        return record.tif == DEFAULT_TIF and _day_rolled(record.decision_ts, now)

    # -- internals: account / status reads --------------------------------

    async def _account_net(self, conid: int) -> float | None:
        """The account's net quantity for *conid*, or ``None`` if it cannot be read.

        ``None`` (a failed read) means "unknown": the caller fails closed — a
        reducing order is REFUSED rather than sent against a book it cannot see,
        because the account net (not our book) is what a close must not exceed.
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
    ) -> _WaitOutcome:
        """A terminal (or completed) status -> a fill, or a typed terminal failure."""
        raw = order_status_of(status)
        fill = self._filled(ticket, intent, status)
        if fill is None:
            return self._no_readable_fill(order_id, intent, ticket, raw, status)
        if not is_fully_filled(status, fill.qty):
            return _WaitOutcome(
                IntentState.UNFILLED,
                Err(
                    FeedError(
                        kind="unfilled",
                        message=(
                            f"order {ticket.order_ref} ({order_id}) status {raw} with "
                            f"only {fill.qty:g} of {intent.qty:g} filled"
                        ),
                        symbol=intent.symbol,
                        filled_qty=fill.qty,
                    )
                ),
            )
        position_id = intent.position_id or str(ticket.conid)
        message = (
            f"{intent.action.value} {intent.symbol} qty={fill.qty:g} "
            f"@ {fill.price:.4f} cOID={ticket.order_ref} order_id={order_id}"
        )
        self._log(message)
        return _WaitOutcome(
            IntentState.FILLED,
            Ok(
                OrderResult(
                    intent=intent,
                    fill=self._fill_event(intent, ticket, fill),
                    ok=True,
                    message=message,
                    position_id=position_id,
                    filled_qty=fill.qty,
                )
            ),
        )

    def _no_readable_fill(
        self,
        order_id: str,
        intent: OrderIntent,
        ticket: Ticket,
        raw: str,
        status: dict[str, object],
    ) -> _WaitOutcome:
        """A terminal body with no readable fill quantity: unknown, not "nothing filled"."""
        if order_state(status) is OrderState.FILLED:
            return _WaitOutcome(
                IntentState.WORKING,
                Err(
                    FeedError(
                        kind="unresolved",
                        message=(
                            f"order {ticket.order_ref} ({order_id}) status {raw}: "
                            f"terminal Filled but cum_fill unreadable — the order "
                            f"filled; its quantity is unknown"
                        ),
                        symbol=intent.symbol,
                    )
                ),
            )
        return _WaitOutcome(
            IntentState.REJECTED,
            Err(
                FeedError(
                    kind="rejected",
                    message=(
                        f"order {ticket.order_ref} ({order_id}) status {raw}: "
                        f"nothing filled"
                    ),
                    symbol=intent.symbol,
                )
            ),
        )

    def _unresolved_result(
        self,
        order_id: str,
        intent: OrderIntent,
        ticket: Ticket,
        status: dict[str, object],
    ) -> _WaitOutcome:
        """No terminal status at the deadline -> a timeout, settling WORKING.

        A partial fill here lives in the MESSAGE, not a terminal state: the order
        is still working, so stamping it UNFILLED would let the next cycle mint a
        duplicate while this order is live.
        """
        raw = order_status_of(status)
        fill = self._filled(ticket, intent, status)
        filled_qty = fill.qty if fill is not None else 0.0
        if filled_qty > 0:
            message = (
                f"order {ticket.order_ref} ({order_id}) still {raw} after "
                f"{self._timeout_s:g}s with {filled_qty:g} of "
                f"{intent.qty:g} filled"
            )
        else:
            message = (
                f"order {ticket.order_ref} ({order_id}) still {raw} after "
                f"{self._timeout_s:g}s: no terminal status"
            )
        return _WaitOutcome(
            IntentState.WORKING,
            Err(
                FeedError(
                    kind="timeout",
                    message=message,
                    symbol=intent.symbol,
                    filled_qty=filled_qty,
                )
            ),
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


def _outcome_for_kind(kind: str) -> OrderOutcome:
    """The ``OrderOutcome`` a failure's kind reports (never a terminal guess).

    The kind is chosen to AGREE with the durable state the failing branch
    settled, so this is just the state's disposition in kind form: a still-live
    deadline is ``timeout`` and a dead order is ``unfilled``/``rejected``. An
    unreadable/auth/rate-limited read is an UNKNOWN state, so it maps to
    ``unresolved`` — and an unknown kind NEVER defaults to ``REJECTED`` (a
    transport failure while polling a live order must not render as a refusal an
    operator answers by re-placing).
    """
    return {
        "rejected": OrderOutcome.REJECTED,
        "unfilled": OrderOutcome.UNFILLED,
        "timeout": OrderOutcome.TIMEOUT,
        "unresolved": OrderOutcome.UNRESOLVED,
        "transport": OrderOutcome.UNRESOLVED,
        "auth": OrderOutcome.UNRESOLVED,
        "rate_limit": OrderOutcome.UNRESOLVED,
        # A deliberate refusal on an unexplained account net — its own outcome, so
        # it is neither a broker rejection nor an unresolved order.
        "divergence": OrderOutcome.DIVERGENCE,
    }.get(kind, OrderOutcome.UNRESOLVED)


def _confirmed_reduce_delta(
    intent: OrderIntent, result: OrderResult, lot_side: ActionType | None
) -> float:
    """The signed change a confirmed reducing fill already made to the ACCOUNT net.

    Signed to match ``BookExposure.net_exposure`` (a long lot adds, a short lot
    subtracts): a close of a LONG lot sells and lowers the account net by the
    filled qty (negative), a SHORT cover raises it (positive). Anything that is
    not a confirmed reducing fill — a non-close, a refusal, an unknown fill size,
    or a close whose lot side cannot be read — contributes nothing, so the guard
    credits no excuse and still refuses on its own merits.
    """
    if intent.action is not ActionType.close or not result.ok:
        return 0.0
    filled = result.filled_qty
    if filled is None or lot_side is None:
        return 0.0
    return -filled if lot_side is ActionType.long else filled


def _failed(
    intent: OrderIntent,
    message: str,
    kind: str = "rejected",
    filled_qty: float | None = None,
) -> OrderResult:
    """A dropped/refused order as a failed ``OrderResult`` carrying its outcome."""
    return OrderResult(
        intent=intent,
        fill=None,
        ok=False,
        message=message,
        outcome=_outcome_for_kind(kind),
        error_kind=kind,
        filled_qty=filled_qty,
    )


def _adopted_result(record: IntentRecord, found: WorkingOrder) -> OrderResult:
    """The report row for an order adopted at cycle start (a synthetic intent).

    ``filled_qty`` is UNKNOWN (``None``), not the working row's current fill: the
    durable intent record does not store the ask, so an adopted row cannot say how
    much of the original order is still unfilled. Reporting ``found.filled_qty``
    against a synthetic intent whose ``qty`` is that same number would fabricate a
    zero shortfall — a partially-filled working order rendered as "filled to ask".
    The synthetic intent keeps a displayable ``qty``; only the shortfall claim is
    withheld.
    """
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
        filled_qty=None,
    )


def _wedged_result(record: IntentRecord) -> OrderResult:
    """A distinct, LOUD report row for an OPEN key nothing could settle.

    Emitted once an OPEN record's ``stuck_cycles`` reaches ``WEDGED_CYCLES`` so a
    never-resolvable intent (a broker-side cancel the gateway cannot confirm, a
    persistently unreadable status endpoint) surfaces instead of one more
    ``unresolved`` line. The message names the operator escape.
    """
    where = f"/{record.key.position_id}" if record.key.position_id else ""
    intent = OrderIntent(
        symbol=record.key.symbol,
        action=record.key.action,
        qty=0.0,
        ref_price=0.0,
        reason=f"WEDGED {record.state.value} across {record.stuck_cycles} resyncs",
        position_id=record.key.position_id,
    )
    return OrderResult(
        intent=intent,
        fill=None,
        ok=False,
        message=(
            f"WEDGED key {record.key.symbol}/{record.key.action.value}{where}: "
            f"{record.state.value} order_id={record.order_id or 'unknown'} "
            f"unresolved across {record.stuck_cycles} resyncs — the broker session "
            f"cannot confirm it. Verify broker-side, then `ibkr live abandon`."
        ),
        position_id=record.key.position_id,
        outcome=OrderOutcome.WEDGED,
        error_kind="unresolved",
    )


__all__ = ["DEFAULT_POLL_INTERVAL_S", "DEFAULT_TIMEOUT_S", "MAX_REPLIES", "IbkrBroker"]
