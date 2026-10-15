"""``IbkrBroker`` — the real routing edge for a live cycle (plan phase 3, MKT).

Implements ``LiveBroker`` over the Client Portal Gateway. Four things make this
adapter real rather than a wrapper:

1. **Identity + broker-truth dedupe.** A ``cOID`` is a pure ``scope_tag-token``
   built by ``src.live.identity`` from ``(scope, symbol, action, position_id)`` —
   bar-free, no attempt tail. There is NO durable open-order store: whether an
   order exists is decided by READS. Before any mint the adapter reads the
   working orders (INV-1) and, when none carries our ref, sweeps the executions
   window; a working order with our ref is ADOPTED, a swept fill means a
   predecessor already filled (``unresolved``, never re-mint).
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

**Fail-closed on identity.** Before any POST — submit or reply-confirm — a
working-orders read must SUCCEED. A failed read is an ``Err`` and no POST
follows: a duplicate order is unbounded exposure, while a skipped cycle is
recoverable on the next bar because the intent is still desired. Any POST that
can submit (a submit OR a confirm) is settled by a SUCCESSFUL working-orders
read; absence in a FAILED read is ``unresolved``, never "not placed".
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import cast

import pandas as pd
from ib_rest_api_client.models import SecdefSearchResponseItem

from src.bt.state import (
    ActionType,
    ExecutionParams,
    FillEvent,
    PortfolioState,
    Position,
)
from src.data.ibkr.client import IbkrClient, IbkrError
from src.data.ibkr.lookup import search_contracts
from src.data.types import SymbolSchema
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
from src.live.adapters.ibkr.protect import BracketProtectionMixin
from src.live.adapters.ibkr.trades import Execution
from src.exec.refs import scope_tag
from src.live.protect import ProtectionTarget, resting_legs
from src.live.pure import OrderResult, position_side_of, trade_signal
from src.live.identity import (
    IntentKey,
    OrderOutcome,
    WorkingOrder,
    intent_key,
    order_ref,
    ref_is_ours,
    ref_prefix,
)
from src.live.ledger_base import LedgerReadError
from src.live.ports import BookExposure
from src.live.result import Err, Ok, Result
from src.live.types import (
    FLAT_QTY_EPS,
    FeedError,
    OrderIntent,
    PortfolioView,
    feed_error,
)

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

#: The kind on an OPEN the cohort's STRUCTURAL rules dropped (``_refuse_opens``/
#: ``_apply_scale``): the cohort cannot state a shared cash bound, or a scaled qty
#: floors to 0 shares. Deliberately NOT ``rejected``: a structural drop recurs
#: every cycle forever, so it must read as unsafe (D3). A genuine cash-exhaustion
#: refusal never reaches here — it arrives from placement as ``rejected``.
_STRUCTURAL_KIND = "unfunded"


@dataclass(frozen=True)
class _WaitOutcome:
    """A finished fill wait, carrying only what the caller reports.

    There is no durable state to settle: the outcome (fill / typed failure)
    comes straight from the wait branch, and the next cycle's reads — working
    orders + positions — are what resolve an order from here on.
    """

    result: Result[OrderResult, FeedError]


def _names_the_ticker(candidate: SecdefSearchResponseItem, ticker: str) -> bool:
    """Whether *candidate*'s underlying symbol is exactly *ticker* (case-folded)."""
    symbol = candidate.symbol
    return isinstance(symbol, str) and symbol.strip().upper() == ticker.strip().upper()


def _db_conid_for_ticker(ticker: str) -> int | None:
    """The conid our candle DB ``symbol`` table pairs with *ticker*, if any.

    A PREFERENCE, never an override: used only as a tie-break among secdef
    candidates that share a ticker. ``None`` when the ticker is unknown to the
    DB or the table is unreadable (a read-only helper, so a missing fixture DB
    degrades to no preference rather than a raise).
    """
    try:
        s = SymbolSchema.get_or_none(SymbolSchema.ticker == ticker.strip().upper())
    except Exception:
        return None
    return int(s.conid) if s is not None else None


def _prefer_db_conid(
    matches: tuple[SecdefSearchResponseItem, ...], ticker: str
) -> SecdefSearchResponseItem:
    """Pick one candidate when several share *ticker*, preferring the DB's conid.

    The secdef search routinely returns multiple listings of one ticker (e.g.
    SHOP — two US STK rows, only one of which is the contract our data/backtest
    treats as SHOP). The candle DB ``symbol`` row is the authoritative local
    answer: among the candidates, the one whose conid matches it wins. If none
    does, we fail closed — a live order must never land on a guess.
    """
    known = _db_conid_for_ticker(ticker)
    preferred = [c for c in matches if known is not None and str(c.conid) == str(known)]
    if preferred:
        return preferred[0]
    hosts = ", ".join(
        f"{c.conid} [{c.description}]"
        for c in sorted(matches, key=lambda c: str(c.conid))
    )
    raise ValueError(
        f"ambiguous contract for {ticker}: candidates {hosts} "
        f"(none matches DB conid {known or 'unknown'})"
    )


async def _default_conid_lookup(ticker: str) -> int:
    """Resolve *ticker* to its IBKR conid, VERIFYING it is the intended contract.

    The search is by ticker, so an unverified first hit can put a real order on
    the wrong instrument. Refuses (``ValueError`` -> a typed ``FeedError``) when
    no candidate names the ticker, when the candidates share no single conid AND
    none matches the candle DB's conid for the ticker (ambiguous), or when the
    chosen contract is ``restricted``.
    """
    candidates = await search_contracts(ticker)
    matches = tuple(c for c in candidates if _names_the_ticker(c, ticker))
    if not matches:
        raise ValueError(
            f"secdef search for {ticker} returned no contract naming {ticker}"
        )
    conids = {str(c.conid) for c in matches}
    if len(conids) == 1:
        chosen = matches[0]
    elif len(conids) > 1:
        chosen = _prefer_db_conid(matches, ticker)
    else:
        raise ValueError(f"secdef search for {ticker} returned no conid")
    if chosen.restricted is True:
        raise ValueError(f"contract for {ticker} is restricted (not tradable)")
    conid = chosen.conid
    if not isinstance(conid, str) or not conid.strip():
        raise ValueError(f"secdef search for {ticker} returned no conid")
    return int(conid)


class IbkrBroker(BracketProtectionMixin):
    """``LiveBroker`` over the gateway: MKT orders, reply loop, bounded fill wait.

    ``dry_run`` is the defence-in-depth flag: a broker built for a dry run places
    nothing, so a missing CLI guard cannot turn into a real order.

    Protective brackets (spec Steps 4–5) are added by :class:`BracketProtectionMixin`:
    a confirmed entry fill places + confirms the OCA pair in the same cycle; a
    close cancels-and-confirms the pair first; a level tighten REPLACEs both legs.
    """

    def __init__(
        self,
        client: IbkrClient,
        *,
        scope: str,
        params: ExecutionParams,
        account: str | None = None,
        exposure: BookExposure | None = None,
        dry_run: bool = False,
        allow_oth: bool = False,
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
        self._params = params
        self._account = account
        self._exposure = exposure
        self._dry_run = dry_run
        self._allow_oth = allow_oth
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
        """Reconcile at cycle start — a no-op now that reads are the authority.

        There is no durable open-order store to reconcile: every order's existence
        is decided from the broker's reads inside :meth:`place` (working orders
        first, then the executions sweep, with the account positions in the
        exposure guard), so a cycle-start pass has nothing local to resolve.

        The protective backstop is NOT here but in :meth:`attest_protection`: it
        needs the cycle's book, which this signature does not carry.
        """
        return Ok(())

    async def attest_protection(
        self, book: PortfolioState
    ) -> Result[tuple[OrderResult, ...], FeedError]:
        """The per-cycle naked-lot backstop over *book*'s open lots (§4.2, D1).

        Builds one target per OPEN lot the scope's own book carries whose
        advisory level says it SHOULD rest a bracket (its ``position_id`` IS the
        conid on this edge), then hands them to the mixin's attestation, which
        reads the working orders once and reports ``UNPROTECTED`` for every lot
        resting no protective leg of ours.

        The level gate is what separates a NAKED lot from one the strategy never
        armed: a lot with neither ``stop_loss`` nor ``take_profit`` was opened
        unprotected by design and is NOT a target — flagging it would be a
        permanent false alarm. On this edge the levels arrive ``None`` (item 5
        of the pending list has not landed the read-side level), so the backstop
        is inert here until the advisory level rides the lot read.

        Scope-clean by construction: only lots in OUR book are checked, so a
        conid another scope or a human holds is never touched. Attest-only — no
        order is placed, cancelled or amended. Empty book ⇒ no read at all.
        """
        targets = tuple(
            ProtectionTarget(
                conid=int(str(lot.position_id)),
                symbol=lot.symbol,
                position_id=str(lot.position_id),
                qty=lot.qty,
            )
            for lot in _open_lots(book)
            if lot.position_id
            and (lot.stop_loss is not None or lot.take_profit is not None)
        )
        return Ok(await self._attest_protection(targets))

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
        account = await self._account_id()
        prepared = await self._prepare_leg(intent)
        if isinstance(prepared, Err):
            return Err(cast("FeedError", prepared.error))
        side, conid = prepared.value
        key = intent_key(self._scope, intent)
        refused = await self._pre_guard(intent, side, conid, in_cycle_delta)
        if refused is not None:
            return Err(refused)
        # INV-1: a SUCCESSFUL working-orders read is a precondition for ANY POST.
        # The read is also the dedupe: a working order carrying our pure ref means
        # the order already exists at the broker — ADOPT it, never re-submit.
        index = await self._working_index()
        if isinstance(index, Err):
            return Err(cast("FeedError", index.error))
        found = match_working(
            tuple(index.value.values()), ref_prefix(key), prefer=order_ref(key)
        )
        if found is not None:
            self._log(
                f"adopting already-working order {found.order_id} for {found.order_ref}"
            )
            return await self._adopt(intent, key, side, conid, found)
        # No working order: the executions sweep proves no predecessor filled.
        # A found fill is a duplicate-open danger — report ``unresolved``, never
        # re-mint. A failed read fails closed (no POST).
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
                results.append(_failed(intent, drop.reason, kind=_STRUCTURAL_KIND))
                continue
            if intent.action is not ActionType.close and close_failed:
                message = (
                    f"dropped open {intent.symbol}: its funding close leg did not "
                    f"fill; refusing to deploy prospective cash"
                )
                self._log(message)
                results.append(_failed(intent, message))
                continue
            if intent.protective:
                results.extend(await self._route_protective(intent))
                continue
            if intent.action is ActionType.close:
                # Invariant 6: a protected lot's bracket is cancelled-and-
                # confirmed BEFORE its close; an unconfirmable cancel refuses.
                refusals = await self._cancel_pair_before_close(intent)
                if refusals is not None:
                    results.extend(refusals)
                    close_failed = True
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
                if (
                    placed.value.ok
                    and placed.value.fill is not None
                    and intent.action
                    in (
                        ActionType.long,
                        ActionType.short,
                    )
                ):
                    results.extend(
                        await self._protect_after_fill(
                            intent, placed.value, in_cycle_delta
                        )
                    )
            if intent.action is ActionType.close and not results[-1].ok:
                close_failed = True
        return Ok(tuple(results))

    async def _route_protective(self, intent: OrderIntent) -> tuple[OrderResult, ...]:
        """One protective intent (a level arm/tighten): place or REPLACE the pair.

        The mixin decides from the working read: a resting pair is REPLACEd
        (cancel both, confirm absent, submit new); nothing resting is a fresh
        arm. The refs anchor on the LOT's own time (the intent's ``decision_ts``
        is the lot's open time for a level update). Any failure is a single
        ``UNPROTECTED`` row — never a silent naked lot.
        """
        conid = await self._conid(intent.symbol)
        if intent.position_side is None:
            return (
                self._unprotected_row(
                    intent, f"protect {intent.symbol}: child carries no lot side"
                ),
            )
        placed = await self._replace_bracket(intent.symbol, (intent,), conid=conid)
        if isinstance(placed, Err):
            error = cast("FeedError", placed.error)
            return (self._unprotected_row(intent, error.message, kind=error.kind),)
        return tuple(placed.value)

    def _unprotected_row(
        self,
        intent: OrderIntent,
        message: str,
        *,
        kind: str = "unprotected",
    ) -> OrderResult:
        """Log a failed protective leg to stderr and shape its UNPROTECTED row.

        The ONE place a naked lot becomes a report row: the diagnostic goes out
        immediately and the outcome is ``UNPROTECTED`` (in the unsafe set, so the
        cycle exits 3) — never a silent naked lot. *kind* may be a transport
        kind, which still maps to an unsafe outcome.
        """
        self._log(f"UNPROTECTED {intent.symbol}: {message}")
        return _failed(intent, message, kind=kind)

    async def _protect_after_fill(
        self,
        intent: OrderIntent,
        result: OrderResult,
        in_cycle_delta: float,
    ) -> tuple[OrderResult, ...]:
        """Place the OCA pair for a confirmed entry fill; a failure is NAKED.

        Spec §4.1: the pair is placed INSIDE the entry cycle right after the
        entry is confirmed filled. The refs anchor on the fill's own timestamp
        (the lot's open/fill time — never the decision bar or the clock). If the
        bracket cannot be proven resting, the lot is NAKED: report UNPROTECTED
        (exit 3, F2) and close it at market rather than leave it exposed.
        """
        if intent.stop_loss is None and intent.take_profit is None:
            return ()
        fill = result.fill
        if fill is None:
            return ()
        conid = await self._conid(intent.symbol)
        placed = await self._place_bracket(
            intent,
            filled_qty=result.filled_qty
            if result.filled_qty is not None
            else fill.filled_qty,
            conid=conid,
            fill_ts=fill.timestamp,
        )
        if isinstance(placed, Err):
            error = cast("FeedError", placed.error)
            unprotected = self._unprotected_row(intent, error.message)
        else:
            return tuple(placed.value)
        # The pair may still have RESTED (an unparseable confirm reply is not
        # proof it did not). Cancel-confirm it BEFORE the market close so a live
        # reduce order cannot race the close and over-reduce (invariant 6). An
        # unconfirmable cancel refuses the close, never risks double-cover.
        refused = await self._cancel_pair_before_close(intent)
        if refused is not None:
            return (unprotected, *refused)
        closed = await self._synthetic_close(
            intent,
            result=result,
            conid=conid,
            in_cycle_delta=in_cycle_delta,
        )
        if isinstance(closed, Err):
            close_error = cast("FeedError", closed.error)
            self._log(
                f"synthetic close of naked {intent.symbol} failed: "
                f"{close_error.message}"
            )
            return (unprotected,)
        return (unprotected, closed.value)

    async def _synthetic_close(
        self,
        intent: OrderIntent,
        *,
        result: OrderResult,
        conid: int,
        in_cycle_delta: float,
    ) -> Result[OrderResult, FeedError]:
        """Close a NAKED lot at market (spec §4.2): the protective response to a
        bracket that cannot be proven resting."""
        fill = result.fill
        if fill is None:
            return Err(
                feed_error(
                    "unprotected",
                    f"synthetic close {intent.symbol}: no fill to close",
                    intent.symbol,
                )
            )
        close = OrderIntent(
            symbol=intent.symbol,
            action=ActionType.close,
            qty=result.filled_qty if result.filled_qty is not None else fill.filled_qty,
            ref_price=fill.executed_price,
            reason="synthetic close: bracket not proven resting",
            position_id=str(conid),
            decision_ts=fill.timestamp,
        )
        return await self._place(close, in_cycle_delta=in_cycle_delta)

    async def _cancel_pair_before_close(
        self, intent: OrderIntent
    ) -> tuple[OrderResult, ...] | None:
        """Cancel-and-confirm every protective leg before the lot's close (inv. 6).

        ``None`` when there is nothing to cancel (or the cancel confirmed): the
        close proceeds. An unconfirmable cancel returns the UNPROTECTED rows
        that REFUSE the close — a live leg plus a close would over-reduce. Any
        leg count is cancelled, so a stop-only strategy's lone ``sl`` is not
        left leaning on a lot being closed (item 4).
        """
        working = await self._working_index()
        if isinstance(working, Err):
            return (
                _failed(
                    intent,
                    "refused close: pre-cancel working-orders read failed; "
                    "cannot prove the leg state",
                    kind="unprotected",
                ),
            )
        conid = await self._conid(intent.symbol)
        resting = resting_legs(
            tuple(working.value.values()),
            scope_tag=scope_tag(self._scope),
            conid=conid,
        )
        if not resting:
            return None
        cancelled = await self._cancel_legs(intent.symbol, resting)
        if isinstance(cancelled, Err):
            error = cast("FeedError", cancelled.error)
            self._log(f"close {intent.symbol} refused: {error.message}")
            return (
                _failed(
                    intent,
                    f"refused close: {error.message}",
                    kind="unprotected",
                ),
            )
        return None

    async def close(self) -> Result[None, FeedError]:
        """Close the HTTP client (its pool outlives nothing once the cycle ends)."""
        await self._client.aclose()
        return Ok(None)

    async def cancel_order(self, order_id: str) -> Result[None, FeedError]:
        """Cancel ONE resting order, confirmed by a follow-up working read.

        The DELETE reply alone is never the confirmation (spec §0/F8): the
        confirming evidence is the read showing the ``orderId`` absent. An
        unreadable or still-present read is ``UNPROTECTED`` (the caller must not
        resubmit or close on top of a possibly-live order).
        """
        account = await self._account_id()
        try:
            await self._client.cancel_order(account, order_id)
        except IbkrError as exc:
            return Err(
                feed_error(
                    "unprotected",
                    f"cancel {order_id} errored ({exc.kind}): {exc}",
                )
            )
        working = await self._working_index()
        if isinstance(working, Err):
            return Err(
                feed_error(
                    "unprotected",
                    f"cancel {order_id}: absence re-read failed; whether it is "
                    f"gone is unknown",
                )
            )
        if order_id in {o.order_id for o in working.value.values()}:
            return Err(
                feed_error(
                    "unprotected",
                    f"cancel {order_id}: still on the working read; cannot prove "
                    f"it is gone",
                )
            )
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
                return _WaitOutcome(Err(cast("FeedError", fetched.error)))
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

    async def _account_id(self) -> str:
        """The account id any POST/DELETE targets: explicit, then session, then read.

        Never a guess: an account the client cannot name is resolved from the
        gateway's own account list (an ``auth`` failure, not a default).
        """
        return (
            self._account
            or self._client.account
            or await self._client.resolve_account()
        )

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

        Every refusal here is a **book divergence** (``kind="divergence"``), not
        a broker ``rejected``: our book holds a lot the account does not, or the
        close qty exceeds the account net. ``rejected`` is excluded from the
        unsafe set (a genuine refused open is re-minted next cycle), so reporting
        it here would let a strategy that CANNOT EXIT exit 0 — the open-side
        equivalent of this same disagreement exits 3. The reducing side must be
        symmetric, so it surfaces as ``DIVERGENCE``.
        """
        net = await self._account_net(conid)
        if net is None:
            return feed_error(
                "divergence",
                f"refused close {intent.symbol} (conid {conid}): account net "
                f"unreadable, cannot prove there is anything to reduce",
                intent.symbol,
            )
        if abs(net) <= _FLAT_EPS:
            return feed_error(
                "divergence",
                f"refused close {intent.symbol} (conid {conid}): account is "
                f"flat, nothing to reduce",
                intent.symbol,
            )
        reducing = net > 0 if side is OrderSide.SELL else net < 0
        if not reducing:
            return feed_error(
                "divergence",
                f"refused close {intent.symbol} (conid {conid}): account net "
                f"{net:g} is on the opposite side to the {side.value} close, which "
                f"would open a position",
                intent.symbol,
            )
        whole = whole_quantity(intent)
        if whole > abs(net) + _FLAT_EPS:
            return feed_error(
                "divergence",
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

    # -- internals: lost-predecessor sweep --------------------------------
    #
    # Before minting a NEW order the executions window is swept for a fill
    # attributable to this key. The ref is pure, so a fill from ANY past send
    # maps to the same key; finding one means a predecessor filled and a submit
    # would be a duplicate. The sweep runs at MINT time for EVERY key with no
    # working order — there is no local record to vouch for a predecessor, so
    # the executions read is the only proof a duplicate cannot exist. A failed
    # executions read blocks the POST (fail closed, INV-1): a duplicate order is
    # unbounded exposure, a skipped cycle is not.

    async def _guard_lost_predecessor(
        self, intent: OrderIntent, key: IntentKey
    ) -> Result[OrderResult, FeedError] | None:
        """Refuse to mint when the executions window proves the key already filled.

        ``None`` means no evidence of a predecessor and the caller may mint; an
        ``Err`` (a failed executions read) fails closed so no POST follows.
        """
        swept = await self._sweep_key_fill(key)
        if isinstance(swept, Err):
            return Err(cast("FeedError", swept.error))
        execution = swept.value
        if execution is None:
            return None
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
        """One execution already attributable to *key*, or ``None`` (fail-closed read).

        A fill matches when its ref carries this scope's tag AND the key's token
        — the pure ref of this key (a legacy attempt tail, should one surface in
        the trades window, still carries the same tag+token, so it is caught
        too).
        """
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
                    if ref_is_ours(self._scope, e.order_ref)
                    and e.order_ref.rsplit("-", 1)[-1] == key.token()
                ),
                None,
            )
        )

    # -- internals: adopt / submit paths ----------------------------------

    async def _adopt(
        self,
        intent: OrderIntent,
        key: IntentKey,
        side: OrderSide,
        conid: int,
        found: WorkingOrder,
    ) -> Result[OrderResult, FeedError]:
        """Wait on a working order the pre-flight read already adopted (no submit).

        The order already exists at the broker — there is nothing to POST, only
        its real status to wait on. A missing working order here means the read
        changed under us: whether it is live is unknown, reported ``unresolved``.
        """
        ticket = build_ticket(intent, conid=conid, side=side, order_ref=found.order_ref)
        result = await self.wait_filled(found.order_id, intent, ticket)
        return self._settle_wait(result, found.order_id, adopted=True)

    async def _submit_new(
        self,
        account: str,
        intent: OrderIntent,
        key: IntentKey,
        side: OrderSide,
        conid: int,
    ) -> Result[OrderResult, FeedError]:
        """Build the ticket with the PURE ref, submit, and settle the wait.

        The same key always mints the same ref, so if the submit lands twice the
        broker's own ``cOID`` dedupe absorbs the second send — and the pre-flight
        reads (working orders + executions sweep) already proved no predecessor
        exists before this POST.
        """
        try:
            ticket = build_ticket(
                intent, conid=conid, side=side, order_ref=order_ref(key)
            )
        except OrderMappingError as exc:
            return Err(feed_error("rejected", f"{intent.symbol}: {exc}", intent.symbol))
        if ticket.rounded:
            self._log(
                f"rounded {intent.symbol} qty {intent.qty:g} -> "
                f"{ticket.body['quantity']:g} (whole shares)"
            )
        submitted = await self._submit(account, ticket, key)
        if isinstance(submitted, Err):
            return Err(cast("FeedError", submitted.error))
        order_id = submitted.value
        self._log(
            f"submitted {intent.action.value} {intent.symbol} qty={intent.qty:g} "
            f"cOID={ticket.order_ref} order_id={order_id}"
        )
        result = await self.wait_filled(order_id, intent, ticket)
        return self._settle_wait(result, order_id, adopted=False)

    async def _submit(
        self, account: str, ticket: Ticket, key: IntentKey
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
                ticket, f"submit {ticket.order_ref} errored ({exc.kind}): {exc}"
            )
        return await self._reply_loop(response, ticket, key)

    async def _reply_loop(
        self, response: object, ticket: Ticket, key: IntentKey
    ) -> Result[str, FeedError]:
        """Classify the submission response, confirming ordinary replies only.

        A confirm reply from the broker outside trading hours is treated as a
        rejection unless ``allow_oth`` is set: the broker prompts for
        confirmation when an order would trade outside regular market hours,
        and the default (no flag) refuses to auto-confirm.
        """
        confirmations = 0
        while True:
            outcome = classify_reply(response)
            if outcome.kind != "confirm":
                return await self._settle_reply(outcome, ticket, confirmations)
            # The broker returned a confirm reply (e.g. OTH prompt). If the
            # operator did not opt in, refuse to auto-confirm — the order is
            # rejected rather than placed behind their back.
            if not self._allow_oth:
                return Err(
                    FeedError(
                        kind="rejected",
                        message=(
                            f"order {ticket.order_ref} not placed: broker requires "
                            f"confirmation (likely outside trading hours); "
                            f"set --allow-oth to auto-confirm: {outcome.message}"
                        ),
                    )
                )
            confirmations += 1
            if confirmations > MAX_REPLIES:
                return await self._settle_ambiguous(
                    ticket,
                    f"still asking for confirmation after {MAX_REPLIES} replies; "
                    f"last message: {outcome.message}",
                )
            # INV-1 before the confirm POST: a SUCCESSFUL read must show no
            # working order for our key (adopt one if it appears; never POST on a
            # failed read).
            guard = await self._working_index()
            if isinstance(guard, Err):
                return self._ambiguous_error(
                    f"confirm {ticket.order_ref}: pre-confirm working-orders read failed",
                    read_ok=False,
                )
            already = match_working(
                tuple(guard.value.values()), ref_prefix(key), prefer=ticket.order_ref
            )
            if already is not None:
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
                    ticket, f"confirm {ticket.order_ref} errored ({exc.kind}): {exc}"
                )

    async def _settle_reply(
        self,
        outcome: ReplyOutcome,
        ticket: Ticket,
        confirmations: int,
    ) -> Result[str, FeedError]:
        """A terminal submission reply: a returned id is closure; a plain abort rejects.

        A refusal BEFORE any confirmation cannot have been submitted, so it is a
        genuine rejection. A refusal AFTER a confirmation goes through
        ``_settle_ambiguous`` — the confirm may have submitted the order. There is
        no durable row to write: the broker's working-orders read settles what
        was placed.
        """
        if outcome.kind == "success":
            return Ok(outcome.order_id)
        if confirmations == 0:
            return Err(
                FeedError(
                    kind="rejected",
                    message=f"order {ticket.order_ref} not placed: {outcome.message}",
                )
            )
        return await self._settle_ambiguous(
            ticket, f"aborted after {confirmations} confirmations: {outcome.message}"
        )

    async def _settle_ambiguous(
        self, ticket: Ticket, reason: str
    ) -> Result[str, FeedError]:
        """Settle an order that MAY have been submitted: adopt, or report UNRESOLVED.

        Shared by the ambiguous-submit, ambiguous-confirm and reply-abort/overflow
        paths. Every one of them sent a POST that can submit the order, so absence
        from a SUCCESSFUL read means the state is UNKNOWN — reported ``unresolved``
        and resolved by the NEXT cycle's read, never as "not placed". A failed
        read is likewise ``unresolved``, never a rejection.
        """
        index = await self._working_index()
        if isinstance(index, Err):
            return self._ambiguous_error(reason, read_ok=False)
        # The ticket's ref IS the key's pure ref — the exact prefix to match.
        found = match_working(
            tuple(index.value.values()), ticket.order_ref, prefer=ticket.order_ref
        )
        if found is not None:
            self._log(f"{reason}; adopting working order {found.order_id}")
            return Ok(found.order_id)
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
        outcome: _WaitOutcome,
        order_id: str | None,
        *,
        adopted: bool,
    ) -> Result[OrderResult, FeedError]:
        """Pass the wait branch's result on; an adopted order keeps its outcome.

        There is no durable state to persist — the fill (or the typed failure)
        IS the cycle's record, and the next cycle's reads resolve the order from
        here on.
        """
        result = outcome.result
        if isinstance(result, Err):
            return Err(cast("FeedError", result.error))
        order = result.value
        return Ok(replace(order, outcome=OrderOutcome.ADOPTED) if adopted else order)

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
        """No terminal status at the deadline -> a timeout, reported as unsure.

        A partial fill here lives in the MESSAGE, never claimed as a known state:
        the order is still working, so the next cycle's read — not this row —
        decides whether it ever filled.
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
        # it is neither a broker rejection nor an unresolved order. The close
        # guard reports this kind from the reducing side too (D3).
        "divergence": OrderOutcome.DIVERGENCE,
        # A STRUCTURAL drop (no shared cash bound / scaled to 0 shares): distinct
        # from a genuine cash-exhaustion ``rejected``, which is re-minted next
        # cycle and stays safe.
        "unfunded": OrderOutcome.UNFUNDED,
        # A protective bracket not proven resting: a refused/unconfirmed child or
        # a levelled lot with no submitted pair. NAKED, and UNSAFE (F2) — never
        # the safe ``rejected`` reading.
        "unprotected": OrderOutcome.UNPROTECTED,
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


def _open_lots(book: PortfolioState) -> tuple[Position, ...]:
    """Every OPEN lot the book carries (a flat/negligible lot is not a target)."""
    return tuple(
        lot
        for lots in book.positions.values()
        for lot in lots
        if lot.qty > FLAT_QTY_EPS
    )


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


__all__ = ["DEFAULT_POLL_INTERVAL_S", "DEFAULT_TIMEOUT_S", "MAX_REPLIES", "IbkrBroker"]
