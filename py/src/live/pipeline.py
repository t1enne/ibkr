"""Live cycle pipeline — one batch pass: read book, screen, reconcile, place, record.

Not a loop: the caller (CLI / cron) drives cadence. The cycle is a thin composer
over four named stages (:func:`read_portfolio`, :func:`screen_signals`,
:func:`diff_intents`, :func:`execute_cohort`) plus the durable side effects
(:func:`_lease`, :func:`_resync`, :func:`_divergences`, :func:`_persist`). All
I/O sits at the edges; ``reconcile`` and the divergence oracle stay pure and are
only WIRED here, never rewritten.

The engine depends on ONE backend seam (:class:`src.live.adapter.LiveAdapter`) and
one ledger capability set (:class:`CycleLedger`); the book travels INTO the
adapter as a parameter and the results come back OUT, so no adapter holds state
between cycles and the ledger stays the single durable book.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Protocol, cast

import pandas as pd

from src.bt.state import PortfolioState
from src.live.adapter import LiveAdapter
from src.live.divergence import Divergence, book_from_executions, guard_divergence
from src.live.freshness import StaleDataError, assert_data_fresh
from src.live.ledger import ExecutionRecord
from src.live.pure import OrderResult
from src.live.protect import (
    ProtectionTarget,
    unprotected_results,
)
from src.live.reconcile import reconcile
from src.live.report import CycleReport
from src.live.result import Err, Ok, Result
from src.live.signals import live_signals
from src.live.types import (
    FLAT_QTY_EPS,
    FeedError,
    LiveConfig,
    LiveSignal,
    OrderIntent,
    PortfolioSnapshot,
)


class PortfolioFetchError(RuntimeError):
    """The portfolio fetch failed; the cycle cannot reconcile without a book."""

    def __init__(self, error: FeedError) -> None:
        super().__init__(error.message)
        self.error = error


class SignalSource(Protocol):
    """The screen bridge: config path -> actionable signals (injectable).

    ``book`` is the live book the cycle READ; the bridge seeds the strategy run
    with it so a lot entered before the screen's short window is still visible
    to the strategy's own exit rule (see ``src.live.signals``).
    """

    def __call__(
        self,
        config_path: str,
        max_age_days: int | None = None,
        *,
        book: PortfolioState | None = None,
    ) -> tuple[LiveSignal, ...]: ...


class CycleLedger(Protocol):
    """The ledger capabilities ``run_cycle`` needs: the lease, the book, the stamp.

    ``cycle_lease(scope)`` is the PER-SCOPE concurrency guard (two adapters run
    concurrently; two cycles on one scope do not). ``executions_of`` is our
    fill-derived book, the divergence guard's "ours" side. ``owned_ids`` scopes
    a close to the lots the scope is recorded as owning. ``record_results`` is the
    ONE write point for placement outcomes, and ``touch_cycle`` is the audit
    stamp. There is no intents housekeeping: no durable open-order store exists
    to prune.
    """

    def cycle_lease(self, scope: str = "") -> AbstractContextManager[None]: ...

    def touch_cycle(self, scope: str, at: pd.Timestamp) -> None: ...

    def owned_ids(self, scope: str) -> frozenset[str]: ...

    def executions_of(self, scope: str) -> tuple[ExecutionRecord, ...]: ...

    def record_results(
        self, scope: str, results: tuple[OrderResult, ...], now: pd.Timestamp
    ) -> None: ...


async def run_cycle(
    config: LiveConfig,
    adapter: LiveAdapter,
    *,
    ledger: CycleLedger,
    strategy_id: str,
    scope: str,
    config_path: str | None = None,
    max_age_days: int = 5,
    dry_run: bool = False,
    db_path: str | Path | None = None,
    now: pd.Timestamp | None = None,
    signal_source: SignalSource = live_signals,
    allow_oth: bool = False,
) -> CycleReport:
    """One full batch pass. Not a loop; the caller drives cadence.

    A non-dry run holds the SCOPE's exclusive cycle lease for the whole pass, so
    an overlapping cycle on the same scope refuses to start
    (``CycleInProgressError``) instead of both placing off the same pre-order
    book; another scope's lease is unaffected, so two adapters run concurrently.

    ``strategy_id`` is the config-hash AUDIT key; ``scope`` is the stable
    OWNERSHIP key the book is keyed by and the lease is taken for.
    ``dry_run=True`` takes no lease and writes nothing at all, so the next cycle
    recomputes the same intents (see :func:`_lease`/:func:`_persist`).
    ``allow_oth`` when true, the broker is allowed to auto-confirm BUY/SELL MKT
    orders placed outside trading hours. Default (false) rejects them.
    """
    now_ts = now if now is not None else pd.Timestamp.now(tz="UTC")
    with _lease(ledger, scope, dry_run):
        # Cheapest gate first: refuse a stale universe before paying for the
        # broker read (pure DB check, no I/O on the adapter).
        assert_data_fresh(config.symbols, max_age_days, now_ts, db_path)
        pf = await read_portfolio(adapter)
        signals = screen_signals(config_path, max_age_days, signal_source, pf.portfolio)
        # The backstop attests the state as READ: it runs BEFORE the resync, whose
        # own protective action (a sim stop firing) legitimately retires a lot
        # within this cycle and would otherwise look like a lot left naked.
        attested = await _attest(adapter, pf.portfolio, dry_run)
        resync_results, resync_error = await _resync(adapter, dry_run)
        divergences = _divergences(ledger, scope, pf.portfolio)
        owned = ledger.owned_ids(scope) if adapter.owns_book else None
        intents = diff_intents(signals, pf.portfolio, config, owned)
        placed = await execute_cohort(adapter, pf.portfolio, intents, dry_run)
        results, placement_error = _merge(resync_results + attested, placed)
        if not dry_run:
            _persist(ledger, adapter, scope, results, strategy_id, now_ts)
        await adapter.close()
    return CycleReport(
        as_of=now_ts,
        signals=signals,
        intents=intents,
        results=results,
        portfolio_before=pf.portfolio,
        placement_error=placement_error,
        resync_error=resync_error,
        divergences=divergences,
        warnings=pf.warnings,
    )


async def read_portfolio(adapter: LiveAdapter) -> PortfolioSnapshot:
    """The adapter's book read; an ``Err`` is fatal (no book, no reconcile)."""
    fetched = await adapter.read_book()
    if isinstance(fetched, Err):
        raise PortfolioFetchError(cast("FeedError", fetched.error))
    return fetched.value


def screen_signals(
    config_path: str | None,
    max_age_days: int,
    source: SignalSource,
    book: PortfolioState,
) -> tuple[LiveSignal, ...]:
    """The screen's actionable signals for *config_path* (the config path is required).

    *book* is the book the cycle just read — the screen seeds the strategy run
    with it (never a re-derivation), so the strategy's exit rules see the lots
    live actually holds.
    """
    assert config_path is not None, (
        "run_cycle requires config_path for the screen bridge"
    )
    return source(config_path, max_age_days, book=book)


def diff_intents(
    signals: tuple[LiveSignal, ...],
    book: PortfolioState,
    config: LiveConfig,
    owned: frozenset[str] | None,
) -> tuple[OrderIntent, ...]:
    """Reconcile the signals against the pre-order book into intents (pure)."""
    return reconcile(signals, book, config, owned)


async def execute_cohort(
    adapter: LiveAdapter,
    book: PortfolioState,
    intents: tuple[OrderIntent, ...],
    dry_run: bool,
) -> Result[tuple[OrderResult, ...], FeedError]:
    """Place *intents* as ONE cohort; a dry run or an empty cohort places nothing."""
    if dry_run or not intents:
        return Ok(())
    return await adapter.place_cohort(book, intents)


def _persist(
    ledger: CycleLedger,
    adapter: LiveAdapter,
    scope: str,
    results: tuple[OrderResult, ...],
    strategy_id: str,
    now_ts: pd.Timestamp,
) -> None:
    """The cycle's durable writes, in order: results, audit stamp.

    ``results`` is the CYCLE's full result set (resync-fired fills first, then
    placement), so a sim broker's fired stop is recorded against our fold too —
    on the sim path the account book and ``live_execution`` must advance
    together or the next cycle's divergence oracle would scream at a stop it
    never saw booked. Skipped entirely on a dry run (the caller never reaches
    here), so a read-only cycle leaves the store byte-identical. No open-order
    housekeeping: the broker's reads are the only open-state owner.
    """
    _record(ledger, adapter, scope, results, now_ts)
    ledger.touch_cycle(scope, now_ts)


async def _attest(
    adapter: LiveAdapter, book: PortfolioState, dry_run: bool
) -> tuple[OrderResult, ...]:
    """The per-cycle naked-lot backstop: attest protection, report, never place.

    Spec §4.2 "missing ⇒ naked, loudly". Runs on the book the cycle READ, BEFORE
    the cycle-start resync: a lot a resync's own protective action retires (a sim
    stop firing) is protection WORKING, not a naked lot. A bracket lost after its
    entry cycle (rejected, cancelled externally, a fired-partial) has no such
    in-cycle explanation, so it still reaches the unsafe set and exits 3 instead
    of passing a HOLD cycle as a clean exit 0. The pass only READS: an adapter
    that cannot prove a lot is protected reports ``UNPROTECTED`` for it —
    ambiguity is never clean, and no POST follows a failed read (invariant 1).

    A cohort-level ``Err`` is reported the same way the resync's is: non-fatal to
    the cycle, but it must reach the report rather than vanish, so a backend that
    cannot be read never reads as "all lots protected".
    """
    if dry_run:
        return ()
    attested = await adapter.attest_protection(book)
    if isinstance(attested, Ok):
        return cast("tuple[OrderResult, ...]", attested.value)
    error = attested.error
    return unprotected_results(
        _targets_of(book),
        reason=f"protection could not be attested ({error.kind}): {error.message}",
    )


def _targets_of(book: PortfolioState) -> tuple[ProtectionTarget, ...]:
    """Every OPEN level-carrying lot of *book* as an attestation target.

    ``position_id`` is the wire handle on BOTH edges (the conid on ibkr, the lot
    id on sim), so the report row can name the lot without a second lookup. Only
    a lot whose advisory level says it SHOULD rest a bracket is a target: a lot
    opened without a stop is unprotected by design, never "unprotected".
    """
    return tuple(
        ProtectionTarget(
            conid=int(lot.position_id) if str(lot.position_id).isdigit() else 0,
            symbol=lot.symbol,
            position_id=str(lot.position_id),
            qty=lot.qty,
        )
        for lots in book.positions.values()
        for lot in lots
        if lot.qty > FLAT_QTY_EPS
        and lot.position_id
        and (lot.stop_loss is not None or lot.take_profit is not None)
    )


def _lease(
    ledger: CycleLedger, scope: str, dry_run: bool
) -> AbstractContextManager[None]:
    """The cycle's concurrency guard: the scope's lease, or nothing on a dry run.

    A read-only run takes NO lease, so a diagnostic ``--dry-run`` can never block
    a live cycle (and two dry runs never block each other).
    """
    return nullcontext() if dry_run else ledger.cycle_lease(scope)


async def _resync(
    adapter: LiveAdapter, dry_run: bool
) -> tuple[tuple[OrderResult, ...], FeedError | None]:
    """Adopt/mark OPEN intents at cycle start; a failed read is reported, not swallowed.

    A failure here is non-fatal — placement's own pre-flight fails closed with the
    same read failure — but it must reach the REPORT: otherwise a HOLD cycle with
    no intents looks like a clean "0 orders" while no OPEN intent was ever
    re-checked. Skipped on a dry run, which must not persist anything.
    """
    if dry_run:
        return (), None
    resynced = await adapter.resync()
    if isinstance(resynced, Err):
        return (), cast("FeedError", resynced.error)
    return tuple(resynced.value), None


def _divergences(
    ledger: CycleLedger, scope: str, account: PortfolioState
) -> tuple[Divergence, ...]:
    """Our fill-derived book vs the account book, compared by the shared oracle.

    On the sim path ``account`` is the operator-editable JSON book, so this
    catches a human edit. On the IBKR path the adapter BUILDS ``account`` from
    the same broker trade window this fold is fed, so the two agree by
    construction and a hit here means sqlite persistence drifted from the
    broker stream — a persistence-integrity check, not a broker-side edit.
    """
    return guard_divergence(book_from_executions(ledger.executions_of(scope)), account)


def _record(
    ledger: CycleLedger,
    adapter: LiveAdapter,
    scope: str,
    results: tuple[OrderResult, ...],
    now_ts: pd.Timestamp,
) -> None:
    """Record the cycle's result set — only when the scope owns the book.

    An IBKR scope's book advances from the broker's own execution stream (the
    broker persists it), so recording placement results there would double-book;
    the sim account book has no such stream, so its results ARE the book's next
    state. A cohort-level placement error is handled by the CALLER: the merged
    ``results`` then carries only the resync-fired fills (each a known fill),
    never an unknown placement row.
    """
    if not adapter.owns_book:
        return
    ledger.record_results(scope, results, now_ts)


def _merge(
    resync_results: tuple[OrderResult, ...],
    placed: Result[tuple[OrderResult, ...], FeedError],
) -> tuple[tuple[OrderResult, ...], FeedError | None]:
    """Cycle results: adopted/marked OPEN intents first, then this cycle's placements.

    A cohort-level placement ``Err`` yields the resync results alone plus the
    error: the orders' true state is unknown, so inventing per-order rows would
    read as settled.
    """
    if isinstance(placed, Err):
        return resync_results, cast("FeedError", placed.error)
    return resync_results + tuple(placed.value), None


__all__ = [
    "CycleLedger",
    "PortfolioFetchError",
    "SignalSource",
    "StaleDataError",
    "assert_data_fresh",
    "diff_intents",
    "execute_cohort",
    "read_portfolio",
    "run_cycle",
    "screen_signals",
]
