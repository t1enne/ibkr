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
from src.live.reconcile import reconcile
from src.live.report import CycleReport
from src.live.result import Err, Ok, Result
from src.live.signals import live_signals
from src.live.types import (
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
    """The screen bridge: config path -> actionable signals (injectable)."""

    def __call__(
        self, config_path: str, max_age_days: int | None = None
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
    """
    now_ts = now if now is not None else pd.Timestamp.now(tz="UTC")
    with _lease(ledger, scope, dry_run):
        # Cheapest gate first: refuse a stale universe before paying for the
        # broker read (pure DB check, no I/O on the adapter).
        assert_data_fresh(config.symbols, max_age_days, now_ts, db_path)
        pf = await read_portfolio(adapter)
        signals = screen_signals(config_path, max_age_days, signal_source)
        resync_results, resync_error = await _resync(adapter, dry_run)
        divergences = _divergences(ledger, scope, pf.portfolio)
        owned = ledger.owned_ids(scope) if adapter.owns_book else None
        intents = diff_intents(signals, pf.portfolio, config, owned)
        placed = await execute_cohort(adapter, pf.portfolio, intents, dry_run)
        results, placement_error = _merge(resync_results, placed)
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
) -> tuple[LiveSignal, ...]:
    """The screen's actionable signals for *config_path* (the config path is required)."""
    assert config_path is not None, (
        "run_cycle requires config_path for the screen bridge"
    )
    return source(config_path, max_age_days)


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
