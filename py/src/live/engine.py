"""Live cycle orchestration — one batch pass: read book → screen → reconcile → place → record.

Not a loop. The caller (CLI / cron) drives cadence. All I/O sits at the edges
(the ``LiveAdapter`` seam, the screen bridge); ``reconcile`` and ``build_report``
are pure. A cycle fails loudly on a stale feed or an unreadable book rather than
trading yesterday's intent.

The engine depends on ONE backend seam (:class:`src.live.adapter.LiveAdapter`) and
one ledger capability set (:class:`CycleLedger`); the book travels INTO the
adapter as a parameter and the results come back OUT, so no adapter holds state
between cycles and the ledger stays the single durable book.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import pandas as pd

from src.bt.state import PortfolioState
from src.data.db import get_connection
from src.live.adapter import LiveAdapter
from src.live.pure import OrderResult
from src.live.divergence import Divergence, book_from_executions, guard_divergence
from src.live.identity import OrderOutcome
from src.live.ledger import ExecutionRecord
from src.live.reconcile import reconcile
from src.live.result import Err, Ok, Result
from src.live.signals import live_signals
from src.live.types import (
    MODELLED_COST,
    CostProvenance,
    FeedError,
    LiveConfig,
    LiveSignal,
    OrderIntent,
    PortfolioSnapshot,
)


@dataclass(frozen=True)
class CycleReport:
    """The full outcome of one cycle: what we saw, decided, placed, book start."""

    as_of: pd.Timestamp
    signals: tuple[LiveSignal, ...]
    intents: tuple[OrderIntent, ...]
    results: tuple[OrderResult, ...]
    portfolio_before: PortfolioState
    #: Which source produced this run's cost figures (plan §7.3). Defaults to the
    #: all-modelled sim provenance; the CLI sets the broker-exact book for IBKR.
    cost: CostProvenance = MODELLED_COST
    #: A cohort-level placement failure: the broker refused the WHOLE cohort before
    #: any per-order result existed, so ``results`` may be empty while the orders'
    #: true state is unknown. ``None`` when placement produced per-order results.
    placement_error: FeedError | None = None
    #: A failed cycle-start ``resync`` (an unreadable open-orders feed). D4: without
    #: this, a HOLD cycle with no intents reports a clean "0 orders" while every
    #: OPEN intent was never re-checked. ``None`` when resync succeeded.
    resync_error: FeedError | None = None
    #: Every lot our fill fold and the account book disagree on (plan §4.3). Our
    #: book is ``book_from_executions`` (append-only, engine-owned); the account's
    #: is what the adapter READ. A manual edit, an untracked entry or a lot we
    #: booked that the account lacks lands here — the operator resolves it, the
    #: engine never silently re-sizes onto the account's number.
    divergences: tuple[Divergence, ...] = ()

    def is_unsafe(self) -> bool:
        """Whether this cycle failed to safely do its job (the exit-code predicate).

        The ONE definition of "unsafe", so the CLI's exit code cannot drift from
        the report it prints. A cycle is unsafe when placement or resync reported
        a cohort-level error, when the two books disagree on any lot, or when any
        order result is in a state that leaves the order's true disposition
        unknown or stuck:

        - ``UNRESOLVED`` — a submit/confirm left "is it live?" unknown;
        - ``WEDGED`` — an OPEN record nothing could settle across ``WEDGED_CYCLES``
          resyncs, which needs an operator;
        - ``TIMEOUT`` — still working at the deadline; not proven terminal;
        - ``DIVERGENCE`` — an OPEN (or a CLOSE the account net cannot absorb —
          the same book disagreement seen from the reducing side) refused; the
          account is in a state we cannot explain, so an operator must look even
          though nothing was placed;
        - ``UNFUNDED`` — a STRUCTURAL drop: a cohort that cannot state a shared
          cash bound refuses EVERY open, and a scaled qty that floors to 0 shares
          drops one. Unlike a ``REJECTED`` open (genuine cash exhaustion for this
          bar, re-minted next cycle) this recurs every cycle forever.

        A clean cycle (``REJECTED``/``UNFILLED`` are terminal and honest, a
        ``PLACED``/``ADOPTED`` fill is settled) is safe.
        """
        if self.placement_error is not None or self.resync_error is not None:
            return True
        if self.divergences:
            return True
        return any(result.outcome in _UNSAFE_OUTCOMES for result in self.results)


#: The order outcomes that mean the cycle did not safely reach a known state.
_UNSAFE_OUTCOMES: frozenset[OrderOutcome] = frozenset(
    {
        OrderOutcome.UNRESOLVED,
        OrderOutcome.WEDGED,
        OrderOutcome.TIMEOUT,
        OrderOutcome.DIVERGENCE,
        OrderOutcome.UNFUNDED,
    }
)


def unsafe_outcomes(report: CycleReport) -> tuple[str, ...]:
    """What made *report* unsafe, in report order — the operator's suppression note.

    Derived from the report, never a fixed list, so the D2 stderr note cannot
    claim something the cycle did not do: the cohort-level ``placement_error``/
    ``resync_error`` kinds first (each prefixed so a kind cannot be mistaken for
    an order outcome), then every distinct unsafe order outcome present (a
    refused close and a structural open drop both surface here as their kind's
    outcome). Empty exactly when :meth:`CycleReport.is_unsafe` is False.
    """
    named: list[str] = []
    if report.placement_error is not None:
        named.append(f"placement_error:{report.placement_error.kind}")
    if report.resync_error is not None:
        named.append(f"resync_error:{report.resync_error.kind}")
    for result in report.results:
        if result.outcome in _UNSAFE_OUTCOMES and result.outcome.value not in named:
            named.append(result.outcome.value)
    return tuple(named)


#: A closed intent record older than this is housekeeping noise; OPEN records
#: are never pruned (they still own state).
_INTENT_RETENTION_DAYS = 90


class StaleDataError(RuntimeError):
    """The universe's newest bar is too old — refuse to trade on a stale tail."""

    def __init__(self, error: FeedError) -> None:
        super().__init__(error.message)
        self.error = error


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
    fill-derived book, the divergence guard's "ours" side. ``sim_open_ids`` scopes
    a close to the lots the scope is recorded as owning. ``record_results`` is the
    ONE write point for placement outcomes, and ``touch_cycle``/``prune`` are the
    audit + housekeeping writes.
    """

    def cycle_lease(self, scope: str = "") -> AbstractContextManager[None]: ...

    def touch_cycle(self, strategy_id: str, at: pd.Timestamp) -> None: ...

    def sim_open_ids(self, scope: str) -> frozenset[str]: ...

    def executions_of(self, scope: str) -> tuple[ExecutionRecord, ...]: ...

    def record_results(
        self, scope: str, results: tuple[OrderResult, ...], now: pd.Timestamp
    ) -> None: ...

    def prune(self, before: pd.Timestamp) -> int: ...


def build_report(
    portfolio: PortfolioState,
    signals: tuple[LiveSignal, ...],
    intents: tuple[OrderIntent, ...],
    results: tuple[OrderResult, ...],
    as_of: pd.Timestamp,
    cost: CostProvenance = MODELLED_COST,
    placement_error: FeedError | None = None,
    resync_error: FeedError | None = None,
    divergences: tuple[Divergence, ...] = (),
) -> CycleReport:
    """Pure: assemble the cycle report. No clock, no I/O."""
    return CycleReport(
        as_of=as_of,
        signals=signals,
        intents=intents,
        results=results,
        portfolio_before=portfolio,
        cost=cost,
        placement_error=placement_error,
        resync_error=resync_error,
        divergences=divergences,
    )


def assert_data_fresh(
    symbols: tuple[str, ...],
    max_age_days: int,
    now: pd.Timestamp,
    db_path: str | Path | None = None,
) -> None:
    """Raise ``StaleDataError`` when the universe's newest bar is too old.

    A cron ``data dl`` failure must not let the cycle trade on a stale tail.
    ``max_age_days <= 0`` disables the gate; an empty universe is skipped.
    """
    if max_age_days <= 0 or not symbols:
        return
    newest_ms = _newest_ms(symbols, db_path)
    if newest_ms is None:
        raise StaleDataError(
            FeedError(kind="stale_data", message="no data for universe")
        )
    # Compare like epochs: the DB stores ms-epoch UTC, so anchor both sides to
    # UTC. A naive local ``now`` would skew the age by the host's UTC offset.
    newest = pd.Timestamp(int(newest_ms), unit="ms", tz="UTC")
    age = _utc(now) - newest
    if age > pd.Timedelta(days=max_age_days):
        raise StaleDataError(
            FeedError(
                kind="stale_data",
                message=(
                    f"universe newest bar {newest} is {age} old (> {max_age_days}d)"
                ),
            )
        )


def _utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Normalise *ts* to UTC: a naive timestamp is taken as UTC, not local."""
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def _newest_ms(symbols: tuple[str, ...], db_path: str | Path | None) -> int | None:
    """Newest candle timestamp (ms epoch) across *symbols*, or ``None`` if none.

    ``candle.ticker`` is stored UPPERCASE, so upper-case the symbols before
    binding — a lowercase config symbol must match, not yield a spurious empty.
    """
    upper = tuple(s.upper() for s in symbols)
    placeholders = ",".join("?" * len(upper))
    con = get_connection(db_path)
    try:
        row = con.execute(
            f"SELECT MAX(timestamp) FROM candle WHERE ticker IN ({placeholders})",
            upper,
        ).fetchone()
    finally:
        con.close()
    return row[0] if row else None


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
    cost: CostProvenance = MODELLED_COST,
) -> CycleReport:
    """One full batch pass. Not a loop; the caller drives cadence.

    A non-dry run holds the SCOPE's exclusive cycle lease for the whole pass, so
    an overlapping cycle on the same scope refuses to start
    (``CycleInProgressError``) instead of both placing off the same pre-order
    book; another scope's lease is unaffected, so two adapters run concurrently.

    ``strategy_id`` is the config-hash AUDIT key (``touch_cycle``); ``scope`` is
    the stable OWNERSHIP key the book is keyed by and the lease is taken for.
    ``dry_run=True`` takes no lease and writes nothing at all, so the next cycle
    recomputes the same intents (see :func:`_lease`/:func:`_persist`).
    """
    now_ts = now if now is not None else pd.Timestamp.now(tz="UTC")
    with _lease(ledger, scope, dry_run):
        snapshot = await _read_book(adapter)
        assert_data_fresh(config.symbols, max_age_days, now_ts, db_path)
        assert config_path is not None, (
            "run_cycle requires config_path for the screen bridge"
        )
        signals = signal_source(config_path, max_age_days)
        resync_results, resync_error = await _resync(adapter, dry_run)
        divergences = _divergences(ledger, scope, snapshot.portfolio)
        placed = await _place(
            adapter, snapshot.portfolio, signals, config, ledger, scope, dry_run
        )
        results, placement_error = _merge(resync_results, placed.results)
        if not dry_run:
            _persist(ledger, adapter, scope, placed, strategy_id, now_ts)
        await adapter.close()
    return build_report(
        snapshot.portfolio,
        signals,
        placed.intents,
        results,
        now_ts,
        cost=cost,
        placement_error=placement_error,
        resync_error=resync_error,
        divergences=divergences,
    )


def _persist(
    ledger: CycleLedger,
    adapter: LiveAdapter,
    scope: str,
    placed: _Placement,
    strategy_id: str,
    now_ts: pd.Timestamp,
) -> None:
    """The cycle's durable writes, in order: book, audit stamp, housekeeping.

    Skipped entirely on a dry run (the caller never reaches here), so a read-only
    cycle leaves the store byte-identical.
    """
    _record(ledger, adapter, scope, placed, now_ts)
    ledger.touch_cycle(strategy_id, now_ts)
    ledger.prune(
        cast("pd.Timestamp", now_ts - pd.Timedelta(days=_INTENT_RETENTION_DAYS))
    )


def _lease(
    ledger: CycleLedger, scope: str, dry_run: bool
) -> AbstractContextManager[None]:
    """The cycle's concurrency guard: the scope's lease, or nothing on a dry run.

    A read-only run takes NO lease, so a diagnostic ``--dry-run`` can never block
    a live cycle (and two dry runs never block each other).
    """
    return nullcontext() if dry_run else ledger.cycle_lease(scope)


async def _read_book(adapter: LiveAdapter) -> PortfolioSnapshot:
    """The adapter's book read; an ``Err`` is fatal (no book, no reconcile)."""
    fetched = await adapter.read_book()
    if isinstance(fetched, Err):
        raise PortfolioFetchError(cast("FeedError", fetched.error))
    return fetched.value


async def _resync(
    adapter: LiveAdapter, dry_run: bool
) -> tuple[tuple[OrderResult, ...], FeedError | None]:
    """Adopt/mark OPEN intents at cycle start; a failed read is reported, not swallowed.

    A failure here is non-fatal — placement's own pre-flight fails closed with the
    same read failure — but it must reach the REPORT (D4): otherwise a HOLD cycle
    with no intents looks like a clean "0 orders" while no OPEN intent was ever
    re-checked. Skipped on a dry run, which must not persist anything.
    """
    if dry_run:
        return (), None
    resynced = await adapter.resync()
    if isinstance(resynced, Err):
        return (), cast("FeedError", resynced.error)
    return tuple(resynced.value), None


def _owned_ids(
    ledger: CycleLedger, adapter: LiveAdapter, scope: str
) -> frozenset[str] | None:
    """The lot ids a close may target, or ``None`` when every lot is closable.

    A book the adapter does not own (IBKR's replayed book: already only our lots)
    needs no filter, so ``owned=None`` — and the ledger must not blank it out. A
    book that may hold lots we never opened (the sim account book, a human-edited
    ``live_position``) is scoped to the lots the scope is recorded as owning.
    """
    return ledger.sim_open_ids(scope) if adapter.owns_book else None


def _divergences(
    ledger: CycleLedger, scope: str, account: PortfolioState
) -> tuple[Divergence, ...]:
    """Our fill-derived book vs the account book, compared by the shared oracle."""
    return guard_divergence(book_from_executions(ledger.executions_of(scope)), account)


@dataclass(frozen=True)
class _Placement:
    """One cycle's placement: the intents it decided on and the cohort's outcome."""

    intents: tuple[OrderIntent, ...]
    results: Result[tuple[OrderResult, ...], FeedError]


async def _place(
    adapter: LiveAdapter,
    book: PortfolioState,
    signals: tuple[LiveSignal, ...],
    config: LiveConfig,
    ledger: CycleLedger,
    scope: str,
    dry_run: bool,
) -> _Placement:
    """Reconcile the signals into intents, then place them as ONE cohort.

    The book is NOT written here: the adapter returns the results and the engine
    records them (:func:`_record`), or the next cycle's read advances the book
    from the broker's own executions. A cohort ``Err`` is returned AS-IS (never
    silently turned into "no orders"): the report carries it so an operator sees
    the failure rather than "0 orders". A dry run places nothing at all.
    """
    intents = reconcile(signals, book, config, _owned_ids(ledger, adapter, scope))
    if dry_run or not intents:
        return _Placement(intents=intents, results=Ok(()))
    return _Placement(
        intents=intents, results=await adapter.place_cohort(book, intents)
    )


def _record(
    ledger: CycleLedger,
    adapter: LiveAdapter,
    scope: str,
    placed: _Placement,
    now_ts: pd.Timestamp,
) -> None:
    """Record this cycle's placement results — only when the scope owns the book.

    An IBKR scope's book advances from the broker's own execution stream (the
    broker persists it), so recording placement results there would double-book;
    the sim account book has no such stream, so its results ARE the book's next
    state. A cohort-level ``Err`` records nothing: the orders' true state is
    unknown, and a recorded row would read as settled.
    """
    if not adapter.owns_book or isinstance(placed.results, Err):
        return
    ledger.record_results(scope, tuple(placed.results.value), now_ts)


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
