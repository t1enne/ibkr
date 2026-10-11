"""What one live cycle reports — the shape the CLI renders and exits on.

Pure value types and predicates over them: no I/O, no pipeline import. The cycle
(``src.live.pipeline``) builds a :class:`CycleReport`; the CLI renders it and
derives its exit code from :meth:`CycleReport.is_unsafe`. Keeping the unsafe
predicate and the report in ONE module means the exit code and the printed
document cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.bt.state import PortfolioState
from src.live.divergence import Divergence
from src.live.identity import OrderOutcome
from src.live.pure import OrderResult
from src.live.types import (
    MODELLED_COST,
    CostProvenance,
    FeedError,
    LiveSignal,
    OrderIntent,
)


@dataclass(frozen=True)
class CycleReport:
    """The full outcome of one cycle: what we saw, decided, placed, book start."""

    as_of: pd.Timestamp
    signals: tuple[LiveSignal, ...]
    intents: tuple[OrderIntent, ...]
    results: tuple[OrderResult, ...]
    portfolio_before: PortfolioState
    #: Which source produced this run's cost figures. Defaults to the all-modelled
    #: sim provenance; the CLI sets the broker-exact book for IBKR.
    cost: CostProvenance = MODELLED_COST
    #: A cohort-level placement failure: the broker refused the WHOLE cohort before
    #: any per-order result existed, so ``results`` may be empty while the orders'
    #: true state is unknown. ``None`` when placement produced per-order results.
    placement_error: FeedError | None = None
    #: A failed cycle-start ``resync`` (an unreadable open-orders feed). Without
    #: this, a HOLD cycle with no intents reports a clean "0 orders" while every
    #: OPEN intent was never re-checked. ``None`` when resync succeeded.
    resync_error: FeedError | None = None
    #: Every lot our fill fold and the account book disagree on. Our book is
    #: ``book_from_executions`` (append-only, engine-owned); the account's is what
    #: the adapter READ. A manual edit, an untracked entry or a lot we booked that
    #: the account lacks lands here — the operator resolves it, the engine never
    #: silently re-sizes onto the account's number.
    divergences: tuple[Divergence, ...] = ()

    def is_unsafe(self) -> bool:
        """Whether this cycle failed to safely do its job (the exit-code predicate).

        The ONE definition of "unsafe", so the CLI's exit code cannot drift from
        the report it prints. A cycle is unsafe when placement or resync reported
        a cohort-level error, when the two books disagree on any lot, or when any
        order result is in a state that leaves the order's true disposition
        unknown or stuck (see :data:`_UNSAFE_OUTCOMES`). A clean cycle
        (``REJECTED``/``UNFILLED`` are terminal and honest, a ``PLACED``/``ADOPTED``
        fill is settled) is safe.
        """
        if self.placement_error is not None or self.resync_error is not None:
            return True
        if self.divergences:
            return True
        return any(result.outcome in _UNSAFE_OUTCOMES for result in self.results)


#: The order outcomes that mean the cycle did not safely reach a known state:
#: ``UNRESOLVED`` (a submit/confirm left "is it live?" unknown), ``WEDGED`` (an
#: OPEN record nothing could settle across ``WEDGED_CYCLES`` resyncs), ``TIMEOUT``
#: (still working at the deadline, not proven terminal), ``DIVERGENCE`` (a book
#: disagreement refused), ``UNFUNDED`` (a STRUCTURAL cohort drop that recurs every
#: cycle forever), ``UNPROTECTED`` (a protective bracket is not proven resting;
#: the lot is NAKED, and a repeated cycle repeats the exit-3 alarm). A
#: ``REJECTED`` open is genuine cash exhaustion for this bar and re-mints next
#: cycle, so it stays safe.
_UNSAFE_OUTCOMES: frozenset[OrderOutcome] = frozenset(
    {
        OrderOutcome.UNRESOLVED,
        OrderOutcome.WEDGED,
        OrderOutcome.TIMEOUT,
        OrderOutcome.DIVERGENCE,
        OrderOutcome.UNFUNDED,
        OrderOutcome.UNPROTECTED,
    }
)


def unsafe_outcomes(report: CycleReport) -> tuple[str, ...]:
    """What made *report* unsafe, in report order — the operator's suppression note.

    Derived from the report, never a fixed list, so the ``--allow-unsafe`` stderr
    note cannot claim something the cycle did not do: the cohort-level
    ``placement_error``/``resync_error`` kinds first (each prefixed so a kind
    cannot be mistaken for an order outcome), then every distinct unsafe order
    outcome present. Empty exactly when :meth:`CycleReport.is_unsafe` is False.
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
