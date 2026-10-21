"""The ONE adapter seam: ``LiveAdapter`` plus its name resolution and dispatch.

``PortfolioSource`` + ``LiveBroker`` were two seams describing one thing — the
backend a cycle trades through. They collapse here. An adapter holds NO state
across cycles: the fetched book travels as a PARAMETER into
``place``/``place_cohort`` and the settled book leaves as the RETURNED results,
so nothing mutable outlives one cycle. The durable book is sqlite
(``live_execution`` + ``live_strategy``).

Two honest qualifications, both load-bearing:

- ``read_book`` is not a pure read (see ``LiveAdapter.read_book``): an adapter
  whose book is derived from the backend's execution stream persists the
  advanced rows during the read.
- The ibkr broker keeps a per-cycle ``_book``/``_conid_cache`` seeded from the
  fetched book. It is correct because the instance is built per cycle, but
  "stateless" describes the NET effect, not the object — a long-lived instance
  must not be reused across cycles.

``run_cycle`` depends only on this Protocol; the concrete adapters live in
``adapters/sim`` and ``adapters/ibkr`` and are reached through
:func:`resolve_adapter`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from src.bt.state import PortfolioState
from src.data.ibkr.client import IbkrClient
from src.live.adapters.ibkr.adapter import build_ibkr_adapter
from src.live.adapters.sim.adapter import build_sim_adapter
from src.live.pure import OrderResult
from src.live.ledger import SqliteLedger
from src.live.result import Result
from src.live.scope import AdapterName
from src.live.types import FeedError, LiveConfig, OrderIntent, PortfolioSnapshot


class LiveAdapter(Protocol):
    """The backend one cycle trades through: read the book, place, reconcile.

    ``scope`` is the ownership key this adapter's ledger rows are keyed by;
    ``owns_book`` says how a close is scoped (``True``: only lots the scope is
    recorded as owning are closable; ``False``: the book ALREADY holds only this
    scope's lots, so every lot in it is ours).

    ``read_book`` builds a fresh snapshot per call — never a cache. It is the
    cycle's BOOK ADVANCE, not a pure read: an adapter whose book is DERIVED from
    the backend's own execution stream (ibkr) folds that window into the scope's
    durable rows and PERSISTS them here, so a caller must not replay it as
    though it were side-effect-free (safe to repeat — the applied-execution
    watermark makes a re-fold a no-op — but not free). ``dry_run`` MUST suppress
    that write; the dry-run contract is asserted per adapter. ``place`` and
    ``place_cohort`` are pure over ``(book, intents)`` plus edge I/O, and RETURN
    their results rather than storing a settled book: the engine records them.
    ``resync`` reconciles the backend's durable OPEN-order state at cycle start
    (genuinely empty for sim). ``close`` releases whatever the adapter opened.

    ``scope``/``owns_book`` are read-only properties, not writable attributes: a
    conforming adapter is a frozen dataclass, which cannot accept a write.
    """

    @property
    def scope(self) -> str: ...

    @property
    def owns_book(self) -> bool: ...

    async def read_book(self) -> Result[PortfolioSnapshot, FeedError]:
        """The cycle's book advance; an ADVANCING adapter persists rows here."""
        ...

    async def resync(self) -> Result[tuple[OrderResult, ...], FeedError]: ...

    async def attest_protection(
        self, book: PortfolioState
    ) -> Result[tuple[OrderResult, ...], FeedError]:
        """The per-cycle backstop: report every OPEN lot of *book* resting no leg.

        ATTEST-ONLY — it never places, cancels or amends, so it cannot be the
        thing that submits an order (a protective POST is ``place_cohort``'s
        job). It reads the backend's working-orders surface once and returns an
        ``UNPROTECTED`` row per lot with no protective leg of ours resting;
        ``UNPROTECTED`` is in the unsafe set, so the cycle exits 3 rather than
        passing a naked lot as clean (spec §4.2). A failed read is reported for
        every target — ambiguity is never clean — and no POST follows it.
        """
        ...

    async def place(
        self, book: PortfolioState, intent: OrderIntent
    ) -> Result[OrderResult, FeedError]: ...

    async def place_cohort(
        self, book: PortfolioState, intents: tuple[OrderIntent, ...]
    ) -> Result[tuple[OrderResult, ...], FeedError]: ...

    async def cancel_order(self, order_id: str) -> Result[None, FeedError]: ...

    async def close(self) -> Result[None, FeedError]: ...


def resolve_adapter(
    cfg: LiveConfig,
    adapter_name: AdapterName,
    scope: str,
    ledger: SqliteLedger,
    dry_run: bool,
    log: Callable[[str], None],
    *,
    client: IbkrClient | None = None,
    allow_oth: bool = False,
) -> LiveAdapter:
    """Build the adapter *adapter_name* names. No mutation, no shared state.

    ``dry_run`` is threaded into BOTH factories as defence in depth: an adapter
    built for a read-only cycle places nothing, so a missing CLI guard cannot
    become a real order. ``client`` is the ibkr gateway client, unused by sim.
    ``allow_oth`` when true, the broker may auto-confirm orders placed outside
    trading hours.
    """
    if adapter_name == "sim":
        return build_sim_adapter(cfg, scope, ledger, dry_run, log)
    return build_ibkr_adapter(
        cfg, scope, ledger, client, dry_run, log, allow_oth=allow_oth
    )


__all__ = ["LiveAdapter", "resolve_adapter"]
