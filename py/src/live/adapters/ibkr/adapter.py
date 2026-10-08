"""TEMPORARY shim: ``LiveAdapter`` over today's ``IbkrPortfolioSource`` + ``IbkrBroker``.

**Replace this with a native ibkr adapter (next stage).** It exists only to put
the ibkr backend behind the ONE seam without rewriting its read/routing edges in
the same change, so the engine can be adapter-shaped while ibkr behaviour stays
byte-identical.

Two consequences of the shim, both accepted for now:

- ``place_cohort`` must ``seed`` the wrapped broker with the passed book, because
  ``IbkrBroker`` holds the replayed book as instance metadata (it consults it for
  a close's lot side). A native adapter would take the book as a parameter all
  the way down and drop the held field.
- The wrapped broker instance is constructed PER CYCLE, not per adapter: the
  factory wires one broker the cycle uses and then closes. A native adapter owns
  its client lifetime explicitly.

``resync``/``place``/``close`` are pure delegations — no behaviour is added.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.bt.state import PortfolioState
from src.data.ibkr.client import IbkrClient
from src.live.adapters.ibkr.broker import IbkrBroker
from src.live.adapters.ibkr.portfolio_source import IbkrPortfolioSource
from src.live.pure import OrderResult
from src.live.ledger import SqliteLedger
from src.live.result import Result
from src.live.types import (
    FeedError,
    LiveConfig,
    OrderIntent,
    PortfolioSnapshot,
    exec_params_of,
)


@dataclass(frozen=True)
class IbkrAdapter:
    """Shim ``LiveAdapter``: delegate to the wrapped source + broker, add nothing."""

    scope: str
    source: IbkrPortfolioSource
    broker: IbkrBroker
    owns_book: bool = False

    async def read_book(self) -> Result[PortfolioSnapshot, FeedError]:
        return await self.source.fetch()

    async def resync(self) -> Result[tuple[OrderResult, ...], FeedError]:
        return await self.broker.resync()

    async def place(
        self, book: PortfolioState, intent: OrderIntent
    ) -> Result[OrderResult, FeedError]:
        self.broker.seed(book)
        return await self.broker.place(intent)

    async def place_cohort(
        self, book: PortfolioState, intents: tuple[OrderIntent, ...]
    ) -> Result[tuple[OrderResult, ...], FeedError]:
        self.broker.seed(book)
        return await self.broker.place_cohort(intents)

    async def close(self) -> Result[None, FeedError]:
        return await self.broker.close()


def build_ibkr_adapter(
    cfg: LiveConfig,
    scope: str,
    ledger: SqliteLedger,
    client: IbkrClient | None,
    dry_run: bool,
    log: Callable[[str], None],
) -> IbkrAdapter:
    """Wire the ibkr read + routing edges for one cycle (see ``resolve_adapter``).

    ``client`` is required for ibkr: the factory cannot reach the gateway without
    it, and a missing client is a wiring bug rather than a runtime condition.
    ``dry_run`` reaches BOTH wrapped edges, so neither can place even if the
    caller's guard were dropped.
    """
    if client is None:
        raise ValueError("the ibkr adapter requires a gateway client")
    return IbkrAdapter(
        scope=scope,
        source=IbkrPortfolioSource(
            client,
            scope=scope,
            ledger=ledger,
            initial_capital=cfg.initial_capital,
            dry_run=dry_run,
        ),
        broker=IbkrBroker(
            client,
            scope=scope,
            intents=ledger,
            params=exec_params_of(cfg),
            exposure=ledger,
            dry_run=dry_run,
            log=log,
        ),
    )


__all__ = ["IbkrAdapter", "build_ibkr_adapter"]
