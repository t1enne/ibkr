"""The ONE adapter seam: ``LiveAdapter`` plus its name resolution and dispatch.

``PortfolioSource`` + ``LiveBroker`` were two seams describing one thing — the
backend a cycle trades through. They collapse here. Both adapters are STATELESS:
the fetched book travels as a PARAMETER into ``place``/``place_cohort`` and the
settled book leaves as the RETURNED results, so nothing mutable lives on the
object. The durable book is sqlite (``live_position`` + ``live_cash``).

``run_cycle`` depends only on this Protocol; the concrete adapters live in
``adapters/sim`` and ``adapters/ibkr`` and are reached through
:func:`resolve_adapter`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol, cast

from src.bt.state import PortfolioState
from src.config import live_adapter
from src.data.ibkr.client import IbkrClient
from src.live.adapters.ibkr.adapter import build_ibkr_adapter
from src.live.adapters.sim.adapter import build_sim_adapter
from src.live.pure import OrderResult
from src.live.ledger import SqliteLedger
from src.live.result import Result
from src.live.scope import AdapterName
from src.live.types import FeedError, LiveConfig, OrderIntent, PortfolioSnapshot

#: Every adapter name a scope may name, in one place (the validator's domain).
_ADAPTER_NAMES: tuple[str, ...] = ("ibkr", "sim")


class LiveAdapter(Protocol):
    """The backend one cycle trades through: read the book, place, reconcile.

    ``scope`` is the ownership key this adapter's ledger rows are keyed by;
    ``owns_book`` says how a close is scoped (``True``: only lots the scope is
    recorded as owning are closable; ``False``: the book ALREADY holds only this
    scope's lots, so every lot in it is ours).

    ``read_book`` builds a fresh snapshot per call — never a cache. ``place`` and
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

    async def read_book(self) -> Result[PortfolioSnapshot, FeedError]: ...

    async def resync(self) -> Result[tuple[OrderResult, ...], FeedError]: ...

    async def place(
        self, book: PortfolioState, intent: OrderIntent
    ) -> Result[OrderResult, FeedError]: ...

    async def place_cohort(
        self, book: PortfolioState, intents: tuple[OrderIntent, ...]
    ) -> Result[tuple[OrderResult, ...], FeedError]: ...

    async def close(self) -> Result[None, FeedError]: ...


def resolve_adapter_name(
    cli_flag: str | None,
    raw: Mapping[str, object],
    params: Mapping[str, object],
) -> AdapterName:
    """Which adapter this run uses, by precedence (plan §2.2).

    CLI ``--adapter`` (an explicitly passed flag) wins, then the config's
    ``adapter`` key, then the back-compat ``broker`` key — each scanned
    top-level first, then ``strategy_params`` — then ``config.toml``'s
    ``[live] adapter`` (see :func:`src.config.live_adapter`). An unknown winner
    is a config error (``ValueError``), never a silent fallback to the default:
    a typo'd adapter must not trade on the wrong side.
    """
    winner: object = cli_flag
    if winner is None:
        winner = _first_named(raw, params, ("adapter", "broker"))
    if winner is None:
        return cast("AdapterName", live_adapter())
    if winner not in _ADAPTER_NAMES:
        raise ValueError(
            f"adapter must be one of {sorted(_ADAPTER_NAMES)}, got {winner!r}"
        )
    return cast("AdapterName", winner)


def resolve_adapter(
    cfg: LiveConfig,
    adapter_name: AdapterName,
    scope: str,
    ledger: SqliteLedger,
    dry_run: bool,
    log: Callable[[str], None],
    *,
    client: IbkrClient | None = None,
) -> LiveAdapter:
    """Build the adapter *adapter_name* names. No mutation, no shared state.

    ``dry_run`` is threaded into BOTH factories as defence in depth: an adapter
    built for a read-only cycle places nothing, so a missing CLI guard cannot
    become a real order. ``client`` is the ibkr gateway client, unused by sim.
    """
    if adapter_name == "sim":
        return build_sim_adapter(cfg, scope, ledger, dry_run, log)
    return build_ibkr_adapter(cfg, scope, ledger, client, dry_run, log)


def _first_named(
    raw: Mapping[str, object],
    params: Mapping[str, object],
    keys: tuple[str, ...],
) -> object | None:
    """The first value for the first of *keys* present, top-level before params.

    Key order is the precedence (``adapter`` before the back-compat ``broker``);
    within a key, the flat/top-level position wins over ``strategy_params``.
    """
    for key in keys:
        for source in (raw, params):
            if key in source:
                return source[key]
    return None


__all__ = ["LiveAdapter", "resolve_adapter", "resolve_adapter_name"]
