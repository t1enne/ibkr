"""Stateless sim adapter — the simulated backend behind the ``LiveAdapter`` seam.

Two things make this a real adapter rather than a fixture reader:

1. **The book is sqlite.** ``read_book`` builds the account book from the scope's
   ``live_position source='account'`` rows plus its derived cash, so a sim book
   persists across cycles in the SAME store a live run uses. Nothing is held on
   the object; nothing is written by this adapter — the engine records what it
   returns.
2. **Fills go through the backtest execution core.** ``place_cohort`` runs the
   shared :func:`src.live.pure.sim_place_cohort` — per-order ``execute_signal``
   pricing, then ONE ``SimExchange.settle_cohort`` over the book it was given —
   with friction from ``exec_params_of(cfg)`` and the cohort scale from the shared
   ``apply_fills`` rule. The path is shared with every sim caller, never forked.

The returned ``OrderResult``s carry the settled ``position_id``/``filled_qty``;
the settled BOOK is discarded here and re-derived from the ledger next cycle, so
the ledger stays the single durable book.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import pandas as pd

from src.bt.exchange import SimExchange, default_exchange
from src.bt.state import ActionType, PortfolioState, Position
from src.live.pure import OrderResult, sim_place_cohort
from src.live.ledger import LedgerReadError, SimLot, SqliteLedger
from src.live.result import Err, Ok, Result
from src.live.types import (
    FeedError,
    LiveConfig,
    OrderIntent,
    PortfolioSnapshot,
    exec_params_of,
)

#: One open lot's worth of a book row: ``(position_id, symbol, side, qty, entry)``.
_BareLot = tuple[str, str, str, float, float]


@dataclass(frozen=True)
class SimAdapter:
    """The sim backend: sqlite book, immediate deterministic fills, no gateway.

    A frozen dataclass of immutable deps — no mutable state. ``owns_book`` is
    ``True``: the account rows may hold lots the strategy never opened, so a
    close is scoped to what the ledger says the scope owns.
    """

    scope: str
    config: LiveConfig
    ledger: SqliteLedger
    dry_run: bool
    log: Callable[[str], None]
    exchange: SimExchange
    owns_book: bool = True

    async def read_book(self) -> Result[PortfolioSnapshot, FeedError]:
        """The scope's account book from sqlite: its lots + its derived cash.

        An unreadable store is an ``Err`` (refuse rather than trade a book we
        cannot read); a never-written scope reads as a flat funded book.
        """
        try:
            portfolio = build_account_book(self.ledger, self.scope, self.config)
        except LedgerReadError as exc:
            return Err(FeedError(kind="bad_fixture", message=str(exc)))
        return Ok(
            PortfolioSnapshot(portfolio=portfolio, as_of=pd.Timestamp.now(tz="UTC"))
        )

    async def resync(self) -> Result[tuple[OrderResult, ...], FeedError]:
        """Nothing to reconcile: a sim fill is settled in the cycle that placed it."""
        return Ok(())

    async def place(
        self, book: PortfolioState, intent: OrderIntent
    ) -> Result[OrderResult, FeedError]:
        """Route ONE order as a cohort of one (a lone open fills or rejects as ever)."""
        placed: Result[tuple[OrderResult, ...], FeedError] = await self.place_cohort(
            book, (intent,)
        )
        if isinstance(placed, Err):
            return Err(cast("FeedError", placed.error))
        return Ok(placed.value[0])

    async def place_cohort(
        self, book: PortfolioState, intents: tuple[OrderIntent, ...]
    ) -> Result[tuple[OrderResult, ...], FeedError]:
        """Price + settle the cohort over *book*; return results, hold nothing.

        ``dry_run`` refuses at the edge (defence in depth): a sim adapter built
        for a read-only cycle places nothing even if the caller drops the guard.
        """
        if self.dry_run:
            return Err(
                FeedError(
                    kind="auth",
                    message="sim adapter built for a dry run: refusing to place",
                )
            )
        if not intents:
            return Ok(())
        _settled, results = sim_place_cohort(
            book,
            intents,
            exec_params_of(self.config),
            self.exchange,
            self.log,
        )
        return Ok(results)

    async def close(self) -> Result[None, FeedError]:
        """Nothing to tear down: no session, no held book."""
        return Ok(None)


def build_account_book(
    ledger: SqliteLedger, scope: str, config: LiveConfig
) -> PortfolioState:
    """The sim account book for *scope*: its open lots + its derived cash.

    The lots are the ``source='account'`` rows (the human/engine-editable
    surface); a row with no fill detail contributes no lot, exactly as
    ``book_from_executions`` drops a lot with no size. Cash is the scope's own
    derived number — never the account summary's, since N strategies share one
    account. ``initial_capital`` falls back to the config's, so a never-written
    scope reads as a funded flat book.
    """
    as_of = pd.Timestamp.now(tz="UTC")
    lots = tuple(_bare_lot(lot) for lot in ledger.sim_open_lots(scope))
    grouped: dict[str, list[Position]] = {}
    for position_id, symbol, side, qty, entry in lots:
        if not symbol:
            continue  # ownership-only: no book position to carry
        grouped.setdefault(symbol, []).append(
            Position(
                symbol=symbol,
                qty=qty,
                entry_price=entry,
                entry_time=as_of,
                stop_loss=None,
                take_profit=None,
                last_price=entry,
                type=ActionType.long if side != "short" else ActionType.short,
                position_id=position_id,
            )
        )
    return PortfolioState(
        cash=ledger.cash_of(scope, config.initial_capital),
        positions={sym: tuple(items) for sym, items in grouped.items()},
        trades=(),
        equity_curve=(),
        initial_capital=(ledger.initial_capital_of(scope) or config.initial_capital),
    )


def _bare_lot(lot: SimLot) -> _BareLot:
    """A stored sim lot reduced to what the book needs.

    An ownership-only row (no symbol/size) is skipped: it names a lot we own but
    cannot describe, and inventing a size would put a phantom position in the
    book the cycle sizes against.
    """
    if not lot.symbol or lot.qty is None or lot.entry_price is None:
        return (lot.position_id, "", "", 0.0, 0.0)
    return (
        lot.position_id,
        lot.symbol,
        lot.side or "",
        lot.qty,
        lot.entry_price,
    )


def build_sim_adapter(
    cfg: LiveConfig,
    scope: str,
    ledger: SqliteLedger,
    dry_run: bool,
    log: Callable[[str], None],
) -> SimAdapter:
    """The sim adapter factory (see ``src.live.adapter.resolve_adapter``)."""
    return SimAdapter(
        scope=scope,
        config=cfg,
        ledger=ledger,
        dry_run=dry_run,
        log=log,
        exchange=default_exchange(),
    )


__all__ = ["SimAdapter", "build_account_book", "build_sim_adapter"]
