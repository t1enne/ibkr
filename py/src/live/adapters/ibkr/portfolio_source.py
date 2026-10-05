"""The IBKR read path: our per-scope sqlite book + derived cash -> ``PortfolioState``.

Read-only. Nothing here places, cancels or modifies an order.

The book is assembled the way plan rev 4.1 §3 says. The account summary and
positions are **display-only**: N strategies share one account's cash and one
positions endpoint, so neither can be a per-strategy truth. A scope's book is
exactly the executions whose ``cOID`` carries its ``slug(scope)`` prefix, applied
to the durable rows in sqlite (``trades.reconcile``); a position no scope owns is
IGNORED BY DESIGN — not our book, not a mismatch, never traded.

Cash and equity for a scope are derived from that scope's OWN executions (its
config ``initial_capital`` advanced by its fills and commissions), never read
from the summary. The summary is read for display context only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.bt.state import ActionType, PortfolioState, Position
from src.data.ibkr.client import IbkrClient, IbkrError
from src.live.adapters.ibkr.mapping import (
    IbkrPosition,
    parse_executions,
    parse_positions,
    parse_summary,
)
from src.live.adapters.ibkr.trades import BookRow, reconcile
from src.live.ledger import SqliteLedger, execution_cash_delta
from src.live.result import Err, Ok, Result
from src.live.types import FeedError, PortfolioSnapshot, feed_error

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IbkrBook:
    """The read result before it is wrapped in a ``Result`` (useful in tests)."""

    snapshot: PortfolioSnapshot
    warnings: tuple[str, ...]


def build_snapshot(
    rows: tuple[BookRow, ...],
    marks: dict[int, float],
    cash: float,
    initial_capital: float,
    positions: tuple[IbkrPosition, ...],
    warnings: tuple[str, ...],
    as_of: pd.Timestamp,
) -> IbkrBook:
    """Pure: our open book rows + derived cash -> ``PortfolioSnapshot``.

    ``positions`` is used ONLY for per-conid marks (``mktPrice``); it is never a
    book input and a position we do not own produces no warning. ``cash`` is the
    scope's own (derived) number, not the account summary's.
    """
    grouped: dict[str, list[Position]] = {}
    for row in rows:
        if not row.is_open or not row.symbol:
            continue
        grouped.setdefault(row.symbol, []).append(
            _to_position(row, marks.get(row.conid, 0.0), as_of)
        )
    portfolio = PortfolioState(
        cash=cash,
        positions={sym: tuple(items) for sym, items in grouped.items()},
        trades=(),
        equity_curve=(),
        initial_capital=initial_capital,
    )
    return IbkrBook(
        snapshot=PortfolioSnapshot(portfolio=portfolio, as_of=as_of),
        warnings=tuple(warnings),
    )


def _to_position(row: BookRow, mkt_price: float, as_of: pd.Timestamp) -> Position:
    """One open book row -> a live ``Position`` (conid IS the position id).

    ``last_price`` is the symbol's mark when the positions endpoint carried one
    (so equity sizing marks to market), else the entry price.
    """
    return Position(
        symbol=row.symbol,
        qty=row.qty,
        entry_price=row.entry_price,
        entry_time=row.opened_at if row.opened_at is not None else as_of,
        stop_loss=row.stop_loss,
        take_profit=row.take_profit,
        last_price=mkt_price if mkt_price > 0 else row.entry_price,
        type=ActionType.long if row.side == "long" else ActionType.short,
        position_id=str(row.conid),
        tag=row.tag,
    )


class IbkrPortfolioSource:
    """``PortfolioSource`` over the gateway: summary (display) + our scope book.

    The book advances from the trades window via ``trades.reconcile`` and is
    persisted to the scope's sqlite rows. A dry run persists nothing.
    """

    owns_book = False

    def __init__(
        self,
        client: IbkrClient,
        *,
        scope: str,
        ledger: SqliteLedger,
        initial_capital: float,
        account: str | None = None,
        dry_run: bool = False,
        now: pd.Timestamp | None = None,
    ) -> None:
        self._client = client
        self._scope = scope
        self._ledger = ledger
        self._initial_capital = initial_capital
        self._account = account
        self._dry_run = dry_run
        self._now = now

    async def fetch(self) -> Result[PortfolioSnapshot, FeedError]:
        """Read summary/positions/trades; advance + persist the scope's book."""
        try:
            summary_raw = await self._client.portfolio_summary(self._account)
            positions_raw = await self._client.positions_all(self._account)
            trades_raw = await self._client.trades()
        except IbkrError as exc:
            return Err(feed_error(exc.kind, str(exc)))

        positions, position_warnings = parse_positions(positions_raw)
        executions, execution_warnings = parse_executions(trades_raw)
        summary, summary_warnings = parse_summary(summary_raw)

        book = self._ledger.load_book(self._scope)
        advanced, reconcile_warnings = reconcile(self._scope, executions, book)
        new_ids = advanced.applied - book.applied
        if not self._dry_run:
            self._ledger.save_book(
                self._scope, advanced, executions, self._initial_capital
            )

        seed = self._ledger.initial_capital_of(self._scope) or self._initial_capital
        cash = self._ledger.cash_of(self._scope, self._initial_capital)
        if self._dry_run:
            # Nothing was persisted, so fold this window's new fills in by hand.
            cash += sum(
                execution_cash_delta(e) for e in executions if e.execution_id in new_ids
            )
        marks = {p.conid: p.mkt_price for p in positions if p.mkt_price > 0}
        as_of = self._now if self._now is not None else pd.Timestamp.now(tz="UTC")
        built = build_snapshot(
            advanced.rows,
            marks,
            cash,
            seed,
            positions,
            position_warnings + execution_warnings,
            as_of,
        )
        for warning in summary_warnings + reconcile_warnings + built.warnings:
            logger.warning("ibkr read: %s", warning)
        logger.info(
            "ibkr read scope=%s: cash %.2f derived from our own executions "
            "(account summary net_liquidation %.2f is display-only)",
            self._scope,
            cash,
            summary.net_liquidation,
        )
        return Ok(built.snapshot)


__all__ = [
    "IbkrBook",
    "IbkrPortfolioSource",
    "build_snapshot",
]
