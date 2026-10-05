"""The IBKR read path: positions + summary + execution replay -> ``PortfolioState``.

Read-only. Nothing here places, cancels or modifies an order.

The book is assembled the way plan §3 says: **trades are the finer record**, so
the strategy book is the execution replay (``trades.replay``), cash comes from the
account summary, and the positions endpoint is used as a *cross-check* — a symbol
whose replayed net disagrees with IBKR's net produces a warning that carries the
offending executions. It is never a silent abort: a manual TWS trade on one of our
symbols must surface, not blind the cycle, and the plan is explicit that the
mismatch is reported "not as a silent abort".

Why a mismatch is expected, not alarming: IBKR nets positions **per instrument**,
so a foreign strategy's AAPL lot and ours are one line in the positions endpoint
while the executions keep them separate. The replay is therefore the only honest
per-strategy view, and the cross-check's job is to say so out loud.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.bt.state import ActionType, PortfolioState, Position
from src.data.ibkr.client import IbkrClient, IbkrError
from src.exec.types import OrderSide
from src.live.adapters.ibkr.mapping import (
    IbkrPosition,
    IbkrSummary,
    parse_executions,
    parse_positions,
    parse_summary,
)
from src.live.adapters.ibkr.trades import ReplayedLot, replay
from src.live.result import Err, Ok, Result
from src.live.types import FeedError, PortfolioSnapshot, feed_error

logger = logging.getLogger(__name__)


def signed_net(lots: tuple[ReplayedLot, ...]) -> dict[str, float]:
    """Signed net quantity per symbol from replayed lots (BUY positive)."""
    net: dict[str, float] = {}
    for lot in lots:
        sign = 1.0 if lot.side is OrderSide.BUY else -1.0
        net[lot.symbol] = net.get(lot.symbol, 0.0) + sign * lot.qty
    return net


def cross_check(
    lots: tuple[ReplayedLot, ...],
    positions: tuple[IbkrPosition, ...],
) -> tuple[str, ...]:
    """Compare our replayed net per symbol with IBKR's net position (pure).

    Returns one warning per mismatch, naming the executions that built our side
    (order id + ref + signed qty) and both numbers. A symbol we hold that the
    positions endpoint does not list, or lists with a different net, is reported —
    IBKR nets per instrument, so any second holder shows up here by design.
    """
    ours = signed_net(lots)
    theirs: dict[str, float] = {}
    for position in positions:
        theirs[position.symbol] = theirs.get(position.symbol, 0.0) + position.qty

    warnings: list[str] = []
    for symbol in sorted(ours):
        if abs(ours[symbol] - theirs.get(symbol, 0.0)) < 1e-9:
            continue
        detail = ", ".join(
            f"{lot.order_id}/{lot.order_ref or '-'} "
            f"{'+' if lot.side is OrderSide.BUY else '-'}{lot.qty:g}"
            for lot in lots
            if lot.symbol == symbol and lot.status == "open"
        )
        warnings.append(
            f"net qty mismatch {symbol}: replay={ours[symbol]:g} "
            f"positions={theirs.get(symbol, 0.0):g} [{detail}]"
        )
    return tuple(warnings)


@dataclass(frozen=True)
class IbkrBook:
    """The read result before it is wrapped in a ``Result`` (useful in tests)."""

    snapshot: PortfolioSnapshot
    warnings: tuple[str, ...]


def build_snapshot(
    lots: tuple[ReplayedLot, ...],
    summary: IbkrSummary,
    positions: tuple[IbkrPosition, ...],
    positions_warnings: tuple[str, ...],
    as_of: pd.Timestamp,
) -> IbkrBook:
    """Pure: open lots + summary -> ``PortfolioSnapshot`` (+ all warnings raised).

    ``cash`` is the summary's cash; ``initial_capital`` falls back to net
    liquidation then cash. A lot's ``last_price`` is the positions endpoint's
    ``mktPrice`` when the symbol is marked — so ``size_mode="equity"`` marks the
    book to market — falling back to the entry price only when no mark exists.
    """
    warnings = list(positions_warnings)
    warnings.extend(cross_check(lots, positions))
    marks = {p.symbol: p.mkt_price for p in positions if p.mkt_price > 0}
    grouped: dict[str, list[Position]] = {}
    for lot in lots:
        if lot.status != "open" or not lot.symbol:
            continue
        grouped.setdefault(lot.symbol, []).append(
            _to_position(lot, marks.get(lot.symbol, 0.0))
        )
    cash = summary.total_cash
    initial = summary.net_liquidation if summary.net_liquidation > 0 else cash
    portfolio = PortfolioState(
        cash=cash,
        positions={sym: tuple(items) for sym, items in grouped.items()},
        trades=(),
        equity_curve=(),
        initial_capital=initial,
    )
    return IbkrBook(
        snapshot=PortfolioSnapshot(portfolio=portfolio, as_of=as_of),
        warnings=tuple(warnings),
    )


def _to_position(lot: ReplayedLot, mkt_price: float) -> Position:
    """One open lot -> a live ``Position`` (broker order id IS the position id).

    ``last_price`` is the symbol's mark when the positions endpoint carried one
    (so equity sizing marks to market), else the entry price.
    """
    return Position(
        symbol=lot.symbol,
        qty=lot.qty,
        entry_price=lot.entry_price,
        entry_time=lot.ts,
        stop_loss=None,
        take_profit=None,
        last_price=mkt_price if mkt_price > 0 else lot.entry_price,
        type=ActionType.long if lot.side is OrderSide.BUY else ActionType.short,
        position_id=lot.order_id,
        tag="",
    )


class IbkrPortfolioSource:
    """``PortfolioSource`` over the gateway: summary + positions + trade replay.

    Async to match the Protocol; every edge failure becomes an ``Err``. Warnings
    (parse skips, cross-check mismatches) are logged AND echoed into the log at
    the point they are found — they never turn into an ``Err``.

    ``owns_book`` is ``False``: the replay keeps only OUR executions (foreign
    orders are excluded in ``trades.replay``), so every lot in this book is
    closable and close scoping must not consult the (empty on a dry run) ledger.
    """

    owns_book = False

    def __init__(
        self,
        client: IbkrClient,
        ref_prefix: str,
        account: str | None = None,
        now: pd.Timestamp | None = None,
    ) -> None:
        self._client = client
        self._ref_prefix = ref_prefix
        self._account = account
        self._now = now

    async def fetch(self) -> Result[PortfolioSnapshot, FeedError]:
        """Read summary, positions and trades; replay into one snapshot."""
        try:
            summary_raw = await self._client.portfolio_summary(self._account)
            positions_raw = await self._client.positions_all(self._account)
            trades_raw = await self._client.trades()
        except IbkrError as exc:
            return Err(feed_error(exc.kind, str(exc)))

        positions, position_warnings = parse_positions(positions_raw)
        executions, execution_warnings = parse_executions(trades_raw)
        summary, summary_warnings = parse_summary(summary_raw)
        book = replay(executions, ref_prefix=self._ref_prefix)
        as_of = self._now if self._now is not None else pd.Timestamp.now(tz="UTC")
        built = build_snapshot(book.lots, summary, positions, position_warnings, as_of)
        for warning in (
            summary_warnings + execution_warnings + book.warnings + built.warnings
        ):
            logger.warning("ibkr read: %s", warning)
        if book.foreign:
            logger.info(
                "ibkr read: %d foreign execution(s) excluded from the strategy book",
                len(book.foreign),
            )
        return Ok(built.snapshot)
