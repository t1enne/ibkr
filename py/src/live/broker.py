"""Broker edge — route ``OrderIntent``s, settle fills through the backtest core.

v1 is simulated: ``SimulatedBroker`` holds a ``PortfolioState`` and folds each
fill through the SAME ``apply_fill`` primitive the backtest uses, so paper and
backtest book accounting are identical by construction. Real IBKR routing is a
later ``IBKRBroker`` implementing the same ``Broker`` Protocol. Failure is a
value (``Result``/``OrderResult.ok``), never an exception that kills the cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from src.bt.execution.pure import execute_signal
from src.bt.portfolio.pure import apply_fill
from src.bt.state import (
    ActionType,
    Candle,
    ExecutionParams,
    FillEvent,
    PortfolioState,
    TradeSignal,
)
from src.live.result import Ok, Result
from src.live.types import FeedError, OrderIntent


@dataclass(frozen=True)
class OrderResult:
    """The outcome of one placement: the fill (when it landed) or why it didn't."""

    intent: OrderIntent
    fill: FillEvent | None  # None when rejected (simulated or broker)
    ok: bool
    message: str = ""
    position_id: str | None = None  # broker lot id on an OPEN; None on a close


class Broker(Protocol):
    """Async order edge. Fails as a value; seed aligns a simulated book."""

    def seed(self, portfolio: PortfolioState) -> None: ...
    async def place(self, intent: OrderIntent) -> Result[OrderResult, FeedError]: ...
    async def close(self) -> Result[None, FeedError]: ...


def intent_to_signal(intent: OrderIntent, ts: pd.Timestamp) -> TradeSignal:
    """Pure: ``OrderIntent`` → ``TradeSignal`` for the shared execution path.

    ``fill_at_next_open=False`` so ``execute_signal`` prices at ``signal.price``
    (the intent's ``ref_price``) rather than a next-open tick. A close intent
    carries ``position_id``, which ``_close_position`` requires.
    """
    return TradeSignal(
        action=intent.action,
        symbol=intent.symbol,
        timestamp=ts,
        price=intent.ref_price,
        qty=intent.qty,
        reason=intent.reason,
        position_id=intent.position_id,
        stop_loss=intent.stop_loss,
        take_profit=intent.take_profit,
        tag=intent.tag,
        fill_at_next_open=False,
    )


class SimulatedBroker:
    """No real routing: log the intent, price it via ``execute_signal``, apply it.

    Keeps a held ``PortfolioState`` as the paper book. ``seed`` replaces it so
    the engine can align the simulated book with a freshly fetched snapshot.
    """

    def __init__(
        self,
        portfolio: PortfolioState,
        params: ExecutionParams,
        log: Callable[[str], None],
    ) -> None:
        self._portfolio = portfolio
        self._params = params
        self._log = log

    def seed(self, portfolio: PortfolioState) -> None:
        """Replace the held book (engine aligns it with the fetched snapshot)."""
        self._portfolio = portfolio

    def portfolio(self) -> PortfolioState:
        """The current simulated book."""
        return self._portfolio

    async def place(self, intent: OrderIntent) -> Result[OrderResult, FeedError]:
        """Fill *intent* against a synthetic ref-price bar and settle it.

        The bar's OHLCV are all ``intent.ref_price`` (no tick data live), so the
        fill is spread/slippage math only. A bad close (missing lot, no
        ``position_id``) raises inside ``apply_fill`` and is caught here — one
        rejected order must not abort the cycle.
        """
        ts = pd.Timestamp.now()
        ref = intent.ref_price
        candle = Candle(
            timestamp=ts,
            symbol=intent.symbol,
            open=ref,
            high=ref,
            low=ref,
            close=ref,
            volume=0.0,
            interval=None,
        )
        fill = execute_signal(intent_to_signal(intent, ts), candle, self._params)
        try:
            self._portfolio = apply_fill(self._portfolio, fill)
        except ValueError as exc:
            return Ok(OrderResult(intent=intent, fill=None, ok=False, message=str(exc)))
        pid = (
            intent.position_id
            if intent.action is ActionType.close
            else f"{intent.symbol}_{ts.timestamp()}"
        )
        message = (
            f"{intent.action.value} {intent.symbol} qty={intent.qty} "
            f"@ {fill.executed_price:.4f}"
        )
        self._log(message)
        return Ok(
            OrderResult(
                intent=intent,
                fill=fill,
                ok=True,
                message=message,
                position_id=pid,
            )
        )

    async def close(self) -> Result[None, FeedError]:
        """Nothing to tear down for the simulated broker."""
        return Ok(None)
