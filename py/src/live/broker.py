"""Broker edge — route ``OrderIntent``s, settle fills through the backtest core.

v1 is simulated: ``SimulatedBroker`` holds a ``PortfolioState`` and settles a
whole cycle's fills through the SAME atomic ``apply_fills`` primitive the
backtest uses, so paper and backtest book accounting are identical by
construction — an over-subscribed multi-open cycle SCALES by one shared cash
factor, exactly as a backtest cohort does. Real IBKR routing is a later
``IBKRBroker`` implementing the same ``Broker`` Protocol. Failure is a value
(``Result``/``OrderResult.ok``), never an exception that kills the cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol, cast

import pandas as pd

from src.bt.exchange import SimExchange, default_exchange
from src.bt.portfolio.pure import FillRejection, next_position_id
from src.bt.state import (
    ActionType,
    Candle,
    ExecutionParams,
    FillEvent,
    PortfolioState,
    Position,
    TradeSignal,
)
from src.live.result import Ok, Result
from src.live.types import FeedError, OrderIntent, PortfolioView


@dataclass(frozen=True)
class OrderResult:
    """The outcome of one placement: the fill (when it landed) or why it didn't."""

    intent: OrderIntent
    fill: FillEvent | None  # None when rejected (simulated or broker)
    ok: bool
    message: str = ""
    position_id: str | None = None  # broker lot id on an OPEN; None on a close


class Broker(Protocol):
    """Async order edge. Fails as a value; seed aligns a simulated book.

    ``place`` routes ONE order (real IBKR routing is per-order); ``place_cohort``
    routes a whole cycle and lets a simulated book settle it atomically. A real
    broker may implement ``place_cohort`` as a loop of ``place``.
    """

    def seed(self, portfolio: PortfolioState) -> None: ...
    async def place(self, intent: OrderIntent) -> Result[OrderResult, FeedError]: ...
    async def place_cohort(
        self, intents: tuple[OrderIntent, ...]
    ) -> Result[tuple[OrderResult, ...], FeedError]: ...
    async def close(self) -> Result[None, FeedError]: ...


def trade_signal(
    *,
    symbol: str,
    action: ActionType,
    price: float,
    qty: float,
    ts: pd.Timestamp,
    reason: str = "",
    position_id: str | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    tag: str = "",
    position_side: ActionType | None = None,
) -> TradeSignal:
    """The ONE probe-``TradeSignal`` factory for the live execution path.

    ``fill_at_next_open=False`` so ``execute_signal`` prices at ``price`` rather
    than a next-open tick. Shared by ``intent_to_signal`` (a real order) and
    reconcile's synthetic sizing probes, so the signal shape is defined once.
    ``position_side`` is the side the fill acts on, feeding ``is_buy_fill`` for a
    close; opens are decided by ``action`` alone.
    """
    return TradeSignal(
        action=action,
        symbol=symbol,
        timestamp=ts,
        price=price,
        qty=qty,
        reason=reason,
        position_id=position_id,
        stop_loss=stop_loss,
        take_profit=take_profit,
        tag=tag,
        position_side=position_side,
        fill_at_next_open=False,
    )


def position_side_of(
    portfolio: PortfolioView | None, intent: OrderIntent
) -> ActionType | None:
    """The side of the lot a close targets, read from the book by ``position_id``.

    ``execute_signal`` needs the position's side to lean friction the right way
    (a long closes by selling, a short by buying to cover) and an ``OrderIntent``
    carries only the lot id. ``None`` for a non-close, an unset id, or a book
    holding no such lot — the broker reports those separately (``_close_error``),
    so this never invents a side it cannot prove.
    """
    if portfolio is None or intent.action is not ActionType.close:
        return None
    pid = intent.position_id
    if not pid:
        return None
    return next(
        (
            pos.type
            for pos in portfolio.positions.get(intent.symbol, ())
            if pos.position_id == pid
        ),
        None,
    )


def intent_to_signal(
    intent: OrderIntent, ts: pd.Timestamp, portfolio: PortfolioView | None = None
) -> TradeSignal:
    """Pure: ``OrderIntent`` → ``TradeSignal`` for the shared execution path.

    A close intent carries ``position_id``, which ``_close_position`` requires;
    passing the pre-cycle book resolves that lot's side (``position_side_of``) so
    the close's friction leans against the position rather than defaulting to a
    sell. Opens ignore the book.
    """
    return trade_signal(
        symbol=intent.symbol,
        action=intent.action,
        price=intent.ref_price,
        qty=intent.qty,
        ts=ts,
        reason=intent.reason,
        position_id=intent.position_id,
        stop_loss=intent.stop_loss,
        take_profit=intent.take_profit,
        tag=intent.tag,
        position_side=position_side_of(portfolio, intent),
    )


def ref_candle(price: float, symbol: str, ts: pd.Timestamp) -> Candle:
    """The synthetic ref-price bar a live fill prices against (all OHLCV = ref)."""
    return Candle(
        timestamp=ts,
        symbol=symbol,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=0.0,
        interval=None,
    )


def _open_ranks(
    intents: tuple[OrderIntent, ...], open_indexes: tuple[int, ...]
) -> dict[int, int]:
    """Map each open intent to its application index among the cohort's opens.

    ``apply_fills`` settles its (long/short) opens sorted by ``signal.symbol``
    (stable), so open ``j`` in that sorted order mints its ``position_id`` from
    seq ``base + j`` where ``base`` is the pre-cohort trade count (see
    ``next_position_id``). Reproducing that index here lets the broker report
    the EXACT id settlement minted, so the ledger owns a truly reachable lot.
    """
    sorted_by_symbol = sorted(
        range(len(open_indexes)), key=lambda j: intents[open_indexes[j]].symbol
    )
    return {open_indexes[j]: k for k, j in enumerate(sorted_by_symbol)}


def _close_error(portfolio: PortfolioState, intent: OrderIntent) -> str | None:
    """Why a close cannot apply, mirroring ``_close_position``'s two guards.

    A missing ``position_id``, or an id that matches no lot on a non-empty
    symbol, raises inside ``_close_position`` and would abort the whole cohort;
    pre-checking keeps one bad close a failed order, not a dead cycle. An empty
    symbol is a legal no-op close (``_close_position`` returns unchanged).
    """
    pid = intent.position_id
    if not pid:
        return (
            f"_close_position requires position_id on TradeSignal. "
            f"Signal for {intent.symbol} has no position_id set."
        )
    lots = portfolio.positions.get(intent.symbol)
    if not lots:
        return None
    if pid not in {p.position_id for p in lots}:
        return (
            f"Position {pid} not found for symbol {intent.symbol}. "
            f"Available: {[p.position_id for p in lots]}"
        )
    return None


def _rejection_message(rejection: FillRejection) -> str:
    """A rejected-open message built from the shared ``FillRejection`` record."""
    return (
        f"open rejected: needs {rejection.cash_used} cash, "
        f"have {rejection.available_cash}"
    )


def _named_lot(
    portfolio: PortfolioState, symbol: str, position_id: str
) -> Position | None:
    """The settled lot with *position_id* on *symbol* (the applied open), if any."""
    return next(
        (
            p
            for p in portfolio.positions.get(symbol, ())
            if p.position_id == position_id
        ),
        None,
    )


class SimulatedBroker:
    """No real routing: price each intent, then settle the cycle as one cohort.

    Keeps a held ``PortfolioState`` as the paper book. ``seed`` replaces it so
    the engine can align the simulated book with a freshly fetched snapshot.
    ``place_cohort`` prices orders per-order through the injected
    ``SimExchange`` and settles them atomically via its cohort settler;
    ``place`` is the single-order convenience wrapper.
    """

    def __init__(
        self,
        portfolio: PortfolioState,
        params: ExecutionParams,
        log: Callable[[str], None],
        exchange: SimExchange | None = None,
    ) -> None:
        self._portfolio = portfolio
        self._params = params
        self._log = log
        self._exchange = exchange if exchange is not None else default_exchange()

    def seed(self, portfolio: PortfolioState) -> None:
        """Replace the held book (engine aligns it with the fetched snapshot)."""
        self._portfolio = portfolio

    def portfolio(self) -> PortfolioState:
        """The current simulated book."""
        return self._portfolio

    async def place(self, intent: OrderIntent) -> Result[OrderResult, FeedError]:
        """Route one order; a cohort of one, so a lone open fills or rejects as before."""
        placed = await self.place_cohort((intent,))
        assert isinstance(placed, Ok)  # SimulatedBroker never fails at the edge
        return Ok(cast("tuple[OrderResult, ...]", placed.value)[0])

    async def place_cohort(
        self, intents: tuple[OrderIntent, ...]
    ) -> Result[tuple[OrderResult, ...], FeedError]:
        """Price every intent, then settle the whole cycle in ONE cohort.

        Each intent is priced per-order via ``execute_signal`` (real routing is
        per-order), but the book is settled once through the exchange's
        ``settle_cohort`` — the SAME atomic primitive the backtest uses — so an
        over-subscribed multi-open cycle SCALES by one shared cash factor rather
        than rejecting the tail, and the settled book equals the backtest's on
        the same fills. A single-intent cycle hits ``apply_fills``' lone-open
        guard and is bit-identical to a bare ``apply_fill``. ``FillRejection``s
        from genuine exhaustion are surfaced per-order; one bad close is a failed
        order, never a dead cohort.
        """
        if not intents:
            return Ok(())
        ts = pd.Timestamp.now()
        rejected: dict[int, str] = {}
        fills: dict[int, FillEvent] = {}
        for i, intent in enumerate(intents):
            if intent.action is ActionType.close:
                error = _close_error(self._portfolio, intent)
                if error is not None:
                    rejected[i] = error
                    continue
            fills[i] = self._exchange.execute_signal(
                intent_to_signal(intent, ts, self._portfolio),
                ref_candle(intent.ref_price, intent.symbol, ts),
                self._params,
            )
        # The pre-cohort trade count is the ``base`` the next open's position_id
        # seq builds from (``next_position_id``); opens settle sorted by symbol.
        open_base = len(self._portfolio.trades)
        open_rank = _open_ranks(
            intents,
            tuple(
                i
                for i, intent in enumerate(intents)
                if intent.action in (ActionType.long, ActionType.short)
            ),
        )
        settled, rejections, _scales = self._exchange.settle_cohort(
            self._portfolio,
            tuple(fills[i] for i in sorted(fills)),
            commission_model=self._params.commission_model,
        )
        self._portfolio = settled
        exhausted = {r.symbol: r for r in rejections}
        results = tuple(
            self._result(
                intent, i, ts, fills, rejected, exhausted, open_base, open_rank
            )
            for i, intent in enumerate(intents)
        )
        return Ok(results)

    def _result(
        self,
        intent: OrderIntent,
        index: int,
        ts: pd.Timestamp,
        fills: dict[int, FillEvent],
        rejected: dict[int, str],
        exhausted: dict[str, FillRejection],
        open_base: int,
        open_rank: dict[int, int],
    ) -> OrderResult:
        """Per-order outcome after the cohort settled (scaled qty for applied opens)."""
        if index in rejected:
            return OrderResult(
                intent=intent, fill=None, ok=False, message=rejected[index]
            )
        routed = fills[index]
        if intent.action is ActionType.close:
            fill, position_id = routed, intent.position_id
        else:
            failure = exhausted.get(intent.symbol)
            if failure is not None:
                return OrderResult(
                    intent=intent,
                    fill=None,
                    ok=False,
                    message=_rejection_message(failure),
                )
            position_id = next_position_id(
                intent.symbol, ts, open_base + open_rank[index]
            )
            lot = _named_lot(self._portfolio, intent.symbol, position_id)
            # The cohort's shared scale only moves qty (SL/TP/price untouched), so
            # report the SETTLED qty, not the pre-scale request.
            qty = lot.qty if lot is not None else routed.signal.qty
            fill = replace(
                routed, filled_qty=qty, signal=replace(routed.signal, qty=qty)
            )
        message = (
            f"{intent.action.value} {intent.symbol} qty={fill.signal.qty} "
            f"@ {fill.executed_price:.4f}"
        )
        self._log(message)
        return OrderResult(
            intent=intent,
            fill=fill,
            ok=True,
            message=message,
            position_id=position_id,
        )

    async def close(self) -> Result[None, FeedError]:
        """Nothing to tear down for the simulated broker."""
        return Ok(None)
