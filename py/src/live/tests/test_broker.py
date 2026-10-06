"""Tests for the simulated broker edge (fill parity with the backtest core)."""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from src.bt.exchange import SimExchange
from src.bt.execution.pure import execute_signal
from src.bt.portfolio.pure import apply_fill, apply_fills
from src.bt.state import (
    ActionType,
    Candle,
    ExecutionParams,
    FillEvent,
    PortfolioState,
    Position,
    TradeSignal,
)
from src.live.broker import OrderResult, SimulatedBroker, intent_to_signal
from src.live.reconcile import reconcile
from src.live.result import Err, Ok, Result
from src.live.types import FeedError, LiveConfig, LiveSignal, OrderIntent

TS = cast("pd.Timestamp", pd.Timestamp("2024-06-03"))

PlaceResult = Result[OrderResult, FeedError]


def _book() -> PortfolioState:
    return PortfolioState(
        cash=100_000.0,
        positions={},
        trades=(),
        equity_curve=(),
        initial_capital=100_000.0,
    )


def _broker(book: PortfolioState | None = None) -> SimulatedBroker:
    seed = _book() if book is None else book
    return SimulatedBroker(seed, ExecutionParams(), lambda _m: None)


def _open_intent(
    action: ActionType = ActionType.long, qty: float = 10.0
) -> OrderIntent:
    return OrderIntent(
        symbol="AAPL",
        action=action,
        qty=qty,
        ref_price=100.0,
        reason="open long (flat->long)",
    )


def _ref_candle() -> Candle:
    return Candle(
        timestamp=TS,
        symbol="AAPL",
        open=100.0,
        high=100.0,
        low=100.0,
        close=100.0,
        volume=0.0,
        interval=None,
    )


def _open(symbol: str, qty: float) -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        action=ActionType.long,
        qty=qty,
        ref_price=100.0,
        reason="open long (flat->long)",
    )


def _ref_candle_of(intent: OrderIntent) -> Candle:
    ref = intent.ref_price
    return Candle(
        timestamp=TS,
        symbol=intent.symbol,
        open=ref,
        high=ref,
        low=ref,
        close=ref,
        volume=0.0,
        interval=None,
    )


def _qty_by_symbol(portfolio: PortfolioState) -> dict[str, tuple[float, ...]]:
    return {
        sym: tuple(p.qty for p in lots) for sym, lots in portfolio.positions.items()
    }


@pytest.mark.asyncio
async def test_place_matches_execute_signal() -> None:
    broker = _broker()
    intent = _open_intent()
    result: PlaceResult = await broker.place(intent)
    assert isinstance(result, Ok)
    order = cast(OrderResult, result.value)
    assert order.ok
    assert order.fill is not None

    expected = execute_signal(
        intent_to_signal(intent, TS), _ref_candle(), ExecutionParams()
    )
    assert order.fill.executed_price == expected.executed_price
    assert order.fill.slippage == expected.slippage
    assert order.fill.commission == expected.commission


@pytest.mark.asyncio
async def test_place_settles_book_like_backtest() -> None:
    broker = _broker()
    result: PlaceResult = await broker.place(_open_intent(qty=10.0))
    assert isinstance(result, Ok)
    order = cast(OrderResult, result.value)
    fill = order.fill
    assert fill is not None

    fresh = apply_fill(_book(), fill)
    assert fresh.cash == 100_000.0 - (10.0 * fill.executed_price + fill.commission)
    lots = fresh.positions["AAPL"]
    assert len(lots) == 1
    assert lots[0].type is ActionType.long
    assert lots[0].qty == 10.0
    # The broker's held book tracks the same settlement.
    assert broker.portfolio() == fresh


@pytest.mark.asyncio
async def test_open_exceeding_cash_is_rejected_book_unchanged() -> None:
    book = PortfolioState(
        cash=100.0,
        positions={},
        trades=(),
        equity_curve=(),
        initial_capital=100.0,
    )
    broker = _broker(book)
    # 10 * 100 = 1000 notional >> 100 cash: the shared cash guard drops it.
    result: PlaceResult = await broker.place(_open_intent(qty=10.0))
    assert isinstance(result, Ok)
    order = cast(OrderResult, result.value)
    assert order.ok is False
    assert order.fill is None
    assert "open rejected" in order.message
    assert broker.portfolio() == book  # held book untouched


@pytest.mark.asyncio
async def test_close_without_position_id_is_rejected_not_raised() -> None:
    broker = _broker()
    intent = OrderIntent(
        symbol="AAPL",
        action=ActionType.close,
        qty=5.0,
        ref_price=100.0,
        reason="close lot None",
        position_id=None,
    )
    result: PlaceResult = await broker.place(intent)
    assert isinstance(result, Ok)
    order = cast(OrderResult, result.value)
    assert order.ok is False
    assert order.fill is None
    assert "position_id" in order.message
    assert broker.portfolio() == _book()


@pytest.mark.asyncio
async def test_seed_replaces_book() -> None:
    broker = _broker()
    replacement = PortfolioState(
        cash=1.0,
        positions={},
        trades=(),
        equity_curve=(),
        initial_capital=1.0,
    )
    broker.seed(replacement)
    assert broker.portfolio() is replacement
    closed = await broker.close()
    assert isinstance(closed, Ok)
    assert closed.value is None


@pytest.mark.asyncio
async def test_open_assigns_synthetic_position_id() -> None:
    broker = _broker()
    result: PlaceResult = await broker.place(_open_intent())
    assert isinstance(result, Ok)
    order = cast(OrderResult, result.value)
    pid = order.position_id
    assert pid is not None
    assert pid.startswith("AAPL_")


@pytest.mark.asyncio
async def test_open_position_id_round_trips_to_close() -> None:
    # BUG-1 regression: the reported open id must equal the settled lot id, so
    # the ledger owns a REACHABLE lot and a later cycle can close it. A phantom
    # ``SYM_{ts}`` id (settlement mints ``SYM_{ts}_{seq}``) makes ``_owned_ids``
    # filter every real lot out, so a close targeting our lot is never emitted
    # and the position is silently unclosable.
    broker = _broker()
    placed: PlaceResult = await broker.place(_open_intent(qty=10.0))
    assert isinstance(placed, Ok)
    opened = cast(OrderResult, placed.value)
    assert opened.ok
    pid = opened.position_id
    assert pid is not None
    live = broker.portfolio()
    # The reported id resolves to the actual settled lot (no phantom).
    assert {p.position_id for p in live.positions["AAPL"]} == {pid}
    cfg = LiveConfig(
        strategy_type="m",
        symbols=("AAPL",),
        initial_capital=100000.0,
        strategy_params={},
        bars=("1d",),
        warmup="1y",
    )
    close_sig = LiveSignal(
        symbol="AAPL",
        action="close",
        score=1.0,
        reasons=(),
        signal_ts=TS,
        price=100.0,
        qty=0.0,
    )
    # Ownership scoping (the ledger ``_owned_ids`` view) sees the lot, so a
    # close intent targeting it is emitted.
    (close_intent,) = reconcile((close_sig,), live, cfg, owned=frozenset({pid}))
    assert close_intent.action is ActionType.close
    assert close_intent.position_id == pid
    # And the broker settles that close: the lot is gone, not left orphaned.
    placed: Result[tuple[OrderResult, ...], FeedError] = await broker.place_cohort(
        (close_intent,)
    )
    assert isinstance(placed, Ok)
    (closed,) = cast("tuple[OrderResult, ...]", placed.value)
    assert closed.ok
    assert "AAPL" not in broker.portfolio().positions


@pytest.mark.asyncio
async def test_flip_open_sized_by_reconcile_survives_settlement() -> None:
    # reconcile sizes the open against this cycle's close settled for real; the
    # broker's cohort settlement must land that exact qty — the sizing book and
    # the settled book are the same accounting.
    lot = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=TS,
        stop_loss=None,
        take_profit=None,
        last_price=100.0,
        type=ActionType.short,
        position_id="S1",
    )
    book = PortfolioState(
        cash=0.0,
        positions={"AAPL": (lot,)},
        trades=(),
        equity_curve=(),
        initial_capital=0.0,
    )
    cfg = LiveConfig(
        strategy_type="m",
        symbols=("AAPL",),
        initial_capital=0.0,
        strategy_params={},
        bars=("1d",),
        warmup="1y",
        size=0.5,
    )
    flip = LiveSignal(
        symbol="AAPL",
        action="long",
        score=1.0,
        reasons=(),
        signal_ts=TS,
        price=100.0,
        qty=0.0,
    )
    close, open_ = reconcile((flip,), book, cfg)

    broker = _broker(book)
    placed = await broker.place_cohort((close, open_))
    assert isinstance(placed, Ok)
    orders = cast("tuple[OrderResult, ...]", placed.value)
    assert all(order.ok for order in orders)

    (new_lot,) = broker.portfolio().positions["AAPL"]
    assert new_lot.type is ActionType.long
    assert new_lot.qty == open_.qty


@pytest.mark.asyncio
async def test_multi_open_cohort_scales_like_backtest() -> None:
    # Two opens whose combined notional over-subscribes cash: the cohort must be
    # SCALED by one shared factor (both land smaller) exactly as bt's
    # ``apply_fills`` does — never reject the tail as sequential ``apply_fill``
    # would.
    book = PortfolioState(
        cash=150.0, positions={}, trades=(), equity_curve=(), initial_capital=150.0
    )
    broker = _broker(book)
    intents = (_open("AAPL", 1.0), _open("MSFT", 1.0))

    result = await broker.place_cohort(intents)
    assert isinstance(result, Ok)
    orders = cast("tuple[OrderResult, ...]", result.value)
    assert [order.ok for order in orders] == [True, True]

    expected_fills = tuple(
        execute_signal(intent_to_signal(i, TS), _ref_candle_of(i), ExecutionParams())
        for i in intents
    )
    expected, rejections, _scales = apply_fills(book, expected_fills)
    assert rejections == ()
    settled = broker.portfolio()
    assert settled.cash == expected.cash
    assert _qty_by_symbol(settled) == _qty_by_symbol(expected)
    assert set(settled.positions) == {"AAPL", "MSFT"}
    # Both scaled below the 1.0 request: the tail was scaled, not dropped.
    assert all(lots[0].qty < 1.0 for lots in settled.positions.values())
    assert all(order.fill is not None for order in orders)
    # The reported id resolves to the real settled lot (BUG-1 fix), so the
    # result carries the SETTLED (post-scale) qty, exactly the interplay
    # ``_result``'s settled-qty lookback is for — never the 1.0 pre-scale
    # request.
    for order in orders:
        assert order.fill is not None
        assert order.position_id is not None
        (lot,) = settled.positions[order.intent.symbol]
        assert lot.position_id == order.position_id
        assert order.fill.filled_qty == lot.qty


@pytest.mark.asyncio
async def test_injected_handler_prices_fills(monkeypatch: pytest.MonkeyPatch) -> None:
    # The SimExchange seam is injectable: a swapped ``execute_signal`` method on
    # the exchange is used instead of the default.
    calls: list[str] = []

    def _spy(signal: TradeSignal, candle: Candle, params: ExecutionParams) -> FillEvent:
        calls.append(signal.symbol)
        return execute_signal(signal, candle, params)

    exchange = SimExchange()
    monkeypatch.setattr(exchange, "execute_signal", _spy)
    broker = SimulatedBroker(_book(), ExecutionParams(), lambda _m: None, exchange)
    result: PlaceResult = await broker.place(_open_intent())
    assert isinstance(result, Ok)
    assert calls == ["AAPL"]


def test_intent_to_signal_prices_at_ref() -> None:
    signal = intent_to_signal(_open_intent(), TS)
    assert signal.fill_at_next_open is False
    assert signal.price == 100.0
    assert signal.qty == 10.0
    assert signal.action is ActionType.long


class _FailingCohort(SimulatedBroker):
    """A broker whose cohort path returns an ``Err`` (the edge never fails, but be safe)."""

    async def place_cohort(
        self, intents: tuple[OrderIntent, ...]
    ) -> Result[tuple[OrderResult, ...], FeedError]:
        return Err(FeedError(kind="transport", message="cohort refused"))


@pytest.mark.asyncio
async def test_place_propagates_a_cohort_err_without_relying_on_an_assert() -> None:
    # place() must not assume place_cohort returned Ok: an assert is stripped
    # under ``python -O``, so the Err must be passed through explicitly.
    broker = _FailingCohort(_book(), ExecutionParams(), lambda _m: None)
    result = await broker.place(_open_intent())
    assert isinstance(result, Err)
    assert cast("FeedError", result.error).message == "cohort refused"
