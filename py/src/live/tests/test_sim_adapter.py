"""Behaviour tests for the sim adapter: signal construction, cohort scaling, closes.

Every assertion is about what a cycle OBSERVES — the fill the shared execution
core would price, the settled qty after a scaled cohort, a rejected close that
stays a failed order rather than a dead cohort. The book is a parameter here
(the adapter holds none); this is the same ``sim_place_cohort`` path
``live run --adapter sim`` takes.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
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
    PortfolioState,
    Position,
)
from src.live.adapters.sim.adapter import SimAdapter, build_sim_adapter
from src.live.identity import OrderOutcome
from src.live.ledger import SqliteLedger
from src.live.pure import OrderResult, intent_to_signal, sim_place_cohort
from src.live.reconcile import reconcile
from src.live.result import Err, Ok, Result
from src.live.types import FeedError, LiveConfig, LiveSignal, OrderIntent

TS = cast("pd.Timestamp", pd.Timestamp("2024-06-03"))

PlaceResult = Result[OrderResult, FeedError]

_NO_LOG: Callable[[str], None] = lambda _m: None  # noqa: E731


@pytest.fixture(name="ledger")
def _ledger() -> SqliteLedger:
    """A throwaway sqlite store these behaviour tests never read."""
    return SqliteLedger(Path(tempfile.mkdtemp()) / "sim.sqlite")


def _config() -> LiveConfig:
    return LiveConfig(
        strategy_type="m",
        symbols=("AAPL",),
        initial_capital=100_000.0,
        strategy_params={},
        bars=("1d",),
        warmup="1y",
    )


def _book(cash: float = 100_000.0) -> PortfolioState:
    return PortfolioState(
        cash=cash, positions={}, trades=(), equity_curve=(), initial_capital=cash
    )


def _adapter(
    ledger: SqliteLedger,
    exchange: SimExchange | None = None,
    cfg: LiveConfig | None = None,
    dry_run: bool = False,
) -> SimAdapter:
    """A sim adapter over *ledger*; ``sim_t_x`` names its book."""
    adapter = build_sim_adapter(cfg or _config(), "sim_t_x", ledger, dry_run, _NO_LOG)
    return adapter if exchange is None else replace(adapter, exchange=exchange)


def _open_intent(
    action: ActionType = ActionType.long, qty: float = 10.0, symbol: str = "AAPL"
) -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        action=action,
        qty=qty,
        ref_price=100.0,
        reason="open long (flat->long)",
    )


def _ref_candle(intent: OrderIntent) -> Candle:
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


@pytest.mark.asyncio
async def test_place_matches_execute_signal(ledger: SqliteLedger) -> None:
    result: PlaceResult = await _adapter(ledger).place(_book(), _open_intent())
    assert isinstance(result, Ok)
    order = cast(OrderResult, result.value)
    assert order.ok
    assert order.fill is not None

    expected = execute_signal(
        intent_to_signal(order.intent, TS), _ref_candle(order.intent), ExecutionParams()
    )
    assert order.fill.executed_price == expected.executed_price
    assert order.fill.slippage == expected.slippage
    assert order.fill.commission == expected.commission


@pytest.mark.asyncio
async def test_place_settles_book_like_backtest(ledger: SqliteLedger) -> None:
    adapter = _adapter(ledger)
    book = _book()
    result: PlaceResult = await adapter.place(book, _open_intent(qty=10.0))
    assert isinstance(result, Ok)
    order = cast(OrderResult, result.value)
    fill = order.fill
    assert fill is not None

    fresh = apply_fill(book, fill)
    assert fresh.cash == 100_000.0 - (10.0 * fill.executed_price + fill.commission)
    lots = fresh.positions["AAPL"]
    assert len(lots) == 1
    assert lots[0].type is ActionType.long
    assert lots[0].qty == 10.0


@pytest.mark.asyncio
async def test_open_exceeding_cash_is_rejected_book_unchanged(
    ledger: SqliteLedger,
) -> None:
    book = _book(cash=100.0)
    # 10 * 100 = 1000 notional >> 100 cash: the shared cash guard drops it.
    result: PlaceResult = await _adapter(ledger).place(book, _open_intent(qty=10.0))
    assert isinstance(result, Ok)
    order = cast(OrderResult, result.value)
    assert order.ok is False
    assert order.fill is None
    assert order.outcome is OrderOutcome.REJECTED
    assert "open rejected" in order.message


@pytest.mark.asyncio
async def test_close_without_position_id_is_rejected_not_raised(
    ledger: SqliteLedger,
) -> None:
    intent = OrderIntent(
        symbol="AAPL",
        action=ActionType.close,
        qty=5.0,
        ref_price=100.0,
        reason="close lot None",
        position_id=None,
    )
    result: PlaceResult = await _adapter(ledger).place(_book(), intent)
    assert isinstance(result, Ok)
    order = cast(OrderResult, result.value)
    assert order.ok is False
    assert order.fill is None
    assert "position_id" in order.message


@pytest.mark.asyncio
async def test_open_assigns_synthetic_position_id(ledger: SqliteLedger) -> None:
    result: PlaceResult = await _adapter(ledger).place(_book(), _open_intent())
    assert isinstance(result, Ok)
    pid = cast(OrderResult, result.value).position_id
    assert pid is not None
    assert pid.startswith("AAPL_")


@pytest.mark.asyncio
async def test_open_position_id_round_trips_to_close(ledger: SqliteLedger) -> None:
    # BUG-1 regression: the reported open id must equal the settled lot id, so
    # the ledger owns a REACHABLE lot and a later cycle can close it. A phantom
    # ``SYM_{ts}`` id (settlement mints ``SYM_{ts}_{seq}``) makes ``_owned_ids``
    # filter every real lot out, so a close targeting our lot is never emitted
    # and the position is silently unclosable.
    book = _book()
    settled, (opened,) = sim_place_cohort(
        book,
        (_open_intent(qty=10.0),),
        ExecutionParams(),
        SimExchange(),
        _NO_LOG,
        ts=TS,
    )
    assert opened.ok
    pid = opened.position_id
    assert pid is not None
    # The reported id resolves to the actual settled lot (no phantom).
    assert {p.position_id for p in settled.positions["AAPL"]} == {pid}
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
    (close_intent,) = reconcile(
        (close_sig,), settled, _config(), owned=frozenset({pid})
    )
    assert close_intent.action is ActionType.close
    assert close_intent.position_id == pid
    # And the cohort settles that close: the lot is gone, not left orphaned.
    after, (closed,) = sim_place_cohort(
        settled, (close_intent,), ExecutionParams(), SimExchange(), _NO_LOG, ts=TS
    )
    assert closed.ok
    assert closed.position_id == pid
    assert "AAPL" not in after.positions


@pytest.mark.asyncio
async def test_flip_open_sized_by_reconcile_survives_settlement() -> None:
    # reconcile sizes the open against this cycle's close settled for real; the
    # cohort settlement must land that exact qty — the sizing book and the
    # settled book are the same accounting.
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

    settled, orders = sim_place_cohort(
        book, (close, open_), ExecutionParams(), SimExchange(), _NO_LOG, ts=TS
    )
    assert all(order.ok for order in orders)
    (new_lot,) = settled.positions["AAPL"]
    assert new_lot.type is ActionType.long
    assert new_lot.qty == open_.qty


@pytest.mark.asyncio
async def test_multi_open_cohort_scales_like_backtest() -> None:
    # Two opens whose combined notional over-subscribes cash: the cohort must be
    # SCALED by one shared factor (both land smaller) exactly as bt's
    # ``apply_fills`` does — never reject the tail as sequential ``apply_fill``
    # would.
    book = _book(cash=150.0)
    intents = (
        _open_intent(symbol="AAPL", qty=1.0),
        _open_intent(symbol="MSFT", qty=1.0),
    )

    settled, orders = sim_place_cohort(
        book, intents, ExecutionParams(), SimExchange(), _NO_LOG, ts=TS
    )
    assert [order.ok for order in orders] == [True, True]

    expected_fills = tuple(
        execute_signal(intent_to_signal(i, TS), _ref_candle(i), ExecutionParams())
        for i in intents
    )
    expected, rejections, _scales = apply_fills(book, expected_fills)
    assert rejections == ()
    assert settled.cash == expected.cash
    assert {
        sym: tuple(p.qty for p in lots) for sym, lots in settled.positions.items()
    } == {sym: tuple(p.qty for p in lots) for sym, lots in expected.positions.items()}
    # Both scaled below the 1.0 request: the tail was scaled, not dropped.
    assert all(lots[0].qty < 1.0 for lots in settled.positions.values())
    for order in orders:
        assert order.fill is not None
        assert order.position_id is not None
        (lot,) = settled.positions[order.intent.symbol]
        assert lot.position_id == order.position_id
        assert order.fill.filled_qty == lot.qty


def test_intent_to_signal_prices_at_ref() -> None:
    signal = intent_to_signal(_open_intent(), TS)
    assert signal.fill_at_next_open is False
    assert signal.price == 100.0
    assert signal.qty == 10.0
    assert signal.action is ActionType.long


@pytest.mark.asyncio
async def test_dry_run_adapter_refuses_to_place(ledger: SqliteLedger) -> None:
    adapter = _adapter(ledger, dry_run=True)
    result = await adapter.place(_book(), _open_intent())
    assert isinstance(result, Err)
    assert cast("FeedError", result.error).message == (
        "sim adapter built for a dry run: refusing to place"
    )


@pytest.mark.asyncio
async def test_place_propagates_a_cohort_err_without_relying_on_an_assert(
    ledger: SqliteLedger,
) -> None:
    # place() must not assume place_cohort returned Ok: an assert is stripped
    # under ``python -O``, so the Err must be passed through explicitly.
    class _RefusedCohort(SimAdapter):
        async def place_cohort(
            self, book: PortfolioState, intents: tuple[OrderIntent, ...]
        ) -> Result[tuple[OrderResult, ...], FeedError]:
            return Err(FeedError(kind="transport", message="cohort refused"))

    base = _adapter(ledger)
    adapter = _RefusedCohort(
        scope=base.scope,
        config=base.config,
        ledger=base.ledger,
        dry_run=base.dry_run,
        log=base.log,
        exchange=base.exchange,
    )
    result = await adapter.place(_book(), _open_intent())
    assert isinstance(result, Err)
    assert cast("FeedError", result.error).message == "cohort refused"
