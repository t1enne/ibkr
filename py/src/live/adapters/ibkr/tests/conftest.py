"""Shared wiring for the ``IbkrBroker`` critical-path tests.

Every test drives the REAL client through ``respx`` against the captured
gateway fixtures; no live call is made and no session is touched. The helpers
here are the ones the removed monolithic module used to carry inline: the broker
construction (no durable ``intents=`` store — the broker is the source of
truth), the empty executions/positions defaults a bare ``place`` reads, and the
intent/book builders.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import httpx
import pandas as pd
import pytest
import respx

from src.bt.state import ActionType, ExecutionParams, PortfolioState, Position
from src.data.ibkr.client import IbkrClient
from src.exec.types import FixedCommission
from src.live.adapters.ibkr.broker import IbkrBroker
from src.live.ports import BookExposure
from src.live.pure import OrderResult
from src.live.result import Err, Ok, Result
from src.live.types import FeedError, OrderIntent

BASE = "https://localhost:5000/v1/api/"
ACCOUNT = "DU452563"
SCOPE = "momentum"
CYCLE_TS = cast("pd.Timestamp", pd.Timestamp("2024-06-03T14:30:00Z"))
ORDER_ID = "979320001"
#: The conid the injected ``_conid`` resolves every ticker to. It is the book's
#: lot handle, so a filled open reports it (not the IBKR order id).
CONID = 265598

SUBMIT = f"{BASE}iserver/account/{ACCOUNT}/orders"
STATUS = f"{BASE}iserver/account/order/status/{ORDER_ID}"
REPLY = f"{BASE}iserver/reply/"
POSITIONS = f"{BASE}portfolio/{ACCOUNT}/positions/0"
OPEN_ORDERS = f"{BASE}iserver/account/orders"
TRADES = f"{BASE}iserver/account/trades"

_FIXTURES = Path(__file__).parent / "fixtures"

#: Zero friction/commission: the cohort-scale tests reduce to the raw arithmetic.
FLAT_PARAMS = ExecutionParams(
    spread_bps=0.0, slippage_bps=0.0, commission_model=FixedCommission(0.0)
)


def captured_working_order(ref: str, order_id: str = ORDER_ID) -> dict[str, object]:
    """A REAL captured working-order row (gateway build 10.50.1a) for our key.

    Row 0 of the captured fixture is OURS (it carries ``order_ref``); its field
    SET is the gateway's, so a test can no longer invent a ``cOID`` key — the
    hand-built mocks that injected ``cOID`` are exactly what hid the parser
    reading the wrong field.
    """
    rows = json.loads((_FIXTURES / "gateway_working_orders.json").read_text())["orders"]
    ours = dict(rows[0])
    ours["order_ref"] = ref
    ours["orderId"] = int(order_id)
    return ours


@pytest.fixture(autouse=True)
def _default_empty_executions():
    """Defaults for the two reads a bare ``place`` makes.

    The executions window is empty: ``place`` sweeps it before minting whenever
    no working order carries the key's pure ref, so an empty read means no lost
    predecessor and most tests proceed to submit. The account is FLAT: the OPEN
    exposure cross-check sees no unexplained net. A test that needs a position
    re-mocks the same ``POSITIONS`` route (respx overrides it in place).
    """
    respx.get(TRADES).mock(return_value=httpx.Response(200, json={"trades": []}))
    respx.get(POSITIONS).mock(return_value=httpx.Response(200, json=[]))
    yield


def mock_no_working_orders() -> None:
    """Pre-flight sees no working order carrying our ref (a fresh submit)."""
    respx.get(OPEN_ORDERS).mock(return_value=httpx.Response(200, json={"orders": []}))


def mock_long_position() -> None:
    """A non-flat account position so the close safety guard lets a close through."""
    respx.get(POSITIONS).mock(
        return_value=httpx.Response(
            200,
            json=[{"conid": CONID, "contractDesc": "AAPL", "position": 1}],
        )
    )


def mock_flip_positions() -> None:
    """The account FLIPS flat once our close fills: long 1, then nothing.

    A close->open flip places and awaits the close to terminal BEFORE the open's
    guard runs, so the account has already applied our close (flat) while the
    book still shows the pre-close lot. Serving the post-close account to the
    open's guard is what a real flip looks like.
    """
    respx.get(POSITIONS).mock(
        side_effect=[
            httpx.Response(
                200,
                json=[{"conid": CONID, "contractDesc": "AAPL", "position": 1}],
            ),
            httpx.Response(200, json=[]),
        ]
    )


class FakeExposure:
    """A ``BookExposure`` with a fixed net on every conid (default: flat)."""

    def __init__(self, net: float = 0.0) -> None:
        self._net = net

    def net_exposure(self, conid: int) -> float:
        return self._net


async def _conid(_ticker: str) -> int:
    return CONID


async def _no_sleep(_seconds: float) -> None:
    """Never really sleep: the deadline is simulated through ``monotonic``."""


def broker(
    *,
    dry_run: bool = False,
    timeout_s: float = 0.0,
    monotonic: Callable[[], float] | None = None,
    now: Callable[[], pd.Timestamp] | None = None,
    log: Callable[[str], None] | None = None,
    exposure: BookExposure | None = None,
    params: ExecutionParams | None = None,
) -> IbkrBroker:
    """An ``IbkrBroker`` wired to the captured-fixture respx routes.

    There is no durable open-order store to inject: the broker itself decides an
    order's existence from its reads (working orders, then the executions
    sweep).
    """
    return IbkrBroker(
        IbkrClient(base_url=BASE, account=ACCOUNT),
        scope=SCOPE,
        params=params if params is not None else FLAT_PARAMS,
        account=ACCOUNT,
        exposure=exposure if exposure is not None else FakeExposure(),
        dry_run=dry_run,
        conid_lookup=_conid,
        poll_interval_s=0.0,
        timeout_s=timeout_s,
        sleep=_no_sleep,
        monotonic=monotonic if monotonic is not None else lambda: 100.0,
        now=now if now is not None else (lambda: CYCLE_TS),
        log=log,
    )


def open_intent(decision_ts: pd.Timestamp | None = None) -> OrderIntent:
    return OrderIntent(
        symbol="AAPL",
        action=ActionType.long,
        qty=1.0,
        ref_price=100.0,
        reason="open long (flat->long)",
        decision_ts=decision_ts,
        cash_bound=1000.0,
    )


def sized_open(symbol: str, qty: float, price: float, bound: float) -> OrderIntent:
    """An OPEN intent sized against ``bound`` (the shared ``view.cash``)."""
    return OrderIntent(
        symbol=symbol,
        action=ActionType.long,
        qty=qty,
        ref_price=price,
        reason="open long (flat->long)",
        cash_bound=bound,
    )


def close_intent(
    *, qty: float = 1.0, price: float = 100.0, position_id: str = "555000111"
) -> OrderIntent:
    return OrderIntent(
        symbol="AAPL",
        action=ActionType.close,
        qty=qty,
        ref_price=price,
        reason="close lot",
        position_id=position_id,
    )


def book(positions: dict[str, tuple[Position, ...]]) -> PortfolioState:
    return PortfolioState(
        cash=1000.0,
        positions=positions,
        trades=(),
        equity_curve=(),
        initial_capital=1000.0,
    )


def long_lot(position_id: str = "555000111", qty: float = 1.0) -> PortfolioState:
    return book(
        {
            "AAPL": (
                Position(
                    symbol="AAPL",
                    qty=qty,
                    entry_price=100.0,
                    entry_time=CYCLE_TS,
                    stop_loss=None,
                    take_profit=None,
                    last_price=100.0,
                    type=ActionType.long,
                    position_id=position_id,
                ),
            )
        }
    )


def filled_status(qty: str = "1", price: str = "100.5") -> dict[str, object]:
    return {
        "order_id": int(ORDER_ID),
        "symbol": "AAPL",
        "order_status": "Filled",
        "cum_fill": qty,
        "total_size": "1",
        "average_price": price,
    }


def failure(error: object) -> FeedError:
    """The ``FeedError`` inside an ``Err`` result (asserted, never guessed)."""
    assert isinstance(error, Err)
    return cast("FeedError", error.error)


def ok_result(result: Result[OrderResult, FeedError]) -> OrderResult:
    """The ``OrderResult`` inside an ``Ok`` result (asserted, never guessed)."""
    assert isinstance(result, Ok)
    return cast("OrderResult", result.value)


def cohort_ok(
    result: Result[tuple[OrderResult, ...], FeedError],
) -> tuple[OrderResult, ...]:
    """The ``OrderResult`` tuple inside an ``Ok`` cohort result (asserted)."""
    assert isinstance(result, Ok)
    return cast("tuple[OrderResult, ...]", result.value)
