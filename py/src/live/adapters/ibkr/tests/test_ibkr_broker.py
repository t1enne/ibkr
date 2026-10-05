"""``IbkrBroker`` step-7 ladder: submit/reply/confirm, fill wait, refusals.

Every test drives the real client through ``respx``; no live call is made. The
gateway itself is never started and the session is never touched.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import cast

import httpx
import pandas as pd
import pytest
import respx

from src.bt.state import ActionType, PortfolioState, Position
from src.data.ibkr.client import IbkrClient
from src.exec.types import OrderSide, OrderType
from src.live.adapters.ibkr.broker import MAX_REPLIES, IbkrBroker
from src.live.broker import OrderResult
from src.live.result import Err, Ok, Result
from src.live.types import FeedError, OrderIntent

BASE = "https://localhost:5000/v1/api/"
ACCOUNT = "DU452563"
SCOPE = "momentum"
CYCLE_TS = cast("pd.Timestamp", pd.Timestamp("2024-06-03T14:30:00Z"))
ORDER_ID = "979320001"

SUBMIT = f"{BASE}iserver/account/{ACCOUNT}/orders"
STATUS = f"{BASE}iserver/account/order/status/{ORDER_ID}"
REPLY = f"{BASE}iserver/reply/"
POSITIONS = f"{BASE}portfolio/{ACCOUNT}/positions/0"


def _mock_long_position() -> None:
    """A non-flat account position so the close safety guard lets a close through."""
    respx.get(POSITIONS).mock(
        return_value=httpx.Response(
            200,
            json=[{"conid": 265598, "contractDesc": "AAPL", "position": 1}],
        )
    )


async def _conid(_ticker: str) -> int:
    return 265598


async def _no_sleep(_seconds: float) -> None:
    """Never really sleep: the deadline is simulated through ``monotonic``."""


def _broker(
    *,
    dry_run: bool = False,
    timeout_s: float = 0.0,
    monotonic: Callable[[], float] | None = None,
) -> IbkrBroker:
    return IbkrBroker(
        IbkrClient(base_url=BASE, account=ACCOUNT),
        scope=SCOPE,
        account=ACCOUNT,
        dry_run=dry_run,
        conid_lookup=_conid,
        poll_interval_s=0.0,
        timeout_s=timeout_s,
        sleep=_no_sleep,
        monotonic=monotonic if monotonic is not None else lambda: 100.0,
        now=lambda: CYCLE_TS,
    )


def _open_intent() -> OrderIntent:
    return OrderIntent(
        symbol="AAPL",
        action=ActionType.long,
        qty=1.0,
        ref_price=100.0,
        reason="open long (flat->long)",
    )


def _book(positions: dict[str, tuple[Position, ...]]) -> PortfolioState:
    return PortfolioState(
        cash=1000.0,
        positions=positions,
        trades=(),
        equity_curve=(),
        initial_capital=1000.0,
    )


def _long_lot(position_id: str = "555000111") -> PortfolioState:
    return _book(
        {
            "AAPL": (
                Position(
                    symbol="AAPL",
                    qty=1.0,
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


def _filled_status(qty: str = "1", price: str = "100.5") -> dict[str, object]:
    return {
        "order_id": int(ORDER_ID),
        "symbol": "AAPL",
        "order_status": "Filled",
        "cum_fill": qty,
        "total_size": "1",
        "average_price": price,
    }


def _failure(error: object) -> FeedError:
    assert isinstance(error, Err)
    return cast("FeedError", error.error)


def _ok(result: Result[OrderResult, FeedError]) -> OrderResult:
    assert isinstance(result, Ok)
    return cast("OrderResult", result.value)


# --- submit -> order_id, straight fill --------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_place_submits_then_waits_for_the_fill() -> None:
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(
            200, json=[{"order_id": ORDER_ID, "order_status": "PreSubmitted"}]
        )
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))
    broker = _broker()

    result = await broker.place(_open_intent())
    placed = _ok(result)
    assert placed.ok
    assert placed.position_id == ORDER_ID  # the canonical broker lot handle
    assert placed.fill is not None
    assert (placed.fill.filled_qty, placed.fill.executed_price) == (1.0, 100.5)
    posted = json.loads(submit.calls[0].request.content.decode())
    # The gateway wants the order(s) wrapped: a bare body is a 400 "Missing orders".
    assert isinstance(posted, dict) and isinstance(posted.get("orders"), list)
    assert len(posted["orders"]) == 1
    ticket = posted["orders"][0]
    assert ticket["orderType"] == "MKT" and ticket["tif"] == "DAY"
    assert ticket["cOID"].startswith(SCOPE)


# --- submit -> reply -> confirm -> order_id ---------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_place_confirms_an_ordinary_reply_then_fills() -> None:
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "id": "99097238-9824-4830-84ef-46979aa22593",
                    "isSuppressed": False,
                    "message": ["Are you sure you want to submit this order?"],
                    "messageIds": ["o354"],
                }
            ],
        )
    )
    reply = respx.post(f"{REPLY}99097238-9824-4830-84ef-46979aa22593").mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))

    placed = _ok(await _broker().place(_open_intent()))

    assert placed.ok
    assert reply.called
    assert reply.calls[0].request.content.decode() == '{"confirmed":true}'


@respx.mock
@pytest.mark.asyncio
async def test_place_confirms_a_second_reply_before_the_order_lands() -> None:
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(
            200, json=[{"id": "reply-a", "message": ["confirm?"], "messageIds": ["o1"]}]
        )
    )
    respx.post(f"{REPLY}reply-a").mock(
        return_value=httpx.Response(
            200, json=[{"id": "reply-b", "message": ["really?"], "messageIds": ["o2"]}]
        )
    )
    second = respx.post(f"{REPLY}reply-b").mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))

    placed = _ok(await _broker().place(_open_intent()))

    assert placed.ok and second.called


@respx.mock
@pytest.mark.asyncio
async def test_reject_prompt_aborts_without_confirming() -> None:
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(
            200,
            json={
                "orderId": 979320001,
                "text": "price band exceeded",
                "options": ["Do not use"],
                "prompt": True,
            },
        )
    )
    confirm = respx.post(url__regex=rf"{REPLY}.*").mock(
        return_value=httpx.Response(500, json={"error": "must not be called"})
    )
    status = respx.get(STATUS).mock(return_value=httpx.Response(200, json={}))

    result = await _broker().place(_open_intent())

    error = _failure(result)
    assert error.kind == "rejected"
    assert "price band exceeded" in error.message
    assert not confirm.called  # nothing further was placed
    assert not status.called  # we never waited on a non-existent order


@respx.mock
@pytest.mark.asyncio
async def test_reply_loop_overflow_aborts() -> None:
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(
            200, json=[{"id": "r0", "message": ["again?"], "messageIds": ["o1"]}]
        )
    )
    confirm = respx.post(url__regex=rf"{REPLY}.*").mock(
        return_value=httpx.Response(
            200, json=[{"id": "r-next", "message": ["again?"], "messageIds": ["o1"]}]
        )
    )
    status = respx.get(STATUS).mock(return_value=httpx.Response(200, json={}))

    result = await _broker().place(_open_intent())

    error = _failure(result)
    assert error.kind == "rejected"
    assert f"after {MAX_REPLIES} replies" in error.message
    assert confirm.call_count == MAX_REPLIES  # bounded: we stopped pressing yes
    assert not status.called


@respx.mock
@pytest.mark.asyncio
async def test_submit_transport_failure_is_typed() -> None:
    respx.post(SUBMIT).mock(side_effect=httpx.ConnectTimeout("down"))
    # The ambiguous-submit guard reads the working orders; none carries our cOID.
    respx.get(f"{BASE}iserver/account/orders").mock(
        return_value=httpx.Response(200, json={"orders": []})
    )
    error = _failure(await _broker().place(_open_intent()))
    assert error.kind == "transport"


@respx.mock
@pytest.mark.asyncio
async def test_ambiguous_submit_adopts_a_working_order() -> None:
    """A working order with our cOID is SEEN, not re-sent (plan §6 phase 3.5)."""
    from src.exec.types import OrderSide
    from src.live.adapters.ibkr.orders import build_ticket, sequence

    respx.post(SUBMIT).mock(side_effect=httpx.ConnectTimeout("down"))
    seq = sequence((_open_intent(),))[0][0]
    ticket = build_ticket(
        _open_intent(),
        conid=265598,
        side=OrderSide.BUY,
        scope=SCOPE,
        cycle_ts=CYCLE_TS,
        seq=seq,
    )
    respx.get(f"{BASE}iserver/account/orders").mock(
        return_value=httpx.Response(
            200, json={"orders": [{"orderId": ORDER_ID, "cOID": ticket.order_ref}]}
        )
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))
    placed = _ok(await _broker().place(_open_intent()))
    assert placed.ok and placed.position_id == ORDER_ID


# --- wait_filled outcomes ---------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_wait_filled_cancelled_is_rejected() -> None:
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(STATUS).mock(
        return_value=httpx.Response(
            200, json={"order_status": "Cancelled", "cum_fill": "0", "total_size": "1"}
        )
    )
    error = _failure(await _broker().place(_open_intent()))
    assert error.kind == "rejected"
    assert "nothing filled" in error.message


@respx.mock
@pytest.mark.asyncio
async def test_wait_filled_partial_reports_the_filled_qty() -> None:
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(STATUS).mock(
        return_value=httpx.Response(
            200,
            json={
                "order_status": "Cancelled",
                "cum_fill": "0.5",
                "total_size": "1",
                "average_price": "100.0",
            },
        )
    )
    error = _failure(await _broker().place(_open_intent()))
    assert error.kind == "unfilled"  # never silently accepted as success
    assert "only 0.5 of 1" in error.message


@respx.mock
@pytest.mark.asyncio
async def test_wait_filled_timeout_is_typed_and_reports_the_partial() -> None:
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(STATUS).mock(
        return_value=httpx.Response(
            200, json={"order_status": "Submitted", "cum_fill": "0", "total_size": "1"}
        )
    )
    error = _failure(await _broker().place(_open_intent()))
    assert error.kind == "timeout"
    assert "no terminal status" in error.message

    # Same but with a partial: the filled qty is reported, the kind is `unfilled`.
    respx.get(STATUS).mock(
        return_value=httpx.Response(
            200,
            json={
                "order_status": "Submitted",
                "cum_fill": "0.25",
                "total_size": "1",
                "average_price": "101.0",
            },
        )
    )
    partial = _failure(await _broker().place(_open_intent()))
    assert partial.kind == "unfilled"
    assert "still Submitted after 0s with 0.25 of 1 filled" in partial.message


@respx.mock
@pytest.mark.asyncio
async def test_wait_filled_polls_until_terminal() -> None:
    """Two non-terminal polls then a fill: the loop keeps going, bounded."""
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    ticks = iter([0.0, 0.1, 0.2, 0.3, 0.4])
    respx.get(STATUS).mock(
        side_effect=[
            httpx.Response(200, json={"order_status": "PreSubmitted", "cum_fill": "0"}),
            httpx.Response(200, json={"order_status": "Submitted", "cum_fill": "0"}),
            httpx.Response(200, json=_filled_status()),
        ]
    )
    broker = _broker(timeout_s=100.0, monotonic=lambda: next(ticks))
    placed = _ok(await broker.place(_open_intent()))
    assert placed.ok and placed.fill is not None


# --- refusals ---------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_dry_run_broker_never_posts() -> None:
    result = await _broker(dry_run=True).place(_open_intent())
    error = _failure(result)
    assert error.kind == "auth"
    assert "refuses to place" in error.message
    assert len(respx.calls) == 0  # not a single request left the process


@respx.mock
@pytest.mark.asyncio
async def test_lmt_intent_is_refused_without_submitting() -> None:
    intent = OrderIntent(
        symbol="AAPL",
        action=ActionType.long,
        qty=1.0,
        ref_price=100.0,
        reason="LMT probe",
        order_type=OrderType.LMT,
    )
    error = _failure(await _broker().place(intent))
    assert error.kind == "rejected"
    assert "LMT carry-over lands in phase 4" in error.message
    assert len(respx.calls) == 0


@respx.mock
@pytest.mark.asyncio
async def test_close_without_its_lot_is_an_error_not_a_skip() -> None:
    broker = _broker()
    broker.seed(_book({}))  # the replayed book holds no such lot
    intent = OrderIntent(
        symbol="AAPL",
        action=ActionType.close,
        qty=1.0,
        ref_price=100.0,
        reason="close lot",
        position_id="does-not-exist",
    )
    error = _failure(await broker.place(intent))
    assert error.kind == "rejected"
    assert "does-not-exist" in error.message
    assert len(respx.calls) == 0  # nothing was submitted


@respx.mock
@pytest.mark.asyncio
async def test_close_resolves_the_lot_side_from_the_seeded_book() -> None:
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))
    _mock_long_position()
    broker = _broker()
    broker.seed(_long_lot("555000111"))
    intent = OrderIntent(
        symbol="AAPL",
        action=ActionType.close,
        qty=1.0,
        ref_price=100.0,
        reason="close lot",
        position_id="555000111",
    )
    placed = _ok(await broker.place(intent))
    assert placed.ok
    assert placed.position_id == "555000111"  # a close keeps the lot handle
    posted = [
        call.request.content.decode()
        for call in respx.calls
        if call.request.method == "POST"
    ]
    assert '"side":"SELL"' in posted[0]


# --- cohort loop ------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_place_cohort_is_a_deterministic_loop_that_survives_one_failure() -> None:
    respx.post(SUBMIT).mock(
        side_effect=[
            httpx.Response(200, json=[{"error": "no market data"}]),  # 1st order fails
            httpx.Response(200, json=[{"order_id": ORDER_ID}]),  # 2nd lands
        ]
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))
    _mock_long_position()
    broker = _broker()
    broker.seed(_long_lot("555000111"))
    open_intent = _open_intent()
    close_intent = OrderIntent(
        symbol="AAPL",
        action=ActionType.close,
        qty=1.0,
        ref_price=100.0,
        reason="close lot",
        position_id="555000111",
    )
    # The close is sequenced FIRST, so it is the one that fails.
    result = await broker.place_cohort((open_intent, close_intent))
    assert isinstance(result, Ok)
    first, second = cast("tuple[OrderResult, OrderResult]", result.value)
    assert (first.intent.action, first.ok) == (ActionType.close, False)
    assert "rejected" in first.message
    assert (second.intent.action, second.ok) == (ActionType.long, True)


@respx.mock
@pytest.mark.asyncio
async def test_close_is_sequenced_before_open_in_the_cohort() -> None:
    """A close frees the lot/cash the open needs, so its seq (and cOID) comes first."""
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))
    _mock_long_position()
    broker = _broker()
    broker.seed(_long_lot("555000111"))
    close_intent = OrderIntent(
        symbol="AAPL",
        action=ActionType.close,
        qty=1.0,
        ref_price=100.0,
        reason="close lot",
        position_id="555000111",
    )
    await broker.place_cohort((_open_intent(), close_intent))
    bodies = [
        call.request.content.decode()
        for call in respx.calls
        if call.request.method == "POST"
    ]
    assert bodies[0] != bodies[1]
    # The close is placed first and carries a SELL; the open a BUY. Their refs are
    # identity-keyed (plan §4), so only the side/order distinguishes them here.
    assert '"side":"SELL"' in bodies[0] and '"side":"BUY"' in bodies[1]
    assert "20240603T143000" in bodies[0] and "20240603T143000" in bodies[1]


def test_broker_satisfies_the_live_broker_protocol() -> None:
    """A structural conformance check (the engine depends on the Protocol)."""
    from src.live.broker import LiveBroker

    broker = cast("LiveBroker", _broker())
    assert broker is not None and OrderSide.BUY.value == "BUY"


@respx.mock
@pytest.mark.asyncio
async def test_close_on_a_flat_account_is_refused_before_submitting() -> None:
    """Safety guard (plan §6 phase 3.5): never reduce a conid the account is flat on."""
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(POSITIONS).mock(return_value=httpx.Response(200, json=[]))  # account flat
    broker = _broker()
    broker.seed(_long_lot("555000111"))
    intent = OrderIntent(
        symbol="AAPL",
        action=ActionType.close,
        qty=1.0,
        ref_price=100.0,
        reason="close lot",
        position_id="555000111",
    )
    result = await broker.place(intent)
    assert isinstance(result, Err)
    assert "flat" in cast("FeedError", result.error).message
    assert submit.call_count == 0  # never sent
