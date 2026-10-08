"""``IbkrBroker`` step-7 ladder: submit/reply/confirm, fill wait, refusals.

Every test drives the real client through ``respx``; no live call is made. The
gateway itself is never started and the session is never touched.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import httpx
import pandas as pd
import pytest
import respx

from src.bt.state import ActionType, ExecutionParams, PortfolioState, Position
from src.data.ibkr.client import IbkrClient
from ib_rest_api_client.models import SecdefSearchResponseItem
from src.exec.types import FixedCommission, OrderType
from src.live.adapters.ibkr import broker as broker_mod
from src.live.adapters.ibkr.broker import MAX_REPLIES, IbkrBroker
from src.live.pure import OrderResult
from src.live.engine import _UNSAFE_OUTCOMES
from src.live.ledger import SqliteLedger
from src.live.identity import (
    WEDGED_CYCLES,
    OPEN_STATES,
    IntentKey,
    IntentRecord,
    IntentState,
    OrderOutcome,
    PendingIntents,
    intent_key,
    order_ref,
)
from src.live.result import Err, Ok, Result
from src.live.ports import BookExposure
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

#: An AUDIT decision bar distinct from the broker's wall clock. Under the
#: bar-free scheme the cOID ignores it (INV-2); it only drives the DAY-order
#: rollover and the audit field.
BAR = cast("pd.Timestamp", pd.Timestamp("2024-06-03T20:00:00Z"))
_FIXTURES = Path(__file__).parent / "fixtures"


def _captured_working_order(ref: str, order_id: str = ORDER_ID) -> dict[str, object]:
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

    The executions window is empty: ``place`` sweeps it before minting when the
    durable store cannot vouch for the predecessor, so an empty read means no
    lost predecessor and most tests proceed to submit. The account is FLAT: the
    OPEN exposure cross-check sees no unexplained net. A test that needs a
    position re-mocks the same ``POSITIONS`` route (respx overrides it in place).
    """
    respx.get(TRADES).mock(return_value=httpx.Response(200, json={"trades": []}))
    respx.get(POSITIONS).mock(return_value=httpx.Response(200, json=[]))
    yield


def _mock_no_working_orders() -> None:
    """Pre-flight sees no working order carrying our cOID (a fresh submit)."""
    respx.get(OPEN_ORDERS).mock(return_value=httpx.Response(200, json={"orders": []}))


def _mock_long_position() -> None:
    """A non-flat account position so the close safety guard lets a close through."""
    respx.get(POSITIONS).mock(
        return_value=httpx.Response(
            200,
            json=[{"conid": 265598, "contractDesc": "AAPL", "position": 1}],
        )
    )


def _mock_flip_positions() -> None:
    """The account FLIPS flat once our close fills: long 1, then nothing.

    A close->open flip places and awaits the close to terminal BEFORE the open's
    guard runs, so the account has already applied our close (flat) while the
    durable book still shows the pre-close lot. Serving the post-close account to
    the open's guard is what a real flip looks like; the constant
    ``_mock_long_position`` hid the D1 bug by never moving off the held lot.
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


#: Zero friction/commission: the cohort-scale tests reduce to the raw arithmetic.
_FLAT_PARAMS = ExecutionParams(
    spread_bps=0.0, slippage_bps=0.0, commission_model=FixedCommission(0.0)
)


class FakeIntents:
    """An in-memory ``PendingIntents`` (the durable store, without sqlite)."""

    def __init__(self) -> None:
        self.records: dict[tuple[str, str], IntentRecord] = {}

    def load(self, key: IntentKey) -> IntentRecord | None:
        return self.records.get((key.scope, key.token()))

    def load_open(self, scope: str) -> tuple[IntentRecord, ...]:
        return tuple(
            r
            for r in self.records.values()
            if r.key.scope == scope and r.state in OPEN_STATES
        )

    def save(self, record: IntentRecord) -> None:
        self.records[(record.key.scope, record.key.token())] = record

    def open_attempt(
        self, key: IntentKey, decision_ts: pd.Timestamp | None, now: pd.Timestamp
    ) -> IntentRecord:
        existing = self.load(key)
        attempt = existing.attempt + 1 if existing is not None else 0
        record = IntentRecord(
            key=key,
            state=IntentState.PENDING,
            attempt=attempt,
            order_ref=order_ref(key, attempt),
            order_id=None,
            decision_ts=decision_ts,
        )
        self.save(record)
        return record

    def close(
        self,
        key: IntentKey,
        state: IntentState,
        order_id: str | None,
        now: pd.Timestamp,
    ) -> None:
        existing = self.load(key)
        if existing is None:
            return
        self.save(
            replace(
                existing,
                state=state,
                order_id=order_id if order_id is not None else existing.order_id,
            )
        )

    def prune(self, before: pd.Timestamp) -> int:
        return 0


def _broker(
    *,
    dry_run: bool = False,
    timeout_s: float = 0.0,
    monotonic: Callable[[], float] | None = None,
    now: Callable[[], pd.Timestamp] | None = None,
    log: Callable[[str], None] | None = None,
    intents: PendingIntents | None = None,
    exposure: BookExposure | None = None,
    params: ExecutionParams | None = None,
) -> IbkrBroker:
    return IbkrBroker(
        IbkrClient(base_url=BASE, account=ACCOUNT),
        scope=SCOPE,
        intents=intents if intents is not None else FakeIntents(),
        params=params if params is not None else _FLAT_PARAMS,
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


def _open_intent(decision_ts: pd.Timestamp | None = None) -> OrderIntent:
    return OrderIntent(
        symbol="AAPL",
        action=ActionType.long,
        qty=1.0,
        ref_price=100.0,
        reason="open long (flat->long)",
        decision_ts=decision_ts,
        cash_bound=1000.0,
    )


def _sized_open(symbol: str, qty: float, price: float, bound: float) -> OrderIntent:
    """An OPEN intent sized against ``bound`` (the shared ``view.cash``)."""
    return OrderIntent(
        symbol=symbol,
        action=ActionType.long,
        qty=qty,
        ref_price=price,
        reason="open long (flat->long)",
        cash_bound=bound,
    )


def _book(positions: dict[str, tuple[Position, ...]]) -> PortfolioState:
    return PortfolioState(
        cash=1000.0,
        positions=positions,
        trades=(),
        equity_curve=(),
        initial_capital=1000.0,
    )


def _long_lot(position_id: str = "555000111", qty: float = 1.0) -> PortfolioState:
    return _book(
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
    _mock_no_working_orders()
    broker = _broker()

    result = await broker.place(_open_intent())
    placed = _ok(result)
    assert placed.ok
    assert placed.position_id == str(CONID)  # the book's lot handle (the conid)
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
    _mock_no_working_orders()

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
    _mock_no_working_orders()

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
    _mock_no_working_orders()

    result = await _broker().place(_open_intent())

    error = _failure(result)
    assert error.kind == "rejected"
    assert "price band exceeded" in error.message
    assert not confirm.called  # nothing further was placed
    assert not status.called  # we never waited on a non-existent order


@respx.mock
@pytest.mark.asyncio
async def test_reply_loop_overflow_reports_unknown_when_order_not_found() -> None:
    """MAX_REPLIES confirmations pressed, no matching working order: state UNKNOWN.

    Every reply we confirmed was a ``{"confirmed": true}`` POST that can submit
    the order, so the outcome is NOT "rejected": it is unresolved, never reported
    as if nothing may be live (finding M2).
    """
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
    working = respx.get(OPEN_ORDERS).mock(
        return_value=httpx.Response(200, json={"orders": []})
    )

    result = await _broker().place(_open_intent())

    error = _failure(result)
    assert error.kind == "unresolved"
    assert "unknown" in error.message
    assert confirm.call_count == MAX_REPLIES  # bounded: we stopped pressing yes
    assert working.called  # we ASKED whether an order is live before deciding
    assert not status.called


@respx.mock
@pytest.mark.asyncio
async def test_reply_loop_overflow_adopts_a_live_order() -> None:
    """The last confirm may have submitted the order: a working one is ADOPTED."""
    intent = _open_intent()
    ref = order_ref(intent_key(SCOPE, intent), 0)
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(
            200, json=[{"id": "r0", "message": ["again?"], "messageIds": ["o1"]}]
        )
    )
    respx.post(url__regex=rf"{REPLY}.*").mock(
        return_value=httpx.Response(
            200, json=[{"id": "r-next", "message": ["again?"], "messageIds": ["o1"]}]
        )
    )
    respx.get(OPEN_ORDERS).mock(
        side_effect=[
            httpx.Response(200, json={"orders": []}),  # pre-flight: nothing yet
            httpx.Response(200, json={"orders": [_captured_working_order(ref)]}),
        ]
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))
    intents = FakeIntents()

    placed = _ok(await _broker(intents=intents).place(intent))

    assert placed.ok and placed.position_id == str(CONID)
    record = intents.load(intent_key(SCOPE, intent))
    assert record is not None and record.state is IntentState.FILLED


@respx.mock
@pytest.mark.asyncio
async def test_post_confirm_abort_reports_unknown_not_not_placed() -> None:
    """An abort AFTER a confirmation is not "not placed" — the confirm may have submitted."""
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(
            200, json=[{"id": "r0", "message": ["confirm?"], "messageIds": ["o1"]}]
        )
    )
    respx.post(f"{REPLY}r0").mock(
        return_value=httpx.Response(200, json={"error": "order rejected"})
    )
    status = respx.get(STATUS).mock(return_value=httpx.Response(200, json={}))
    _mock_no_working_orders()

    error = _failure(await _broker().place(_open_intent()))

    assert error.kind == "unresolved"
    assert "not placed:" not in error.message
    assert "order rejected" in error.message
    assert not status.called


@respx.mock
@pytest.mark.asyncio
async def test_submit_transport_failure_is_unresolved_not_rejected() -> None:
    """A POST that raised may have landed: the state is UNKNOWN, never "not placed".

    The ambiguous-submit path re-reads the working orders; none carries our cOID,
    so the order is reported ``unresolved`` (and a durable UNRESOLVED record is
    persisted) rather than a transport failure a human might answer by re-placing.
    """
    respx.post(SUBMIT).mock(side_effect=httpx.ConnectTimeout("down"))
    # The ambiguous-submit guard reads the working orders; none carries our cOID.
    respx.get(f"{BASE}iserver/account/orders").mock(
        return_value=httpx.Response(200, json={"orders": []})
    )
    intent = _open_intent()
    intents = FakeIntents()
    error = _failure(await _broker(intents=intents).place(intent))
    assert error.kind == "unresolved"
    assert "do not assume it was not placed" in error.message
    record = intents.load(intent_key(SCOPE, intent))
    assert record is not None and record.state is IntentState.UNRESOLVED


@respx.mock
@pytest.mark.asyncio
async def test_ambiguous_submit_adopts_a_working_order() -> None:
    """A POST that errored but landed: the working order is SEEN, not re-sent."""
    intent = _open_intent()
    ref = order_ref(intent_key(SCOPE, intent), 0)
    respx.post(SUBMIT).mock(side_effect=httpx.ConnectTimeout("down"))
    # The pre-flight sees nothing (the POST has not landed yet); the post-error
    # re-check finds the order the failed POST actually placed.
    respx.get(OPEN_ORDERS).mock(
        side_effect=[
            httpx.Response(200, json={"orders": []}),
            httpx.Response(200, json={"orders": [_captured_working_order(ref)]}),
        ]
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))
    intents = FakeIntents()
    placed = _ok(await _broker(intents=intents).place(intent))
    assert placed.ok and placed.position_id == str(CONID)


@respx.mock
@pytest.mark.asyncio
async def test_later_bar_adopts_an_order_minted_on_an_earlier_bar() -> None:
    """The CORE identity fix: adoption survives a bar change.

    Cycle 1's open timed out while live. Cycle 2 re-runs on a LATER decision bar
    with a DIFFERENT wall clock. The ref embeds a bar-free key token, so the
    pre-flight finds cycle 1's working order and ADOPTS it — no second submit.
    Under the old bar-anchored ref a new bar minted a fresh cOID and duplicated
    the position.
    """
    early_bar = cast("pd.Timestamp", pd.Timestamp("2024-06-03T20:00:00Z"))
    late_bar = cast("pd.Timestamp", pd.Timestamp("2024-06-04T20:00:00Z"))
    prior_ref = order_ref(intent_key(SCOPE, _open_intent()), 0)
    submit = respx.post(SUBMIT).mock(return_value=httpx.Response(500, json={}))
    respx.get(OPEN_ORDERS).mock(
        return_value=httpx.Response(
            200, json={"orders": [_captured_working_order(prior_ref)]}
        )
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))
    intents = FakeIntents()
    broker = _broker(
        now=lambda: cast("pd.Timestamp", pd.Timestamp("2024-06-05T09:00:00Z")),
        intents=intents,
    )

    placed = _ok(await broker.place(_open_intent(decision_ts=late_bar)))

    assert placed.ok and placed.position_id == str(CONID)
    assert placed.outcome is OrderOutcome.ADOPTED
    assert not submit.called  # the earlier-bar order was adopted, not re-sent
    assert early_bar < late_bar  # the bars genuinely differ


@respx.mock
@pytest.mark.asyncio
async def test_confirm_post_adopts_when_a_working_order_appears() -> None:
    """INV-1: before the confirm POST a working order for our key is ADOPTED, not confirmed."""
    intent = _open_intent()
    ref = order_ref(intent_key(SCOPE, intent), 0)
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(
            200, json=[{"id": "r0", "message": ["confirm?"], "messageIds": ["o1"]}]
        )
    )
    confirm = respx.post(f"{REPLY}r0").mock(
        return_value=httpx.Response(500, json={"error": "must not be called"})
    )
    respx.get(OPEN_ORDERS).mock(
        side_effect=[
            httpx.Response(200, json={"orders": []}),  # pre-flight
            httpx.Response(200, json={"orders": [_captured_working_order(ref)]}),
        ]
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))

    placed = _ok(await _broker().place(intent))

    assert placed.ok
    assert not confirm.called  # the working order was adopted, not re-confirmed


@respx.mock
@pytest.mark.asyncio
async def test_failed_read_before_confirm_sends_no_confirm() -> None:
    """INV-1/INV-3: a failed read before the confirm POST sends nothing, unresolved."""
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(
            200, json=[{"id": "r0", "message": ["confirm?"], "messageIds": ["o1"]}]
        )
    )
    confirm = respx.post(f"{REPLY}r0").mock(
        return_value=httpx.Response(500, json={"error": "must not be called"})
    )
    respx.get(OPEN_ORDERS).mock(
        side_effect=[
            httpx.Response(200, json={"orders": []}),  # pre-flight
            httpx.Response(500, json={}),  # the pre-confirm read fails
        ]
    )

    error = _failure(await _broker().place(_open_intent()))

    assert error.kind == "unresolved"
    assert not confirm.called  # fail CLOSED: no confirm POST


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
    _mock_no_working_orders()
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
    _mock_no_working_orders()
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
    _mock_no_working_orders()
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
    assert partial.kind == "timeout"  # still live at the deadline, not a dead order
    assert "still Submitted after 0s with 0.25 of 1 filled" in partial.message


@respx.mock
@pytest.mark.asyncio
async def test_complete_fill_ahead_of_the_status_flip_is_a_fill() -> None:
    """A filled order whose status has not flipped is a FILL, not an unfilled timeout.

    ``cum_fill == total_size`` while the order still reads ``Submitted``: a fill
    cannot exceed the size, so this is final (finding M7b) — no poll to the
    deadline, no "unfilled ... 1 of 1 filled".
    """
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(STATUS).mock(
        return_value=httpx.Response(
            200,
            json={
                "order_status": "Submitted",
                "cum_fill": "1",
                "total_size": "1",
                "average_price": "100.5",
            },
        )
    )
    _mock_no_working_orders()

    placed = _ok(await _broker().place(_open_intent()))

    assert placed.ok and placed.fill is not None
    assert (placed.fill.filled_qty, placed.fill.executed_price) == (1.0, 100.5)


@respx.mock
@pytest.mark.asyncio
async def test_terminal_filled_without_a_readable_cum_fill_is_unresolved() -> None:
    """A terminal ``Filled`` with a garbled ``cum_fill`` is NOT "nothing filled".

    The order DID fill; only its size is unreadable, so it is reported unknown
    (finding M7a) — never a zero-fill rejection an operator might answer by
    re-placing by hand.
    """
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(STATUS).mock(
        return_value=httpx.Response(
            200, json={"order_status": "Filled", "total_size": "1"}
        )
    )
    _mock_no_working_orders()

    error = _failure(await _broker().place(_open_intent()))

    assert error.kind == "unresolved"
    assert "nothing filled" not in error.message
    assert "unknown" in error.message


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
    _mock_no_working_orders()
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
async def test_stop_carrying_open_is_refused_without_submitting() -> None:
    # No resting stop exists on this adapter: an open that carries a stop must be
    # refused, not sent naked with the levels silently dropped from the report.
    intent = OrderIntent(
        symbol="AAPL",
        action=ActionType.long,
        qty=1.0,
        ref_price=100.0,
        reason="open long (flat->long)",
        stop_loss=95.0,
        take_profit=110.0,
    )
    error = _failure(await _broker().place(intent))
    assert error.kind == "rejected"
    assert "naked order" in error.message
    assert "95.0" in error.message and "110.0" in error.message
    assert len(respx.calls) == 0


@respx.mock
@pytest.mark.asyncio
async def test_open_over_deploying_its_cash_bound_is_refused() -> None:
    """An open whose notional exceeds the scope's funded cash is refused (M5).

    The bound is the decision-time cash reconcile sized against; a stale or
    explicit-qty open sizing past it must not reach the broker on margin.
    """
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    intent = OrderIntent(
        symbol="AAPL",
        action=ActionType.long,
        qty=100.0,
        ref_price=100.0,
        reason="open long (flat->long)",
        cash_bound=1000.0,
    )
    error = _failure(await _broker().place(intent))
    assert error.kind == "rejected"
    assert "exceeds funded cash" in error.message
    assert submit.call_count == 0  # never sent


@respx.mock
@pytest.mark.asyncio
async def test_open_without_a_cash_bound_fails_closed() -> None:
    """An intent that cannot state its funded cash is refused, not traded unbounded."""
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    intent = OrderIntent(
        symbol="AAPL",
        action=ActionType.long,
        qty=1.0,
        ref_price=100.0,
        reason="open long (flat->long)",
    )
    error = _failure(await _broker().place(intent))
    assert error.kind == "rejected"
    assert "no decision-time cash bound" in error.message
    assert submit.call_count == 0


@respx.mock
@pytest.mark.asyncio
async def test_open_deploying_the_full_cash_bound_is_allowed() -> None:
    """A full deployment (notional == bound) is within the guard, not over it."""
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))
    _mock_no_working_orders()
    intent = OrderIntent(
        symbol="AAPL",
        action=ActionType.long,
        qty=10.0,
        ref_price=100.0,
        reason="open long (flat->long)",
        cash_bound=1000.0,
    )
    placed = _ok(await _broker().place(intent))
    assert placed.ok


@respx.mock
@pytest.mark.asyncio
async def test_open_diverging_from_the_account_net_is_refused() -> None:
    """B: a lagged executions feed leaves an account net our book cannot explain.

    The account holds 1 AAPL but the ledger books a flat conid, so a fill we
    cannot see may be live — a second full-size open would double it. Refused
    with the distinct ``divergence`` kind, before any submit.
    """
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(POSITIONS).mock(
        return_value=httpx.Response(
            200, json=[{"conid": CONID, "contractDesc": "AAPL", "position": 1}]
        )
    )
    error = _failure(await _broker().place(_open_intent()))
    assert error.kind == "divergence"
    assert submit.call_count == 0  # never sent


@respx.mock
@pytest.mark.asyncio
async def test_open_refused_when_our_book_is_ahead_of_the_account() -> None:
    """B inverse: a phantom lot (book ahead of the account) is refused too."""
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    broker = _broker(exposure=FakeExposure(5.0))  # book long 5
    error = _failure(await broker.place(_open_intent()))  # account flat
    assert error.kind == "divergence"
    assert submit.call_count == 0


@respx.mock
@pytest.mark.asyncio
async def test_open_allowed_when_account_matches_the_booked_exposure() -> None:
    """B positive control: a shared book that MATCHES the account is not blocked."""
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))
    _mock_no_working_orders()
    respx.get(POSITIONS).mock(
        return_value=httpx.Response(
            200, json=[{"conid": CONID, "contractDesc": "AAPL", "position": 5}]
        )
    )
    broker = _broker(exposure=FakeExposure(5.0))  # another scope books the 5
    placed = _ok(await broker.place(_open_intent()))
    assert placed.ok


@respx.mock
@pytest.mark.asyncio
async def test_open_fails_closed_when_the_account_net_is_unreadable() -> None:
    """B: an unreadable account read refuses the open, consistent with the edge."""
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(POSITIONS).mock(return_value=httpx.Response(500))
    error = _failure(await _broker().place(_open_intent()))
    assert error.kind == "divergence"
    assert submit.call_count == 0


@respx.mock
@pytest.mark.asyncio
async def test_open_fails_closed_without_an_exposure_oracle() -> None:
    """B: no way to see the book is not a licence to open — refuse (fail closed)."""
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    broker = IbkrBroker(
        IbkrClient(base_url=BASE, account=ACCOUNT),
        scope=SCOPE,
        intents=FakeIntents(),
        params=_FLAT_PARAMS,
        account=ACCOUNT,
        conid_lookup=_conid,
    )
    error = _failure(await broker.place(_open_intent()))
    assert error.kind == "divergence"
    assert submit.call_count == 0


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
    _mock_no_working_orders()
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
    _mock_no_working_orders()
    broker = _broker()
    # Two opens (no closes), a big shared bound so nothing scales.
    result = await broker.place_cohort(
        (
            _sized_open("AAPL", 10.0, 100.0, 100_000.0),
            _sized_open("MSFT", 10.0, 100.0, 100_000.0),
        )
    )
    assert isinstance(result, Ok)
    first, second = cast("tuple[OrderResult, OrderResult]", result.value)
    assert (first.intent.symbol, first.ok) == ("AAPL", False)
    assert "rejected" in first.message
    assert (second.intent.symbol, second.ok) == ("MSFT", True)


@respx.mock
@pytest.mark.asyncio
async def test_a_failed_funding_close_drops_the_opens_it_was_funding() -> None:
    """B2: opens are funded by a close that may never arrive.

    The close leg is refused, but the opens were sized against a book with the
    close already settled (their shared cash bound). Without a live
    available-funds read the strictest safe rule is to DROP the opens and report
    them, rather than deploy the prospective cash as silent leverage.
    """
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"error": "no market data"}])
    )
    _mock_long_position()
    _mock_no_working_orders()
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
    result = await broker.place_cohort((_open_intent(), close_intent))
    assert isinstance(result, Ok)
    by_action = {
        r.intent.action: r for r in cast("tuple[OrderResult, ...]", result.value)
    }
    assert by_action[ActionType.close].ok is False
    open_result = by_action[ActionType.long]
    assert open_result.ok is False
    assert "funding close" in open_result.message
    assert open_result.outcome is OrderOutcome.REJECTED


@respx.mock
@pytest.mark.asyncio
async def test_over_cash_open_cohort_is_scaled_by_one_shared_factor() -> None:
    """Two opens at 100% of the shared cash each are scaled, not both sent full."""
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))
    _mock_no_working_orders()
    messages: list[str] = []
    broker = _broker(log=messages.append)

    result = await broker.place_cohort(
        (
            _sized_open("AAPL", qty=100.0, price=100.0, bound=15000.0),
            _sized_open("MSFT", qty=100.0, price=100.0, bound=15000.0),
        )
    )

    assert isinstance(result, Ok)
    results = cast("tuple[OrderResult, ...]", result.value)
    assert all(r.ok for r in results)
    # requested 20000 > budget 15000 -> scale 0.75 -> floor(100 * 0.75) = 75 each.
    posted = sorted(
        json.loads(call.request.content.decode())["orders"][0]["quantity"]
        for call in submit.calls
    )
    assert posted == [75.0, 75.0]
    notice = [m for m in messages if m.startswith("cohort scaled")]
    assert len(notice) == 1
    assert "AAPL" in notice[0] and "MSFT" in notice[0]


@respx.mock
@pytest.mark.asyncio
async def test_lone_open_is_never_scaled() -> None:
    """A lone open is full-size-or-reject: an over-bound one is refused, not shaved."""
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))
    _mock_no_working_orders()
    messages: list[str] = []
    broker = _broker(log=messages.append)

    result = await broker.place_cohort(
        (_sized_open("AAPL", qty=100.0, price=100.0, bound=5000.0),)
    )

    assert isinstance(result, Ok)
    results = cast("tuple[OrderResult, ...]", result.value)
    assert results[0].ok is False
    assert not submit.called  # never submitted, and never scaled to half size
    assert not any(m.startswith("cohort scaled") for m in messages)


@respx.mock
@pytest.mark.asyncio
async def test_scaled_open_that_floors_to_zero_is_dropped() -> None:
    """A scaled open whose whole-share qty is 0 is reported, never submitted."""
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))
    _mock_no_working_orders()
    broker = _broker()

    result = await broker.place_cohort(
        (
            _sized_open("AAPL", qty=100.0, price=100.0, bound=5000.0),
            _sized_open("MSFT", qty=1.0, price=100.0, bound=5000.0),
        )
    )

    assert isinstance(result, Ok)
    results = cast("tuple[OrderResult, ...]", result.value)
    by_symbol = {r.intent.symbol: r for r in results}
    assert by_symbol["AAPL"].ok and by_symbol["AAPL"].intent.qty == 49.0
    assert by_symbol["MSFT"].ok is False
    assert "floors to 0 shares" in by_symbol["MSFT"].message
    posted = [
        json.loads(call.request.content.decode())["orders"][0]["quantity"]
        for call in submit.calls
    ]
    assert posted == [49.0]  # only AAPL was sent


@respx.mock
@pytest.mark.asyncio
async def test_an_all_opens_dropped_cohort_reports_the_unsafe_unfunded_outcome() -> (
    None
):
    """D3: a STRUCTURAL drop carries ``UNFUNDED``, never a safe ``REJECTED``.

    Two opens that disagree on their cash bound make the shared budget
    unprovable, so ``_refuse_opens`` drops EVERY open. A config that can never
    state that bound does this every cycle forever, so the outcomes must be
    unsafe (exit 3) — unlike a genuine cash-exhaustion refusal, which re-mints
    next cycle and stays safe.
    """
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    _mock_no_working_orders()
    broker = _broker()

    result = await broker.place_cohort(
        (
            _sized_open("AAPL", qty=10.0, price=100.0, bound=15000.0),
            _sized_open("MSFT", qty=10.0, price=100.0, bound=9000.0),
        )
    )

    assert isinstance(result, Ok)
    results = cast("tuple[OrderResult, ...]", result.value)
    assert len(results) == 2
    assert all(r.ok is False for r in results)  # every open was dropped
    assert {r.outcome for r in results} == {OrderOutcome.UNFUNDED}
    assert {r.error_kind for r in results} == {"unfunded"}
    assert OrderOutcome.UNFUNDED in _UNSAFE_OUTCOMES
    assert OrderOutcome.UNFUNDED is not OrderOutcome.REJECTED  # never conflate
    assert submit.call_count == 0  # unprovable budget sends nothing


@respx.mock
@pytest.mark.asyncio
async def test_a_genuine_cash_exhaustion_open_stays_a_safe_rejected() -> None:
    """D3 boundary: real cash exhaustion is ``REJECTED`` (safe), not ``UNFUNDED``.

    A lone open over its own bound is refused by the cash guard with kind
    ``rejected`` — nothing was taken and the intent re-mints next bar, so the
    cycle legitimately exits 0. Only a STRUCTURAL drop becomes unsafe.
    """
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    _mock_no_working_orders()
    broker = _broker()

    result = await broker.place_cohort(
        (_sized_open("AAPL", qty=100.0, price=100.0, bound=5000.0),)
    )

    assert isinstance(result, Ok)
    (open_result,) = cast("tuple[OrderResult, ...]", result.value)
    assert open_result.ok is False
    assert open_result.outcome is OrderOutcome.REJECTED
    assert OrderOutcome.REJECTED not in _UNSAFE_OUTCOMES
    assert submit.call_count == 0


@respx.mock
@pytest.mark.asyncio
async def test_close_is_sequenced_before_open_in_the_cohort() -> None:
    """A close frees the lot/cash the open needs, so its seq (and cOID) comes first."""
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))
    _mock_flip_positions()
    _mock_no_working_orders()
    broker = _broker(exposure=FakeExposure(1.0))
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
    # bar-free key tokens (distinct per intent) with no decision-bar segment.
    assert '"side":"SELL"' in bodies[0] and '"side":"BUY"' in bodies[1]
    assert "20240603T143000" not in bodies[0]
    assert f'"cOID":"{SCOPE}-' in bodies[0] and f'"cOID":"{SCOPE}-' in bodies[1]


@respx.mock
@pytest.mark.asyncio
async def test_flip_close_then_open_is_allowed_once_the_close_fills() -> None:
    """D1: a same-cycle close->open flip is NOT refused as a divergence.

    The cohort places the close first and awaits it to terminal before the open.
    By the open's guard the account has already applied our close (flat) while the
    durable book still shows the pre-close long lot, so ``net`` is 0 against a
    ``booked`` of 1. Before the fix ``abs(net - booked) > tol`` refused the
    legitimate open with ``divergence``; the guard must credit our own confirmed
    close (delta -1) and ALLOW it.
    """
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))
    _mock_flip_positions()
    _mock_no_working_orders()
    broker = _broker(exposure=FakeExposure(1.0))
    broker.seed(_long_lot("555000111"))
    close_intent = OrderIntent(
        symbol="AAPL",
        action=ActionType.close,
        qty=1.0,
        ref_price=100.0,
        reason="close lot",
        position_id="555000111",
    )

    result = await broker.place_cohort((_open_intent(), close_intent))

    assert isinstance(result, Ok)
    placed = cast("tuple[OrderResult, ...]", result.value)
    opens = [r for r in placed if r.intent.action is ActionType.long]
    assert opens and opens[0].ok, opens  # the flip's open landed, not DIVERGENCE
    assert opens[0].outcome is OrderOutcome.PLACED
    assert submit.call_count == 2  # close AND open both sent


@respx.mock
@pytest.mark.asyncio
async def test_flip_open_is_refused_when_the_account_holds_more_than_our_close() -> (
    None
):
    """D1 control: the delta excuses ONLY our own close, never a foreign net.

    The account holds 6, we book 1, and our close sells 1 — so the account still
    shows 5 while ``booked + delta`` is 0. That 5 is a holding our sends did not
    produce, so the open must STILL be refused: the in-cycle delta is not a blanket
    excuse for an unexplained net.
    """
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))
    respx.get(POSITIONS).mock(
        side_effect=[
            httpx.Response(
                200,
                json=[{"conid": CONID, "contractDesc": "AAPL", "position": 6}],
            ),
            httpx.Response(
                200,
                json=[{"conid": CONID, "contractDesc": "AAPL", "position": 5}],
            ),
        ]
    )
    _mock_no_working_orders()
    broker = _broker(exposure=FakeExposure(1.0))
    broker.seed(_long_lot("555000111"))
    close_intent = OrderIntent(
        symbol="AAPL",
        action=ActionType.close,
        qty=1.0,
        ref_price=100.0,
        reason="close lot",
        position_id="555000111",
    )

    result = await broker.place_cohort((_open_intent(), close_intent))

    assert isinstance(result, Ok)
    placed = cast("tuple[OrderResult, ...]", result.value)
    opens = [r for r in placed if r.intent.action is ActionType.long]
    assert opens and not opens[0].ok
    assert opens[0].error_kind == "divergence"
    assert submit.call_count == 1  # the close was sent; the open was refused


@respx.mock
@pytest.mark.asyncio
async def test_close_on_a_flat_account_is_refused_before_submitting() -> None:
    """Safety guard (plan §6 phase 3.5): never reduce a conid the account is flat on.

    D3: the refusal is a BOOK DIVERGENCE (our book holds a lot the account does
    not), so it surfaces as ``OrderOutcome.DIVERGENCE`` — an unsafe cycle at exit
    3 — not as a broker ``rejected`` (which the exit contract treats as safe).
    """
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
    error = cast("FeedError", result.error)
    assert "flat" in error.message
    assert error.kind == "divergence"  # D3: unsafe, not a safe "rejected"
    assert submit.call_count == 0  # never sent


@respx.mock
@pytest.mark.asyncio
async def test_close_larger_than_a_shared_net_is_refused_before_submitting() -> None:
    """D3: a close on a SHARED account must not exceed the account net.

    Our book holds long 60, but another scope/human holds 40 short on the same
    conid, so the account nets to 20. The old guard only checked ``abs(net) > eps``,
    so A's SELL 60 passed and flipped the account to -40 — an unintended naked
    short. The refusal is a book divergence, so the strategy that cannot exit
    surfaces at exit 3.
    """
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    # Our long 60 and a FOREIGN short 40 on the same conid net the account to 20.
    respx.get(POSITIONS).mock(
        return_value=httpx.Response(
            200,
            json=[
                {"conid": 265598, "contractDesc": "AAPL", "position": 60},
                {"conid": 265598, "contractDesc": "AAPL", "position": -40},
            ],
        )
    )
    _mock_no_working_orders()
    broker = _broker()
    broker.seed(_long_lot("555000111", qty=60.0))
    intent = OrderIntent(
        symbol="AAPL",
        action=ActionType.close,
        qty=60.0,
        ref_price=100.0,
        reason="close lot",
        position_id="555000111",
    )
    result = await broker.place(intent)
    assert isinstance(result, Err)
    error = cast("FeedError", result.error)
    assert "exceeds the account net" in error.message
    assert error.kind == "divergence"  # D3: unsafe, not a safe "rejected"
    assert submit.call_count == 0  # the flip is refused, nothing sent


@respx.mock
@pytest.mark.asyncio
async def test_close_against_the_opposite_net_side_is_refused() -> None:
    """D3: a SELL cannot reduce a short net — it must be refused, not flipped."""
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(POSITIONS).mock(
        return_value=httpx.Response(
            200, json=[{"conid": 265598, "contractDesc": "AAPL", "position": -5}]
        )
    )
    _mock_no_working_orders()
    broker = _broker()
    broker.seed(_long_lot("555000111", qty=1.0))
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
    error = cast("FeedError", result.error)
    assert "opposite side" in error.message
    assert error.kind == "divergence"  # D3: unsafe, not a safe "rejected"
    assert submit.call_count == 0


@respx.mock
@pytest.mark.asyncio
async def test_a_refused_close_reports_the_unsafe_divergence_outcome() -> None:
    """D3, end to end through a cohort: the refusal carries ``DIVERGENCE``.

    ``place_cohort`` maps the guard's kind to an outcome; a close the account net
    cannot absorb must yield ``OrderOutcome.DIVERGENCE`` (in the engine's unsafe
    set) so the cycle exits 3 — the strategy cannot exit, and cron must see it.
    """
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    # Our long 60 and a FOREIGN short 40 on the conid net the account to 20, so a
    # SELL 60 would flip it to -40: the guard refuses the unexplained net.
    respx.get(POSITIONS).mock(
        return_value=httpx.Response(
            200,
            json=[
                {"conid": CONID, "contractDesc": "AAPL", "position": 60},
                {"conid": CONID, "contractDesc": "AAPL", "position": -40},
            ],
        )
    )
    _mock_no_working_orders()
    broker = _broker()
    broker.seed(_long_lot("555000111", qty=60.0))
    close_intent = OrderIntent(
        symbol="AAPL",
        action=ActionType.close,
        qty=60.0,
        ref_price=100.0,
        reason="close lot",
        position_id="555000111",
    )

    result = await broker.place_cohort((close_intent,))

    assert isinstance(result, Ok)
    (close_result,) = cast("tuple[OrderResult, ...]", result.value)
    assert close_result.ok is False
    assert close_result.error_kind == "divergence"
    assert close_result.outcome is OrderOutcome.DIVERGENCE
    assert OrderOutcome.DIVERGENCE in _UNSAFE_OUTCOMES
    assert submit.call_count == 0  # nothing was sent against an unexplained net


@respx.mock
@pytest.mark.asyncio
async def test_close_within_the_net_is_allowed() -> None:
    """D3 positive control: a close no larger than the net on the reduce side passes."""
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))
    respx.get(POSITIONS).mock(
        return_value=httpx.Response(
            200, json=[{"conid": 265598, "contractDesc": "AAPL", "position": 60}]
        )
    )
    _mock_no_working_orders()
    broker = _broker()
    broker.seed(_long_lot("555000111", qty=60.0))
    intent = OrderIntent(
        symbol="AAPL",
        action=ActionType.close,
        qty=60.0,
        ref_price=100.0,
        reason="close lot",
        position_id="555000111",
    )
    assert _ok(await broker.place(intent)).ok


# --- conid resolution: verification + caching (finding L10) -----------------


def _contract(
    conid: str, symbol: str, *, restricted: bool | None = None
) -> SecdefSearchResponseItem:
    """A US-stock search candidate (primary exchange NASDAQ passes ``_is_usd_stock``)."""
    item = SecdefSearchResponseItem(
        conid=conid, symbol=symbol, description="NASDAQ", restricted=restricted
    )
    return item


async def _candidates(
    *items: SecdefSearchResponseItem,
) -> tuple[SecdefSearchResponseItem, ...]:
    return items


@pytest.mark.asyncio
async def test_default_conid_lookup_refuses_ambiguity_without_a_db_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two distinct conids also naming the ticker and NEITHER matches the DB conid:
    # refuse, naming each candidate's description, rather than pick one.
    monkeypatch.setattr(
        broker_mod,
        "search_contracts",
        lambda _t: _candidates(_contract("1", "AAPL"), _contract("2", "AAPL")),
    )
    monkeypatch.setattr(broker_mod, "_db_conid_for_ticker", lambda t: None)
    with pytest.raises(ValueError, match="ambiguous"):
        await broker_mod._default_conid_lookup("AAPL")


@pytest.mark.asyncio
async def test_default_conid_lookup_refuses_a_mismatched_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The search for AAPL returned a contract for MSFT: never order on it.
    monkeypatch.setattr(
        broker_mod, "search_contracts", lambda _t: _candidates(_contract("1", "MSFT"))
    )
    with pytest.raises(ValueError, match="naming AAPL"):
        await broker_mod._default_conid_lookup("AAPL")


@pytest.mark.asyncio
async def test_default_conid_lookup_refuses_a_restricted_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        broker_mod,
        "search_contracts",
        lambda _t: _candidates(_contract("1", "AAPL", restricted=True)),
    )
    with pytest.raises(ValueError, match="restricted"):
        await broker_mod._default_conid_lookup("AAPL")


@pytest.mark.asyncio
async def test_prefer_db_conid_picks_the_db_matching_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SUCCESS path: among several same-ticker listings, the DB's conid wins.

    A live order routed to the WRONG conid of a ticker lands on the wrong
    instrument, so this is the safety-critical branch. SHOP's real shape: two US
    STK rows share the ticker, only one is the contract our data treats as SHOP.
    The preference must be LOAD-BEARING — flipping the DB's conid to the other
    candidate must change the winner, or the test would pass on any pick.
    """
    monkeypatch.setattr(
        broker_mod,
        "search_contracts",
        lambda _t: _candidates(_contract("1", "SHOP"), _contract("2", "SHOP")),
    )
    monkeypatch.setattr(broker_mod, "_db_conid_for_ticker", lambda _t: 2)
    assert (await broker_mod._default_conid_lookup("SHOP")) == 2
    # The other candidate would have won had the preference pointed at it: the
    # DB row, not the search order, decides.
    monkeypatch.setattr(broker_mod, "_db_conid_for_ticker", lambda _t: 1)
    assert (await broker_mod._default_conid_lookup("SHOP")) == 1


@pytest.mark.asyncio
async def test_default_conid_lookup_resolves_a_single_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One unambiguous candidate resolves to its conid, no DB preference needed."""
    monkeypatch.setattr(
        broker_mod, "search_contracts", lambda _t: _candidates(_contract("7", "AAPL"))
    )
    assert (await broker_mod._default_conid_lookup("AAPL")) == 7


# --- INV-1 fail-closed pre-flight / resync / attempt minting ---------------


@respx.mock
@pytest.mark.asyncio
async def test_failed_working_orders_read_sends_nothing() -> None:
    """INV-1/INV-3: a failed open-orders read is an ``Err`` and NO POST follows.

    A duplicate order is unbounded exposure; a skipped cycle is recoverable next
    bar because the intent is still desired.
    """
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(OPEN_ORDERS).mock(return_value=httpx.Response(500, json={}))
    error = _failure(await _broker().place(_open_intent()))
    assert error.kind in {"transport", "auth", "rate_limit"}
    assert submit.call_count == 0  # fail CLOSED: nothing sent


@respx.mock
@pytest.mark.asyncio
async def test_resync_adopts_a_prior_working_order_at_cycle_start() -> None:
    """INV-4: resync adopts an OPEN key's working order and persists WORKING."""
    intent = _open_intent()
    key = intent_key(SCOPE, intent)
    intents = FakeIntents()
    intents.save(
        IntentRecord(
            key=key,
            state=IntentState.WORKING,
            attempt=0,
            order_ref=order_ref(key, 0),
            order_id=ORDER_ID,
            decision_ts=None,
        )
    )
    respx.get(OPEN_ORDERS).mock(
        return_value=httpx.Response(
            200,
            json={"orders": [_captured_working_order(order_ref(key, 0))]},
        )
    )
    broker = _broker(intents=intents)

    adopted = await broker.resync()

    assert isinstance(adopted, Ok)
    results = cast("tuple[OrderResult, ...]", adopted.value)
    assert [r.outcome for r in results] == [OrderOutcome.ADOPTED]
    assert results[0].filled_qty is None  # the ask is not durably stored -> unknown
    record = intents.load(key)
    assert record is not None and record.state is IntentState.WORKING


@respx.mock
@pytest.mark.asyncio
async def test_adopted_working_order_reports_an_unknown_shortfall_not_a_zero() -> None:
    """D4: a PARTIAL adopted row must not fabricate a zero shortfall.

    The durable intent record stores no ask, so an adopted order's shortfall is
    genuinely unknown. The old row set ``filled_qty`` to the working row's partial
    fill against a synthetic intent of the SAME qty, so the shortfall was
    identically 0.0 — a half-filled working order rendered as "filled to ask".
    The row now reports the unknown as ``None``.
    """
    intent = _open_intent()
    key = intent_key(SCOPE, intent)
    intents = FakeIntents()
    intents.save(
        IntentRecord(
            key=key,
            state=IntentState.WORKING,
            attempt=0,
            order_ref=order_ref(key, 0),
            order_id=ORDER_ID,
            decision_ts=None,
        )
    )
    row = _captured_working_order(order_ref(key, 0))
    row["filledQuantity"] = 0.5  # a PARTIAL working order: half filled
    respx.get(OPEN_ORDERS).mock(
        return_value=httpx.Response(200, json={"orders": [row]})
    )

    adopted = await _broker(intents=intents).resync()

    assert isinstance(adopted, Ok)
    result = cast("tuple[OrderResult, ...]", adopted.value)[0]
    assert result.outcome is OrderOutcome.ADOPTED
    assert result.filled_qty is None  # unknown, never a fabricated 0.0


@respx.mock
@pytest.mark.asyncio
async def test_resync_resolves_a_known_id_with_no_working_order() -> None:
    """A vanished working order with a known id is settled by ``order_status``."""
    intent = _open_intent()
    key = intent_key(SCOPE, intent)
    intents = FakeIntents()
    intents.save(
        IntentRecord(
            key=key,
            state=IntentState.WORKING,
            attempt=0,
            order_ref=order_ref(key, 0),
            order_id=ORDER_ID,
            decision_ts=None,
        )
    )
    respx.get(OPEN_ORDERS).mock(return_value=httpx.Response(200, json={"orders": []}))
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))

    await _broker(intents=intents).resync()

    record = intents.load(key)
    assert record is not None and record.state is IntentState.FILLED


@respx.mock
@pytest.mark.asyncio
async def test_a_new_attempt_is_minted_only_after_the_prior_intent_closed() -> None:
    """INV-4: an OPEN record with a known id and no working order does NOT re-mint."""
    intent = _open_intent()
    key = intent_key(SCOPE, intent)
    intents = FakeIntents()
    intents.save(
        IntentRecord(
            key=key,
            state=IntentState.WORKING,
            attempt=0,
            order_ref=order_ref(key, 0),
            order_id=ORDER_ID,
            decision_ts=None,
        )
    )
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(OPEN_ORDERS).mock(return_value=httpx.Response(200, json={"orders": []}))
    error = _failure(await _broker(intents=intents).place(intent))
    assert error.kind == "unresolved"
    assert submit.call_count == 0  # no re-mint while the intent is still OPEN


@respx.mock
@pytest.mark.asyncio
async def test_a_closed_prior_intent_mints_a_new_attempt() -> None:
    """A terminal prior record lets the same key submit a fresh attempt."""
    intent = _open_intent()
    key = intent_key(SCOPE, intent)
    intents = FakeIntents()
    intents.save(
        IntentRecord(
            key=key,
            state=IntentState.UNFILLED,
            attempt=0,
            order_ref=order_ref(key, 0),
            order_id=ORDER_ID,
            decision_ts=None,
        )
    )
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))
    _mock_no_working_orders()

    placed = _ok(await _broker(intents=intents).place(intent))

    assert placed.ok
    record = intents.load(key)
    assert record is not None and record.attempt == 1  # a NEW attempt, new cOID


@respx.mock
@pytest.mark.asyncio
async def test_open_cash_guard_bounds_the_rounded_ticket_quantity() -> None:
    """B1: a bound of 1050 at price 300 refuses qty 3.5 (which rounds to 4 = $1200)."""
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    intent = OrderIntent(
        symbol="AAPL",
        action=ActionType.long,
        qty=3.5,
        ref_price=300.0,
        reason="open long (flat->long)",
        cash_bound=1050.0,
    )
    error = _failure(await _broker().place(intent))
    assert error.kind == "rejected"
    assert "exceeds funded cash" in error.message
    assert submit.call_count == 0  # the $1200 order was never sent


# --- blocker 1: UNRESOLVED must not re-mint; the executions sweep guards it ---


@respx.mock
@pytest.mark.asyncio
async def test_unresolved_without_an_order_id_never_submits() -> None:
    """Block1: an UNRESOLVED record with no id is NOT proof it was never placed.

    ``/iserver/account/orders`` lists WORKING orders, so absence cannot tell\n    \"never sent\" from \"sent and filled\"; the record must not re-mint a duplicate.\n"""
    intent = _open_intent()
    key = intent_key(SCOPE, intent)
    intents = FakeIntents()
    intents.records[(key.scope, key.token())] = IntentRecord(
        key=key,
        state=IntentState.UNRESOLVED,
        attempt=0,
        order_ref=order_ref(key, 0),
        order_id=None,
        decision_ts=None,
    )
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(OPEN_ORDERS).mock(return_value=httpx.Response(200, json={"orders": []}))

    error = _failure(await _broker(intents=intents).place(intent))

    assert error.kind == "unresolved"
    assert submit.call_count == 0  # no new attempt while the state is unknown


@respx.mock
@pytest.mark.asyncio
async def test_lost_intent_row_does_not_mint_when_the_executions_window_has_the_fill() -> (
    None
):
    """Block1 sweep: the executions channel protects a lost durable row.

    No durable record exists, but the trades window already carries a fill for\n    this key — a submit would be a duplicate, so the broker finds it via the\n    ref (bar-free) and refuses to mint.\n"""
    intent = _open_intent()
    key = intent_key(SCOPE, intent)
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(OPEN_ORDERS).mock(return_value=httpx.Response(200, json={"orders": []}))
    respx.get(TRADES).mock(
        return_value=httpx.Response(
            200,
            json={
                "trades": [
                    {
                        "execution_id": "e1",
                        "order_id": "OLD1",
                        "order_ref": order_ref(key, 0),
                        "conid": str(CONID),
                        "side": "BUY",
                        "size": "1",
                        "price": "100",
                        "trade_time_r": "1717439400000",
                        "symbol": "AAPL",
                    }
                ]
            },
        )
    )

    error = _failure(await _broker().place(intent))

    assert error.kind == "unresolved"
    assert submit.call_count == 0  # no duplicate minted


@respx.mock
@pytest.mark.asyncio
async def test_lost_intent_row_sweep_fails_closed_when_the_executions_read_fails() -> (
    None
):
    """Block1 sweep: a failed executions read blocks the mint (INV-1, fail closed)."""
    intent = _open_intent()
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(OPEN_ORDERS).mock(return_value=httpx.Response(200, json={"orders": []}))
    respx.get(TRADES).mock(return_value=httpx.Response(500, json={}))

    error = _failure(await _broker().place(intent))

    assert error.kind in {"transport", "auth", "rate_limit"}
    assert submit.call_count == 0  # a duplicate is unbounded exposure


# --- blocker 2: a deadline partial settles WORKING, never a terminal state ---


@respx.mock
@pytest.mark.asyncio
async def test_deadline_partial_settles_working_not_terminal() -> None:
    """Block2/N3: a partial at the wait deadline keeps the intent WORKING.

    The order is STILL LIVE, so stamping it UNFILLED (terminal) would let the next\n    cycle re-mint a duplicate. The reported OUTCOME must agree with that durable\n    state — non-terminal vocabulary, never ``unfilled``/``rejected`` — so an\n    operator does not read a live order as dead.\n"""
    intent = _open_intent()
    key = intent_key(SCOPE, intent)
    intents = FakeIntents()
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
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
    _mock_no_working_orders()
    broker = _broker(intents=intents)

    result = await broker.place_cohort((intent,))

    assert isinstance(result, Ok)
    (failed,) = cast("tuple[OrderResult, ...]", result.value)
    assert failed.ok is False
    assert failed.outcome not in (OrderOutcome.UNFILLED, OrderOutcome.REJECTED)
    assert failed.filled_qty == 0.25
    record = intents.load(key)
    assert record is not None and record.state is IntentState.WORKING


@respx.mock
@pytest.mark.asyncio
async def test_a_dead_order_carries_its_partial_fill() -> None:
    """A terminal partial reports the shares it filled, not just a message.

    A dead DAY order that filled only what the venue had leaves the position under
    the sizer's target; since the posture diff compares sides and never sizes, the
    residual is never chased (decision A). ``filled_qty`` on the result is what
    makes that shortfall visible to an operator and to the report renderer.
    """
    intent = _open_intent()
    intents = FakeIntents()
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
                "average_price": "101.0",
            },
        )
    )
    _mock_no_working_orders()
    broker = _broker(intents=intents)

    result = await broker.place_cohort((intent,))

    assert isinstance(result, Ok)
    (failed,) = cast("tuple[OrderResult, ...]", result.value)
    assert failed.ok is False
    assert failed.outcome is OrderOutcome.UNFILLED
    assert failed.filled_qty == 0.5


# --- blocker 1: resync ages an unresolved-no-id record out on its day roll ----


@respx.mock
@pytest.mark.asyncio
async def test_resync_ages_out_an_unresolved_no_id_record_on_its_day_roll() -> None:
    """Block1: an UNRESOLVED record with no id is aged to UNFILLED on a day roll.

    Before the rollover no evidence may age it out (the order might still be\n    live); once its DAY order's calendar day passes and nothing is working, it has\n    expired unfilled.\n"""
    intent = _open_intent()
    key = intent_key(SCOPE, intent)
    intents = FakeIntents()
    decision = cast("pd.Timestamp", pd.Timestamp("2024-06-03T20:00:00Z"))
    intents.records[(key.scope, key.token())] = IntentRecord(
        key=key,
        state=IntentState.UNRESOLVED,
        attempt=0,
        order_ref=order_ref(key, 0),
        order_id=None,
        decision_ts=decision,
    )
    respx.get(OPEN_ORDERS).mock(return_value=httpx.Response(200, json={"orders": []}))
    broker = _broker(
        intents=intents,
        now=lambda: cast("pd.Timestamp", pd.Timestamp("2024-06-05T09:00:00Z")),
    )

    await broker.resync()

    record = intents.load(key)
    assert record is not None and record.state is IntentState.UNFILLED


# --- integration: the broker over the REAL SqliteLedger seam -----------------


@respx.mock
@pytest.mark.asyncio
async def test_broker_places_over_the_real_sqlite_ledger(tmp_path: Path) -> None:
    """Integration: ``IbkrBroker`` drives the real ``SqliteLedger`` seam end to end.

    This exercises the seam's identity keying and state persistence that\n    ``FakeIntents`` re-implements without (a filled order lands in sqlite and is\n    read back as the broker's FILLED intent).\n"""
    intent = _open_intent()
    key = intent_key(SCOPE, intent)
    ledger = SqliteLedger(tmp_path / "intents.sqlite")
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))
    _mock_no_working_orders()

    placed = _ok(await _broker(intents=ledger).place(intent))

    assert placed.ok
    # The durable FILLED record is readable back from sqlite by its identity key.
    record = ledger.load(key)
    assert record is not None and record.state is IntentState.FILLED
    assert record.order_id == ORDER_ID
    assert record.order_ref == order_ref(key, 0)
    assert ledger.load_open(SCOPE) == ()


# --- N4: a dead same-session order must not block a legitimate re-send -------


@respx.mock
@pytest.mark.asyncio
async def test_a_cancelled_same_session_order_does_not_block_a_resend() -> None:
    """N4: ``/iserver/account/orders`` lists cancelled session orders.

    The parsed status is terminal, so the row must be skipped when building the
    working index — otherwise it is adopted and the broker waits on a dead order,
    blocking the legitimate re-send for the rest of the session.
    """
    intent = _open_intent()
    key = intent_key(SCOPE, intent)
    intents = FakeIntents()
    intents.save(
        IntentRecord(
            key=key,
            state=IntentState.UNFILLED,
            attempt=0,
            order_ref=order_ref(key, 0),
            order_id="OLD",
            decision_ts=None,
        )
    )
    cancelled = _captured_working_order(order_ref(key, 0))
    cancelled["status"] = "Cancelled"
    cancelled["order_ccp_status"] = "Cancelled"
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(OPEN_ORDERS).mock(
        return_value=httpx.Response(200, json={"orders": [cancelled]})
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))

    placed = _ok(await _broker(intents=intents).place(intent))

    assert placed.ok
    assert submit.called  # the dead order did NOT block the re-send
    record = intents.load(key)
    assert record is not None and record.attempt == 1  # a NEW attempt


# --- N1: a status read failure must not wedge a key forever ------------------


@respx.mock
@pytest.mark.asyncio
async def test_status_read_failure_settles_a_rolled_day_order_unfilled() -> None:
    """N1: the status endpoint only covers the CURRENT session (503 otherwise).

    A DAY order cannot outlive its session, so once its day has rolled and the
    status cannot be read it has expired unfilled — settle it (re-mintable) rather
    than leaving the record immortal.
    """
    intent = _open_intent()
    key = intent_key(SCOPE, intent)
    intents = FakeIntents()
    intents.save(
        IntentRecord(
            key=key,
            state=IntentState.WORKING,
            attempt=0,
            order_ref=order_ref(key, 0),
            order_id=ORDER_ID,
            decision_ts=cast("pd.Timestamp", pd.Timestamp("2024-06-03T20:00:00Z")),
        )
    )
    respx.get(OPEN_ORDERS).mock(return_value=httpx.Response(200, json={"orders": []}))
    respx.get(STATUS).mock(side_effect=httpx.ConnectError("down"))

    broker = _broker(
        intents=intents,
        now=lambda: cast("pd.Timestamp", pd.Timestamp("2024-06-05T09:00:00Z")),
    )
    await broker.resync()

    record = intents.load(key)
    assert record is not None and record.state is IntentState.UNFILLED


@respx.mock
@pytest.mark.asyncio
async def test_status_read_failure_same_day_keeps_working() -> None:
    """N1: a same-day unreadable status is NOT proof the order is gone."""
    intent = _open_intent()
    key = intent_key(SCOPE, intent)
    intents = FakeIntents()
    intents.save(
        IntentRecord(
            key=key,
            state=IntentState.WORKING,
            attempt=0,
            order_ref=order_ref(key, 0),
            order_id=ORDER_ID,
            decision_ts=CYCLE_TS,
        )
    )
    respx.get(OPEN_ORDERS).mock(return_value=httpx.Response(200, json={"orders": []}))
    respx.get(STATUS).mock(side_effect=httpx.ConnectError("down"))

    await _broker(intents=intents, now=lambda: CYCLE_TS).resync()

    record = intents.load(key)
    assert record is not None and record.state is IntentState.WORKING


@respx.mock
@pytest.mark.asyncio
async def test_a_wedged_key_is_surfaced_distinctly_after_n_resyncs() -> None:
    """N1: an OPEN key nothing can settle across WEDGED_CYCLES resyncs is LOUD.

    A distinct ``WEDGED`` outcome in the report (not one more ``unresolved`` line)
    names the operator escape, so a broker-side cancel the gateway cannot confirm
    does not sit silent forever.
    """
    intent = _open_intent()
    key = intent_key(SCOPE, intent)
    intents = FakeIntents()
    intents.save(
        IntentRecord(
            key=key,
            state=IntentState.WORKING,
            attempt=0,
            order_ref=order_ref(key, 0),
            order_id=ORDER_ID,
            decision_ts=CYCLE_TS,
        )
    )
    respx.get(OPEN_ORDERS).mock(return_value=httpx.Response(200, json={"orders": []}))
    respx.get(STATUS).mock(side_effect=httpx.ConnectError("down"))
    broker = _broker(intents=intents, now=lambda: CYCLE_TS)

    for _ in range(WEDGED_CYCLES - 1):
        quiet = await broker.resync()
        assert isinstance(quiet, Ok)
        assert not [
            r
            for r in cast("tuple[OrderResult, ...]", quiet.value)
            if r.outcome is OrderOutcome.WEDGED
        ]
    loud = await broker.resync()

    assert isinstance(loud, Ok)
    wedged = [
        r
        for r in cast("tuple[OrderResult, ...]", loud.value)
        if r.outcome is OrderOutcome.WEDGED
    ]
    assert len(wedged) == 1 and "abandon" in wedged[0].message
    record = intents.load(key)
    assert record is not None and record.stuck_cycles == WEDGED_CYCLES


@respx.mock
@pytest.mark.asyncio
async def test_a_gtc_order_is_not_expired_by_a_day_roll() -> None:
    """N1: the day-roll expiry is gated on the persisted ``tif`` (a GTC survives)."""
    intent = _open_intent()
    key = intent_key(SCOPE, intent)
    intents = FakeIntents()
    intents.save(
        IntentRecord(
            key=key,
            state=IntentState.WORKING,
            attempt=0,
            order_ref=order_ref(key, 0),
            order_id=ORDER_ID,
            decision_ts=cast("pd.Timestamp", pd.Timestamp("2024-06-03T20:00:00Z")),
            tif="GTC",
        )
    )
    respx.get(OPEN_ORDERS).mock(return_value=httpx.Response(200, json={"orders": []}))
    respx.get(STATUS).mock(side_effect=httpx.ConnectError("down"))

    broker = _broker(
        intents=intents,
        now=lambda: cast("pd.Timestamp", pd.Timestamp("2024-06-05T09:00:00Z")),
    )
    await broker.resync()

    record = intents.load(key)
    assert record is not None and record.state is IntentState.WORKING


# --- N5: a falsy order id is UNKNOWN, never a real id ------------------------


@respx.mock
@pytest.mark.asyncio
async def test_resync_treats_an_empty_order_id_as_unknown() -> None:
    """N5: ``canonical_order_id(None)`` is ``''``; a falsy id must be unknown.

    Otherwise the record never ages and ``_status('')`` errors every cycle.
    """
    intent = _open_intent()
    key = intent_key(SCOPE, intent)
    intents = FakeIntents()
    intents.save(
        IntentRecord(
            key=key,
            state=IntentState.UNRESOLVED,
            attempt=0,
            order_ref=order_ref(key, 0),
            order_id="",
            decision_ts=None,
        )
    )
    respx.get(OPEN_ORDERS).mock(return_value=httpx.Response(200, json={"orders": []}))
    status = respx.get(STATUS).mock(return_value=httpx.Response(200, json={}))

    await _broker(intents=intents).resync()

    assert not status.called  # no status read for an empty id


@respx.mock
@pytest.mark.asyncio
async def test_a_working_row_with_no_readable_order_id_is_not_adopted() -> None:
    """N5: a working row we cannot join (no id) is skipped, never adopted."""
    intent = _open_intent()
    row = _captured_working_order(order_ref(intent_key(SCOPE, intent), 0))
    del row["orderId"]
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(OPEN_ORDERS).mock(
        return_value=httpx.Response(200, json={"orders": [row]})
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))

    placed = _ok(await _broker().place(intent))

    assert placed.ok and submit.called


# --- N6: the skip branch must not downgrade a WORKING record -----------------


@respx.mock
@pytest.mark.asyncio
async def test_skip_does_not_downgrade_a_working_record() -> None:
    """N6: a WORKING record's provenance (ref/id) must survive a skip, not be erased."""
    intent = _open_intent()
    key = intent_key(SCOPE, intent)
    intents = FakeIntents()
    intents.save(
        IntentRecord(
            key=key,
            state=IntentState.WORKING,
            attempt=0,
            order_ref=order_ref(key, 0),
            order_id=ORDER_ID,
            decision_ts=None,
        )
    )
    respx.get(OPEN_ORDERS).mock(return_value=httpx.Response(200, json={"orders": []}))

    error = _failure(await _broker(intents=intents).place(intent))

    assert error.kind == "unresolved"
    record = intents.load(key)
    assert record is not None and record.state is IntentState.WORKING  # not erased


# --- N8: the durable attempt never regresses ---------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_adopting_an_older_working_order_does_not_lower_the_attempt() -> None:
    """N8: ``_persist_working`` must not persist an attempt BELOW the live record."""
    intent = _open_intent()
    key = intent_key(SCOPE, intent)
    intents = FakeIntents()
    intents.save(
        IntentRecord(
            key=key,
            state=IntentState.WORKING,
            attempt=5,
            order_ref=order_ref(key, 5),
            order_id=ORDER_ID,
            decision_ts=None,
        )
    )
    respx.get(OPEN_ORDERS).mock(
        return_value=httpx.Response(
            200,
            json={"orders": [_captured_working_order(order_ref(key, 0))]},
        )
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=_filled_status()))

    await _broker(intents=intents).resync()

    record = intents.load(key)
    assert record is not None and record.attempt == 5


# --- N3: the reported outcome follows the durable state, never a bare kind ---


@respx.mock
@pytest.mark.asyncio
async def test_status_transport_failure_reports_unresolved_not_rejected() -> None:
    """N3: a transport failure while polling a LIVE order is never 'rejected'.

    The durable state is WORKING (the order may be live), so the reported outcome
    maps transport/auth/rate_limit to the unknown/unresolved vocabulary.
    """
    intent = _open_intent()
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(STATUS).mock(side_effect=httpx.ConnectError("down"))
    _mock_no_working_orders()
    broker = _broker(intents=FakeIntents())

    result = await broker.place_cohort((intent,))

    assert isinstance(result, Ok)
    (failed,) = cast("tuple[OrderResult, ...]", result.value)
    assert failed.ok is False
    assert failed.outcome is OrderOutcome.UNRESOLVED
    assert failed.error_kind in {"transport", "auth", "rate_limit"}
