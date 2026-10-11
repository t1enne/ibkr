"""``IbkrBroker`` pre-submit guards: fail-closed reads and deadly bounds.

These source paths (``_pre_guard`` / ``_open_exposure_guard`` / ``_close_guard``
/ ``_open_cash_guard``) are unchanged by the refactor and still deadly. Every
one of them fails CLOSED: a read we cannot trust sends nothing, and a bound the
account cannot absorb refuses rather than flips a position.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from src.exec.types import OrderType
from src.live.adapters.ibkr.tests.conftest import (
    BASE,
    CONID,
    ORDER_ID,
    STATUS,
    SUBMIT,
    TRADES,
    FakeExposure,
    broker,
    close_intent,
    cohort_ok,
    failure,
    filled_status,
    long_lot,
    mock_no_working_orders,
    ok_result,
    open_intent,
)
from src.live.identity import OrderOutcome, intent_key, order_ref
from src.live.report import _UNSAFE_OUTCOMES
from src.live.types import OrderIntent
from src.bt.state import ActionType

OPEN_ORDERS = f"{BASE}iserver/account/orders"
POSITIONS = f"{BASE}portfolio/DU452563/positions/0"


@respx.mock
@pytest.mark.asyncio
async def test_failed_working_orders_read_sends_nothing() -> None:
    """INV-1: a failed open-orders read is an ``Err`` and NO POST follows.

    A duplicate order is unbounded exposure; a skipped cycle is recoverable next
    bar because the intent is still desired.
    """
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(OPEN_ORDERS).mock(return_value=httpx.Response(500, json={}))

    error = failure(await broker().place(open_intent()))

    assert error.kind in {"transport", "auth", "rate_limit"}
    assert submit.call_count == 0  # fail CLOSED: nothing sent


@respx.mock
@pytest.mark.asyncio
async def test_executions_sweep_refuses_to_remint_a_filled_predecessor() -> None:
    """The executions sweep protects a lost key: a fill for it blocks a duplicate.

    No durable record exists, but the trades window already carries a fill for
    this key — a submit would be a duplicate, so the broker finds it via the
    (bar-free) ref and refuses to mint.
    """
    intent = open_intent()
    key = intent_key("momentum", intent)
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
                        "order_ref": order_ref(key),
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

    error = failure(await broker().place(intent))

    assert error.kind == "unresolved"
    assert submit.call_count == 0  # no duplicate minted


@respx.mock
@pytest.mark.asyncio
async def test_executions_sweep_fails_closed_when_the_read_fails() -> None:
    """A failed executions read blocks the mint (INV-1, fail closed)."""
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(OPEN_ORDERS).mock(return_value=httpx.Response(200, json={"orders": []}))
    respx.get(TRADES).mock(return_value=httpx.Response(500, json={}))

    error = failure(await broker().place(open_intent()))

    assert error.kind in {"transport", "auth", "rate_limit"}
    assert submit.call_count == 0  # a duplicate is unbounded exposure


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

    error = failure(await broker().place(intent))

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

    error = failure(await broker().place(intent))

    assert error.kind == "rejected"
    assert "naked order" in error.message
    assert "95.0" in error.message and "110.0" in error.message
    assert len(respx.calls) == 0


@respx.mock
@pytest.mark.asyncio
async def test_open_over_deploying_its_cash_bound_is_refused() -> None:
    """An open whose notional exceeds the scope's funded cash is refused.

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

    error = failure(await broker().place(intent))

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

    error = failure(await broker().place(intent))

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
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=filled_status()))
    mock_no_working_orders()
    intent = OrderIntent(
        symbol="AAPL",
        action=ActionType.long,
        qty=10.0,
        ref_price=100.0,
        reason="open long (flat->long)",
        cash_bound=1000.0,
    )

    placed = ok_result(await broker().place(intent))

    assert placed.ok


@respx.mock
@pytest.mark.asyncio
async def test_open_diverging_from_the_account_net_is_refused() -> None:
    """A lagged executions feed leaves an account net our book cannot explain.

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

    error = failure(await broker().place(open_intent()))

    assert error.kind == "divergence"
    assert submit.call_count == 0  # never sent


@respx.mock
@pytest.mark.asyncio
async def test_open_refused_when_our_book_is_ahead_of_the_account() -> None:
    """B inverse: a phantom lot (book ahead of the account) is refused too."""
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    wired = broker(exposure=FakeExposure(5.0))  # book long 5

    error = failure(await wired.place(open_intent()))  # account flat

    assert error.kind == "divergence"
    assert submit.call_count == 0


@respx.mock
@pytest.mark.asyncio
async def test_close_on_a_flat_account_is_refused_before_submitting() -> None:
    """Never reduce a conid the account is flat on.

    The refusal is a BOOK DIVERGENCE (our book holds a lot the account does not),
    so it surfaces as ``divergence`` — an unsafe cycle — not a broker
    ``rejected`` (which the exit contract treats as safe).
    """
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(POSITIONS).mock(return_value=httpx.Response(200, json=[]))  # flat
    wired = broker()
    wired.seed(long_lot("555000111"))

    error = failure(await wired.place(close_intent()))

    assert "flat" in error.message
    assert error.kind == "divergence"  # unsafe, not a safe "rejected"
    assert submit.call_count == 0  # never sent


@respx.mock
@pytest.mark.asyncio
async def test_close_larger_than_a_shared_net_is_refused_before_submitting() -> None:
    """A close on a SHARED account must not exceed the account net.

    Our book holds long 60, but another scope/human holds 40 short on the same
    conid, so the account nets to 20. A SELL 60 would flip the account to -40 —
    an unintended naked short. The refusal is a book divergence.
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
    mock_no_working_orders()
    wired = broker()
    wired.seed(long_lot("555000111", qty=60.0))

    error = failure(await wired.place(close_intent(qty=60.0)))

    assert "exceeds the account net" in error.message
    assert error.kind == "divergence"  # unsafe, not a safe "rejected"
    assert submit.call_count == 0  # the flip is refused, nothing sent


@respx.mock
@pytest.mark.asyncio
async def test_close_against_the_opposite_net_side_is_refused() -> None:
    """A SELL cannot reduce a short net — it must be refused, not flipped."""
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(POSITIONS).mock(
        return_value=httpx.Response(
            200, json=[{"conid": 265598, "contractDesc": "AAPL", "position": -5}]
        )
    )
    mock_no_working_orders()
    wired = broker()
    wired.seed(long_lot("555000111", qty=1.0))

    error = failure(await wired.place(close_intent()))

    assert "opposite side" in error.message
    assert error.kind == "divergence"  # unsafe, not a safe "rejected"
    assert submit.call_count == 0


@respx.mock
@pytest.mark.asyncio
async def test_a_refused_close_reports_the_unsafe_divergence_outcome() -> None:
    """End to end through a cohort: the refusal carries ``DIVERGENCE``.

    ``place_cohort`` maps the guard's kind to an outcome; a close the account net
    cannot absorb must yield ``OrderOutcome.DIVERGENCE`` (in the engine's unsafe
    set) so the cycle exits non-zero — the strategy cannot exit, and cron must see
    it.
    """
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(POSITIONS).mock(
        return_value=httpx.Response(
            200,
            json=[
                {"conid": CONID, "contractDesc": "AAPL", "position": 60},
                {"conid": CONID, "contractDesc": "AAPL", "position": -40},
            ],
        )
    )
    mock_no_working_orders()
    wired = broker()
    wired.seed(long_lot("555000111", qty=60.0))

    result = await wired.place_cohort((close_intent(qty=60.0),))

    (close_result,) = cohort_ok(result)
    assert close_result.ok is False
    assert close_result.error_kind == "divergence"
    assert close_result.outcome is OrderOutcome.DIVERGENCE
    assert OrderOutcome.DIVERGENCE in _UNSAFE_OUTCOMES
    assert submit.call_count == 0  # nothing was sent against an unexplained net
