"""``IbkrBroker`` clean placements: submit -> reply -> confirm, then fill wait.

The critical-path assertions: a clean submit mints the PURE single-arg ref (the
broker's own ``cOID`` dedupe then absorbs a duplicate), and every terminal wait
branch reports its honest outcome — never a silent success.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from src.live.adapters.ibkr.tests.conftest import (
    BASE,
    CONID,
    ORDER_ID,
    SCOPE,
    STATUS,
    SUBMIT,
    broker,
    cohort_ok,
    failure,
    filled_status,
    mock_no_working_orders,
    ok_result,
    open_intent,
)
from src.live.identity import intent_key, order_ref


@respx.mock
@pytest.mark.asyncio
async def test_place_submits_then_waits_for_the_fill() -> None:
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(
            200, json=[{"order_id": ORDER_ID, "order_status": "PreSubmitted"}]
        )
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=filled_status()))
    mock_no_working_orders()
    intent = open_intent()

    placed = ok_result(await broker().place(intent))

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
    # The PURE single-arg ref: same key always mints the same cOID, no attempt tail.
    assert ticket["cOID"] == order_ref(intent_key(SCOPE, intent))


@respx.mock
@pytest.mark.asyncio
async def test_complete_fill_ahead_of_the_status_flip_is_a_fill() -> None:
    """A filled order whose status has not flipped is a FILL, not an unfilled timeout.

    ``cum_fill == total_size`` while the order still reads ``Submitted``: a fill
    cannot exceed the size, so this is final — no poll to the deadline, no
    "unfilled ... 1 of 1 filled".
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
    mock_no_working_orders()

    placed = ok_result(await broker().place(open_intent()))

    assert placed.ok and placed.fill is not None
    assert (placed.fill.filled_qty, placed.fill.executed_price) == (1.0, 100.5)


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
    mock_no_working_orders()

    error = failure(await broker().place(open_intent()))

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
    mock_no_working_orders()

    error = failure(await broker().place(open_intent()))

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
    mock_no_working_orders()

    error = failure(await broker().place(open_intent()))

    assert error.kind == "timeout"
    assert "no terminal status" in error.message

    # Same but with a partial: the filled qty is reported, the kind stays `timeout`
    # (still live at the deadline, not a dead order).
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
    partial = failure(await broker().place(open_intent()))
    assert partial.kind == "timeout"
    assert "still Submitted after 0s with 0.25 of 1 filled" in partial.message


@respx.mock
@pytest.mark.asyncio
async def test_terminal_filled_without_a_readable_cum_fill_is_unresolved() -> None:
    """A terminal ``Filled`` with a garbled ``cum_fill`` is NOT "nothing filled".

    The order DID fill; only its size is unreadable, so it is reported unknown —
    never a zero-fill rejection an operator might answer by re-placing by hand.
    """
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(STATUS).mock(
        return_value=httpx.Response(
            200, json={"order_status": "Filled", "total_size": "1"}
        )
    )
    mock_no_working_orders()

    error = failure(await broker().place(open_intent()))

    assert error.kind == "unresolved"
    assert "nothing filled" not in error.message
    assert "unknown" in error.message


@respx.mock
@pytest.mark.asyncio
async def test_status_transport_failure_reports_unresolved_not_rejected() -> None:
    """A transport failure while polling a LIVE order is never 'rejected'.

    The order may be live, so the reported outcome maps transport/auth/rate_limit
    to the unknown/unresolved vocabulary — never a refusal an operator answers by
    re-placing.
    """
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(STATUS).mock(side_effect=httpx.ConnectError("down"))
    mock_no_working_orders()

    result = await broker().place_cohort((open_intent(),))

    (failed,) = cohort_ok(result)
    assert failed.ok is False
    assert failed.outcome.value == "unresolved"
    assert failed.error_kind in {"transport", "auth", "rate_limit"}


@respx.mock
@pytest.mark.asyncio
async def test_reject_prompt_aborts_without_confirming() -> None:
    """A refusal BEFORE any confirmation cannot have been submitted -> rejected."""
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
    confirm = respx.post(url__regex=rf"{BASE}iserver/reply/.*").mock(
        return_value=httpx.Response(500, json={"error": "must not be called"})
    )
    status = respx.get(STATUS).mock(return_value=httpx.Response(200, json={}))
    mock_no_working_orders()

    error = failure(await broker().place(open_intent()))

    assert error.kind == "rejected"
    assert "price band exceeded" in error.message
    assert not confirm.called  # nothing further was placed
    assert not status.called  # we never waited on a non-existent order


@respx.mock
@pytest.mark.asyncio
async def test_dry_run_broker_never_posts() -> None:
    error = failure(await broker(dry_run=True).place(open_intent()))

    assert error.kind == "auth"
    assert "refuses to place" in error.message
    assert len(respx.calls) == 0  # not a single request left the process
