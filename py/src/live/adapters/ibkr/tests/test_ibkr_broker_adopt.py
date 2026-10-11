"""``IbkrBroker`` adoption: a working order for our key is USED, never re-sent.

Broker-truth dedupe: the working-orders read is the authority. A row carrying
our pure ``ref_prefix(key)`` means the order already exists — wait on it
(``ADOPTED``), no POST. The same holds for a POST that errored but landed, a
confirm that would duplicate a live order, and a reply-loop overflow.
"""

from __future__ import annotations

from typing import cast

import httpx
import pandas as pd
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
    captured_working_order,
    cohort_ok,
    filled_status,
    ok_result,
    open_intent,
)
from src.live.identity import OrderOutcome, intent_key, order_ref


@respx.mock
@pytest.mark.asyncio
async def test_working_order_for_our_key_is_adopted_not_resubmitted() -> None:
    """A working order carrying our pure ref means the order EXISTS — adopt it."""
    intent = open_intent()
    ref = order_ref(intent_key(SCOPE, intent))
    submit = respx.post(SUBMIT).mock(return_value=httpx.Response(500, json={}))
    respx.get(f"{BASE}iserver/account/orders").mock(
        return_value=httpx.Response(200, json={"orders": [captured_working_order(ref)]})
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=filled_status()))

    placed = ok_result(await broker().place(intent))

    assert placed.ok and placed.position_id == str(CONID)
    assert placed.outcome is OrderOutcome.ADOPTED
    assert not submit.called  # adopted, never re-sent


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
    prior_ref = order_ref(intent_key(SCOPE, open_intent()))
    submit = respx.post(SUBMIT).mock(return_value=httpx.Response(500, json={}))
    respx.get(f"{BASE}iserver/account/orders").mock(
        return_value=httpx.Response(
            200, json={"orders": [captured_working_order(prior_ref)]}
        )
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=filled_status()))
    wired = broker(
        now=lambda: cast("pd.Timestamp", pd.Timestamp("2024-06-05T09:00:00Z"))
    )

    placed = ok_result(await wired.place(open_intent(decision_ts=late_bar)))

    assert placed.ok and placed.position_id == str(CONID)
    assert placed.outcome is OrderOutcome.ADOPTED
    assert not submit.called  # the earlier-bar order was adopted, not re-sent
    assert early_bar < late_bar  # the bars genuinely differ


@respx.mock
@pytest.mark.asyncio
async def test_ambiguous_submit_adopts_a_working_order() -> None:
    """A POST that errored but landed: the working order is SEEN, not re-sent."""
    intent = open_intent()
    ref = order_ref(intent_key(SCOPE, intent))
    respx.post(SUBMIT).mock(side_effect=httpx.ConnectTimeout("down"))
    # The pre-flight sees nothing (the POST has not landed yet); the post-error
    # re-check finds the order the failed POST actually placed.
    respx.get(f"{BASE}iserver/account/orders").mock(
        side_effect=[
            httpx.Response(200, json={"orders": []}),
            httpx.Response(200, json={"orders": [captured_working_order(ref)]}),
        ]
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=filled_status()))

    placed = ok_result(await broker().place(intent))

    assert placed.ok and placed.position_id == str(CONID)


@respx.mock
@pytest.mark.asyncio
async def test_confirm_post_adopts_when_a_working_order_appears() -> None:
    """INV-1: before the confirm POST a working order for our key is ADOPTED, not confirmed."""
    intent = open_intent()
    ref = order_ref(intent_key(SCOPE, intent))
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(
            200, json=[{"id": "r0", "message": ["confirm?"], "messageIds": ["o1"]}]
        )
    )
    confirm = respx.post(f"{BASE}iserver/reply/r0").mock(
        return_value=httpx.Response(500, json={"error": "must not be called"})
    )
    respx.get(f"{BASE}iserver/account/orders").mock(
        side_effect=[
            httpx.Response(200, json={"orders": []}),  # pre-flight
            httpx.Response(200, json={"orders": [captured_working_order(ref)]}),
        ]
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=filled_status()))

    placed = ok_result(await broker().place(intent))

    assert placed.ok
    assert not confirm.called  # the working order was adopted, not re-confirmed


@respx.mock
@pytest.mark.asyncio
async def test_reply_loop_overflow_adopts_a_live_order() -> None:
    """The last confirm may have submitted the order: a working one is ADOPTED."""
    intent = open_intent()
    ref = order_ref(intent_key(SCOPE, intent))
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(
            200, json=[{"id": "r0", "message": ["again?"], "messageIds": ["o1"]}]
        )
    )
    respx.post(url__regex=rf"{BASE}iserver/reply/.*").mock(
        return_value=httpx.Response(
            200, json=[{"id": "r-next", "message": ["again?"], "messageIds": ["o1"]}]
        )
    )
    respx.get(f"{BASE}iserver/account/orders").mock(
        side_effect=[
            httpx.Response(200, json={"orders": []}),  # pre-flight: nothing yet
            httpx.Response(200, json={"orders": [captured_working_order(ref)]}),
        ]
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=filled_status()))

    placed = ok_result(await broker().place(intent))

    assert placed.ok and placed.position_id == str(CONID)


@respx.mock
@pytest.mark.asyncio
async def test_a_cancelled_same_session_order_does_not_block_a_resend() -> None:
    """``/iserver/account/orders`` lists cancelled session orders.

    The parsed status is terminal, so the row must be skipped when building the
    working index — otherwise it is adopted and the broker waits on a dead order,
    blocking the legitimate re-send for the rest of the session.
    """
    intent = open_intent()
    ref = order_ref(intent_key(SCOPE, intent))
    cancelled = captured_working_order(ref)
    cancelled["status"] = "Cancelled"
    cancelled["order_ccp_status"] = "Cancelled"
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(f"{BASE}iserver/account/orders").mock(
        return_value=httpx.Response(200, json={"orders": [cancelled]})
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=filled_status()))

    placed = ok_result(await broker().place(intent))

    assert placed.ok
    assert submit.called  # the dead order did NOT block the re-send


@respx.mock
@pytest.mark.asyncio
async def test_a_working_row_with_no_readable_order_id_is_not_adopted() -> None:
    """A working row we cannot join (no id) is skipped, never adopted."""
    intent = open_intent()
    row = captured_working_order(order_ref(intent_key(SCOPE, intent)))
    del row["orderId"]
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(f"{BASE}iserver/account/orders").mock(
        return_value=httpx.Response(200, json={"orders": [row]})
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=filled_status()))

    placed = ok_result(await broker().place(intent))

    assert placed.ok and submit.called


@respx.mock
@pytest.mark.asyncio
async def test_resync_is_a_noop() -> None:
    """The broker is the source of truth: a cycle-start resync reconciles nothing."""
    result = await broker().resync()

    assert cohort_ok(result) == ()
    assert len(respx.calls) == 0  # nothing read, nothing written
