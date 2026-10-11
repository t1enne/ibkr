"""``IbkrBroker.place_cohort``: deterministic sequencing, scaling, unsafe drops.

The cohort loop is broker-truth too: one bad symbol never abandons the rest, an
open cohort is scaled by ONE shared cash factor, and a STRUCTURAL drop (no shared
cash bound / scaled to 0 shares) reports ``UNFUNDED`` — unsafe at exit — unlike a
genuine cash-exhaustion ``REJECTED``, which re-mints next cycle.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from src.bt.state import ActionType
from src.live.adapters.ibkr.tests.conftest import (
    POSITIONS,
    STATUS,
    SUBMIT,
    FakeExposure,
    broker,
    close_intent,
    cohort_ok,
    filled_status,
    long_lot,
    mock_flip_positions,
    mock_no_working_orders,
    open_intent,
    sized_open,
)
from src.live.identity import OrderOutcome
from src.live.report import _UNSAFE_OUTCOMES


@respx.mock
@pytest.mark.asyncio
async def test_place_cohort_is_a_deterministic_loop_that_survives_one_failure() -> None:
    respx.post(SUBMIT).mock(
        side_effect=[
            httpx.Response(200, json=[{"error": "no market data"}]),  # 1st order fails
            httpx.Response(200, json=[{"order_id": "979320001"}]),  # 2nd lands
        ]
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=filled_status()))
    mock_no_working_orders()

    result = await broker().place_cohort(
        (
            sized_open("AAPL", 10.0, 100.0, 100_000.0),
            sized_open("MSFT", 10.0, 100.0, 100_000.0),
        )
    )

    first, second = cohort_ok(result)
    assert (first.intent.symbol, first.ok) == ("AAPL", False)
    assert "rejected" in first.message
    assert (second.intent.symbol, second.ok) == ("MSFT", True)


@respx.mock
@pytest.mark.asyncio
async def test_over_cash_open_cohort_is_scaled_by_one_shared_factor() -> None:
    """Two opens at 100% of the shared cash each are scaled, not both sent full."""
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": "979320001"}])
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=filled_status()))
    mock_no_working_orders()
    messages: list[str] = []
    wired = broker(log=messages.append)

    result = await wired.place_cohort(
        (
            sized_open("AAPL", qty=100.0, price=100.0, bound=15000.0),
            sized_open("MSFT", qty=100.0, price=100.0, bound=15000.0),
        )
    )

    results = cohort_ok(result)
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
        return_value=httpx.Response(200, json=[{"order_id": "979320001"}])
    )
    mock_no_working_orders()
    messages: list[str] = []
    wired = broker(log=messages.append)

    result = await wired.place_cohort(
        (sized_open("AAPL", qty=100.0, price=100.0, bound=5000.0),)
    )

    (open_result,) = cohort_ok(result)
    assert open_result.ok is False
    assert not submit.called  # never submitted, and never scaled to half size
    assert not any(m.startswith("cohort scaled") for m in messages)


@respx.mock
@pytest.mark.asyncio
async def test_scaled_open_that_floors_to_zero_is_dropped() -> None:
    """A scaled open whose whole-share qty is 0 is reported, never submitted."""
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": "979320001"}])
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=filled_status()))
    mock_no_working_orders()

    result = await broker().place_cohort(
        (
            sized_open("AAPL", qty=100.0, price=100.0, bound=5000.0),
            sized_open("MSFT", qty=1.0, price=100.0, bound=5000.0),
        )
    )

    results = cohort_ok(result)
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
    """A STRUCTURAL drop carries ``UNFUNDED``, never a safe ``REJECTED``.

    Two opens that disagree on their cash bound make the shared budget
    unprovable, so every open is dropped. A config that can never state that
    bound does this every cycle forever, so the outcomes must be unsafe — unlike
    a genuine cash-exhaustion refusal, which re-mints next cycle and stays safe.
    """
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": "979320001"}])
    )
    mock_no_working_orders()

    result = await broker().place_cohort(
        (
            sized_open("AAPL", qty=10.0, price=100.0, bound=15000.0),
            sized_open("MSFT", qty=10.0, price=100.0, bound=9000.0),
        )
    )

    results = cohort_ok(result)
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
    """Boundary: real cash exhaustion is ``REJECTED`` (safe), not ``UNFUNDED``.

    A lone open over its own bound is refused by the cash guard with kind
    ``rejected`` — nothing was taken and the intent re-mints next bar, so the
    cycle legitimately exits 0. Only a STRUCTURAL drop becomes unsafe.
    """
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": "979320001"}])
    )
    mock_no_working_orders()

    result = await broker().place_cohort(
        (sized_open("AAPL", qty=100.0, price=100.0, bound=5000.0),)
    )

    (open_result,) = cohort_ok(result)
    assert open_result.ok is False
    assert open_result.outcome is OrderOutcome.REJECTED
    assert OrderOutcome.REJECTED not in _UNSAFE_OUTCOMES
    assert submit.call_count == 0


@respx.mock
@pytest.mark.asyncio
async def test_close_is_sequenced_before_open_in_the_cohort() -> None:
    """A close frees the lot/cash the open needs, so its seq (and ref) comes first."""
    respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": "979320001"}])
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=filled_status()))
    mock_flip_positions()
    mock_no_working_orders()
    wired = broker(exposure=FakeExposure(1.0))
    wired.seed(long_lot("555000111"))

    await wired.place_cohort((open_intent(), close_intent()))

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


@respx.mock
@pytest.mark.asyncio
async def test_flip_close_then_open_is_allowed_once_the_close_fills() -> None:
    """A same-cycle close->open flip is NOT refused as a divergence.

    The cohort places the close first and awaits it to terminal before the open.
    By the open's guard the account has already applied our close (flat) while
    the book still shows the pre-close long lot, so ``net`` is 0 against a
    ``booked`` of 1. The guard must credit our own confirmed close (delta -1) and
    ALLOW it.
    """
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": "979320001"}])
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=filled_status()))
    mock_flip_positions()
    mock_no_working_orders()
    wired = broker(exposure=FakeExposure(1.0))
    wired.seed(long_lot("555000111"))

    result = await wired.place_cohort((open_intent(), close_intent()))

    placed = cohort_ok(result)
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
    produce, so the open must STILL be refused.
    """
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": "979320001"}])
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=filled_status()))
    respx.get(POSITIONS).mock(
        side_effect=[
            httpx.Response(
                200, json=[{"conid": 265598, "contractDesc": "AAPL", "position": 6}]
            ),
            httpx.Response(
                200, json=[{"conid": 265598, "contractDesc": "AAPL", "position": 5}]
            ),
        ]
    )
    mock_no_working_orders()
    wired = broker(exposure=FakeExposure(1.0))
    wired.seed(long_lot("555000111"))

    result = await wired.place_cohort((open_intent(), close_intent()))

    placed = cohort_ok(result)
    opens = [r for r in placed if r.intent.action is ActionType.long]
    assert opens and not opens[0].ok
    assert opens[0].error_kind == "divergence"
    assert submit.call_count == 1  # the close was sent; the open was refused
