"""Step 5 critical paths: entry fill -> placed OCA bracket; UNPROTECTED fail-closed.

Every request is driven through the REAL ``IbkrClient`` against ``respx`` using
the captured gateway fixtures (step-2 conveys) — no hand-invented gateway field
name, no live call. Behaviours pinned from the spec's §4/§6:

- a confirmed entry fill places + CONFIRMS the OCA pair in the SAME cycle and
  only reports it after a working-orders re-read shows every leg ``Submitted``;
- the confirm success is parsed from the CAPTURED ``{"reply": {...}}`` shape
  (the only live evidence) so the o354 loop can reach ``success``;
- a lone-``sl`` bracket (one leg) is verified and reported without indexing a
  pair it does not have;
- a pair never re-read as ``Submitted`` reports ``UNPROTECTED`` (exit 3, F2),
  never "placed".

REGRESSION tests: each pins a wire path (submit -> challenge -> confirm ->
re-read) the placement depends on; a rename in a fixture breaks the parser, not
a cosmetic change.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import httpx
import pandas as pd
import pytest
import respx

from src.bt.state import ActionType, PortfolioState
from src.exec.types import OrderType
from src.live.adapters.ibkr.broker import IbkrBroker
from src.live.adapters.ibkr.tests.conftest import (
    ACCOUNT,
    BASE,
    OPEN_ORDERS,
    ORDER_ID,
    REPLY,
    STATUS,
    SUBMIT,
    broker,
    close_intent,
    cohort_ok,
    filled_status,
    long_lot,
    mock_no_working_orders,
)
from src.live.identity import OrderOutcome
from src.live.protect import ProtectionTarget
from src.live.pure import OrderResult
from src.live.result import Ok
from src.live.types import OrderIntent

_FIXTURES = Path(__file__).parent / "fixtures"
TS = cast("pd.Timestamp", pd.Timestamp("2024-06-03T14:30:00Z"))
#: The captured conid/app/account the broker tests are wired to (conftest SCOPE).
SCOPE = "momentum"
_CANCEL = f"{BASE}iserver/account/{ACCOUNT}/order/"


def _tightened_stop() -> tuple[OrderIntent, ...]:
    """The single new stop leg a tighten of a stop-only lot would mint."""
    return (
        OrderIntent(
            symbol="AAPL",
            action=ActionType.close,
            qty=1.0,
            ref_price=94.0,
            reason="protect AAPL stop",
            position_id=str(265598),
            stop_loss=94.0,
            decision_ts=TS,
            order_type=OrderType.STP,
            protective=True,
            position_side=ActionType.long,
        ),
    )


def _entry_intent(sl: float | None, tp: float | None = None) -> OrderIntent:
    """A levelled long entry the broker will fill, then protect."""
    return OrderIntent(
        symbol="AAPL",
        action=ActionType.long,
        qty=1.0,
        ref_price=100.0,
        reason="open long (flat->long)",
        stop_loss=sl,
        take_profit=tp,
        decision_ts=TS,
        cash_bound=1000.0,
    )


def _challenge() -> list[dict[str, object]]:
    """The o354 challenge the pair submit returns (spec §0: id + o10331)."""
    return [
        {
            "id": "dlm_ua_unit",
            "message": ["You are about to submit a stop order (test body)."],
            "messageIds": ["o10331"],
            "isSuppressible": True,
        }
    ]


def _leg_rows(refs: tuple[str, ...]) -> list[dict[str, object]]:
    """Resting rows carrying ``refs`` (Stop for the 1st, Limit for the 2nd).

    Backed by the captured ``gateway_bracket_working.json`` legs — enough rows to
    match ONE ref (a lone ``sl`` bracket) or TWO (a full OCA pair). Kind comes
    from the capture's field names, never invented here.
    """
    rows = json.loads((_FIXTURES / "gateway_bracket_working.json").read_text())[
        "orders"
    ]
    out: list[dict[str, object]] = []
    for i, ref in enumerate(refs):
        mine = dict(rows[i])
        mine["order_ref"] = ref
        mine["orderId"] = 463681097 + i
        out.append(mine)
    return out


def _confirmed_reply() -> dict[str, object]:
    """The confirm success body exactly as production ``confirm()`` returns it.

    The ONLY live-captured evidence (spec §6 ``gateway_stop_submit_confirmed.json``)
    wraps the payload under ``"reply"`` — this is what the broker's ``confirm()``
    hands ``classify_reply``, so the test feeds the wrapper, never the unwrapped
    inner dict (reviewer n1/C1).
    """
    return {"reply": {"order_id": "463681097", "oca_group_id": "oco-x"}}


class _BracketMocks:
    """The respx routes + working read that MOVE as the bracket lands.

    ``_working`` returns nothing until the pair is submitted, then BOTH (or the
    sole) leg as Submitted with the refs the submit carried — the broker state
    moves exactly when our order lands (AGENTS § Testing the live path).
    """

    def __init__(self) -> None:
        self.submit_calls: list[dict[str, object]] = []

    def submit(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        self.submit_calls.append(cast("dict[str, object]", body))
        if "isSingleGroup" in json.dumps(body.get("orders", [])):
            return httpx.Response(200, json=_challenge())
        return httpx.Response(200, json=[{"order_id": ORDER_ID}])

    def working(self, request: httpx.Request) -> httpx.Response:
        for call in self.submit_calls:
            orders = cast("list[object]", call.get("orders", []))
            if any("isSingleGroup" in json.dumps(o) for o in orders):
                refs = tuple(str(cast("dict[str, object]", o)["cOID"]) for o in orders)
                return httpx.Response(200, json={"orders": _leg_rows(refs)})
        return httpx.Response(200, json={"orders": []})

    @classmethod
    def wire(cls) -> "_BracketMocks":
        m = cls()
        respx.post(SUBMIT).mock(side_effect=m.submit)
        respx.get(STATUS).mock(return_value=httpx.Response(200, json=filled_status()))
        respx.post(url__regex=rf"{REPLY}.*").mock(
            return_value=httpx.Response(200, json=_confirmed_reply())
        )
        respx.get(OPEN_ORDERS).mock(side_effect=m.working)
        return m


@respx.mock
@pytest.mark.asyncio
async def test_entry_fill_places_confirms_and_proves_the_oca_pair_resting() -> None:
    """A confirmed entry fill submits + confirms + re-reads the pair as resting."""
    _BracketMocks.wire()

    results = cohort_ok(
        await broker().place_cohort((_entry_intent(sl=95.0, tp=110.0),))
    )

    stop = [r for r in results if r.message.startswith("protect AAPL stop")]
    take = [r for r in results if r.message.startswith("protect AAPL take_profit")]
    assert len(stop) == 1 and len(take) == 1
    assert stop[0].ok is True and "resting GTC" in stop[0].message
    assert take[0].ok is True and "resting GTC" in take[0].message


@respx.mock
@pytest.mark.asyncio
async def test_lone_sl_bracket_is_verified_without_a_pair_index() -> None:
    """A lone-``sl`` (one-leg) bracket rests and reports without an IndexError."""
    _BracketMocks.wire()

    results = cohort_ok(await broker().place_cohort((_entry_intent(sl=95.0),)))

    stop = [r for r in results if r.message.startswith("protect AAPL stop")]
    assert len(stop) == 1
    assert stop[0].ok is True and "resting GTC" in stop[0].message


@respx.mock
@pytest.mark.asyncio
async def test_pair_never_re_read_as_submitted_reports_unprotected() -> None:
    """A pair submitted but never read Submitted is NAKED -> UNPROTECTED, not placed."""
    respx.post(SUBMIT).mock(
        side_effect=[
            httpx.Response(200, json=[{"order_id": ORDER_ID}]),
            httpx.Response(200, json=_challenge()),
        ]
    )
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=filled_status()))
    respx.post(url__regex=rf"{REPLY}.*").mock(
        return_value=httpx.Response(200, json=_confirmed_reply())
    )
    # The re-read NEVER shows the pair (always empty): the lot stays NAKED.
    mock_no_working_orders()

    results = cohort_ok(await broker().place_cohort((_entry_intent(sl=95.0),)))

    unprotected = [r for r in results if r.outcome.value == "unprotected"]
    assert len(unprotected) == 1
    row = unprotected[0]
    assert row.ok is False
    assert row.error_kind == "unprotected"


# --- item 2: the per-cycle naked-lot backstop -------------------------------

#: The conid the bracket fixture rests on (AAPL on the captured read).
_CONID = 265598


def _target(*, conid: int = _CONID, symbol: str = "AAPL") -> ProtectionTarget:
    return ProtectionTarget(conid=conid, symbol=symbol, position_id=str(conid), qty=1.0)


def _lot_book(*, conid: int = _CONID, symbol: str = "AAPL") -> PortfolioState:
    """A scope book holding ONE open, ARMED lot (its ``position_id`` IS the conid).

    The advisory ``stop_loss`` is what makes the lot a backstop target: a lot
    the strategy opened without a stop is unprotected by design, never NAKED.
    The item-5 read path is what will populate this level on the IBKR edge.
    """
    p = long_lot(position_id=str(conid))
    (position,) = p.positions["AAPL"]
    return replace(p, positions={symbol: (replace(position, stop_loss=95.0),)})


@respx.mock
@pytest.mark.asyncio
async def test_backstop_reports_a_lot_resting_no_leg_as_unprotected() -> None:
    """REGRESSION: the missing caller. A bracket lost after the entry cycle left
    the lot unprotected with a clean exit 0 — this pass must report UNPROTECTED."""
    mock_no_working_orders()

    results = await _attest(broker())

    assert len(results) == 1
    assert results[0].outcome is OrderOutcome.UNPROTECTED
    assert results[0].error_kind == "unprotected"
    assert results[0].position_id == str(_CONID)


@respx.mock
@pytest.mark.asyncio
async def test_backstop_silent_for_a_lot_resting_a_lone_sl() -> None:
    """REGRESSION (item 4 x item 2): a stop-only strategy's lone ``sl`` IS proof.

    With the pair-only reader this pass would flag the lot naked forever.
    """
    respx.get(OPEN_ORDERS).mock(
        return_value=httpx.Response(200, json={"orders": [_lone_leg()]})
    )

    assert await _attest(broker()) == ()


@respx.mock
@pytest.mark.asyncio
async def test_backstop_reports_unprotected_on_a_failed_working_read() -> None:
    """REGRESSION: ambiguity is never clean — an unreadable read attests nothing."""
    respx.get(OPEN_ORDERS).mock(return_value=httpx.Response(503, json={"error": "x"}))

    results = await _attest(broker())

    assert [r.outcome for r in results] == [OrderOutcome.UNPROTECTED]
    assert "read failed" in results[0].message


@respx.mock
@pytest.mark.asyncio
async def test_backstop_places_nothing_and_attests_only() -> None:
    """The backstop must never POST: this pass decides coverage, not entries."""
    mocks = _BracketMocks.wire()

    await _attest(broker())

    assert mocks.submit_calls == []


# --- item 4: a lone leg is acted on by replace and cancel -------------------


def _lone_leg(
    *, ref: str | None = None, status: str = "Submitted"
) -> dict[str, object]:
    """Row 0 of the captured bracket read: OUR lone STP leg (kind from the wire).

    The capture's own ``order_ref`` is a probe's, so it is re-stamped with ours —
    the field NAME (and every other field) stays the gateway's.
    """
    from src.exec.refs import scope_tag

    rows = json.loads((_FIXTURES / "gateway_bracket_working.json").read_text())[
        "orders"
    ]
    row: dict[str, object] = dict(rows[0])
    row["status"] = status
    row["order_ccp_status"] = status
    row["order_ref"] = ref if ref is not None else f"{scope_tag(SCOPE)}-11111111"
    return row


def _new_leg(ref: str) -> dict[str, object]:
    """The captured row re-stamped as the leg a just-submitted ticket landed as."""
    row = _lone_leg(ref=ref)
    row["orderId"] = 463681199
    return row


async def _attest(b: IbkrBroker) -> tuple[OrderResult, ...]:
    """Drive the real backstop over the conftest broker, returning its rows."""
    attested = await b.attest_protection(_lot_book())
    assert isinstance(attested, Ok)
    return cast("tuple[OrderResult, ...]", attested.value)


@respx.mock
@pytest.mark.asyncio
async def test_replace_on_a_lone_sl_cancels_the_existing_leg_never_duplicates() -> None:
    """REGRESSION: a stop-only lot's lone leg is cancelled before the tighten.

    With the pair-only reader the replace saw "nothing resting" and POSTed a
    second leg on top of the live one.
    """
    from src.exec.refs import scope_tag

    tag = scope_tag(SCOPE)
    cancel_calls: list[str] = []
    posts: list[list[dict[str, object]]] = []

    def working(_request: httpx.Request) -> httpx.Response:
        # The broker's own working surface moves with our orders: the old lone
        # leg rests until it is cancelled, then the NEW leg replaces it once the
        # submit landed (a constant read would hide a real refusal bug).
        if posts:
            return httpx.Response(
                200, json={"orders": [_new_leg(str(posts[0][0]["cOID"]))]}
            )
        if cancel_calls:
            return httpx.Response(200, json={"orders": []})
        return httpx.Response(200, json={"orders": [_lone_leg(ref=f"{tag}-11111111")]})

    def cancel(request: httpx.Request) -> httpx.Response:
        cancel_calls.append(str(request.url))
        return httpx.Response(200, json={"order": {"orderCancelSuccess": True}})

    def submit(request: httpx.Request) -> httpx.Response:
        posts.append(json.loads(request.content)["orders"])
        return httpx.Response(200, json=[{"order_id": ORDER_ID}])

    respx.get(OPEN_ORDERS).mock(side_effect=working)
    respx.delete(url__regex=rf"{_CANCEL}.*").mock(side_effect=cancel)
    respx.post(SUBMIT).mock(side_effect=submit)

    children = _tightened_stop()
    placed = await broker()._replace_bracket("AAPL", children, conid=_CONID)

    assert isinstance(placed, Ok)
    # The live leg was cancelled BEFORE the new one was submitted, so the conid
    # never carries two stops at once.
    assert len(cancel_calls) == 1
    # Only the NEW leg was POSTed once (no duplicate on top of the live leg),
    # and it is the leg the re-read proves resting.
    assert len(posts) == 1
    assert [str(o["orderType"]) for o in posts[0]] == ["STP"]
    assert str(posts[0][0]["cOID"]).startswith(f"{tag}-")


@respx.mock
@pytest.mark.asyncio
async def test_close_of_a_lot_holding_a_lone_sl_cancels_it_first() -> None:
    """REGRESSION (invariant 6, item 4): the lone leg is cancelled-and-confirmed
    before the close, so a live reduce order cannot race it and over-reduce."""
    from src.exec.refs import scope_tag

    tag = scope_tag(SCOPE)
    cancel_calls: list[str] = []

    def working(_request: httpx.Request) -> httpx.Response:
        if cancel_calls:
            return httpx.Response(200, json={"orders": []})
        return httpx.Response(200, json={"orders": [_lone_leg(ref=f"{tag}-11111111")]})

    def cancel(request: httpx.Request) -> httpx.Response:
        cancel_calls.append(str(request.url))
        return httpx.Response(200, json={"order": {"orderCancelSuccess": True}})

    respx.get(OPEN_ORDERS).mock(side_effect=working)
    respx.delete(url__regex=rf"{_CANCEL}.*").mock(side_effect=cancel)

    refused = await broker()._cancel_pair_before_close(close_intent())

    assert refused is None  # the cancel confirmed, so the close proceeds
    assert len(cancel_calls) == 1


@respx.mock
@pytest.mark.asyncio
async def test_close_refused_when_the_lone_leg_cannot_be_proven_gone() -> None:
    """REGRESSION: an unconfirmable cancel REFUSES the close (never over-reduce)."""
    from src.exec.refs import scope_tag

    tag = scope_tag(SCOPE)
    # The leg never leaves the working read: absence cannot be proven.
    respx.get(OPEN_ORDERS).mock(
        return_value=httpx.Response(
            200, json={"orders": [_lone_leg(ref=f"{tag}-11111111")]}
        )
    )
    respx.delete(url__regex=rf"{_CANCEL}.*").mock(
        return_value=httpx.Response(200, json={"order": {"orderCancelSuccess": True}})
    )

    refused = await broker()._cancel_pair_before_close(close_intent())

    assert refused is not None
    (row,) = refused
    assert row.outcome is OrderOutcome.UNPROTECTED
