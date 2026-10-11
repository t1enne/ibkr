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
from pathlib import Path
from typing import cast

import httpx
import pandas as pd
import pytest
import respx

from src.bt.state import ActionType
from src.live.adapters.ibkr.tests.conftest import (
    OPEN_ORDERS,
    ORDER_ID,
    REPLY,
    STATUS,
    SUBMIT,
    broker,
    cohort_ok,
    filled_status,
    mock_no_working_orders,
)
from src.live.types import OrderIntent

_FIXTURES = Path(__file__).parent / "fixtures"
TS = cast("pd.Timestamp", pd.Timestamp("2024-06-03T14:30:00Z"))


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
