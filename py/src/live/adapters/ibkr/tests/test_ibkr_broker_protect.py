"""Step 5 critical paths: entry fill -> placed OCA bracket; UNPROTECTED fail-closed.

Every request is driven through the REAL ``IbkrClient`` against ``respx`` using
the captured gateway fixtures (step-2 conveys) — no hand-invented gateway field
name, no live call. Two behaviours are pinned, both from the spec's §4/§6:

- a confirmed entry fill places + CONFIRMS the OCA pair in the SAME cycle and
  only reports it after a working-orders re-read shows both legs ``Submitted``;
- a pair that can never be re-read as ``Submitted`` reports ``UNPROTECTED``
  (exit 3, F2), never "placed".

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


def _pair_rows(refs: tuple[str, str]) -> list[dict[str, object]]:
    """Two resting rows (our STP + LMT legs) carrying ``refs``, from the capture."""
    rows = json.loads((_FIXTURES / "gateway_bracket_working.json").read_text())[
        "orders"
    ]
    out: list[dict[str, object]] = []
    for row, ref, order_id in (
        (rows[0], refs[0], 463681097),
        (rows[1], refs[1], 463681098),
    ):
        mine = dict(row)
        mine["order_ref"] = ref
        mine["orderId"] = order_id
        out.append(mine)
    return out


@respx.mock
@pytest.mark.asyncio
async def test_entry_fill_places_confirms_and_proves_the_oca_pair_resting() -> None:
    """A confirmed entry fill submits + confirms + re-reads the pair as resting."""
    submit_calls: list[dict[str, object]] = []

    def _submit(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        submit_calls.append(cast("dict[str, object]", body))
        if "isSingleGroup" in json.dumps(body.get("orders", [])):
            return httpx.Response(200, json=_challenge())
        return httpx.Response(200, json=[{"order_id": ORDER_ID}])

    respx.post(SUBMIT).mock(side_effect=_submit)
    respx.get(STATUS).mock(return_value=httpx.Response(200, json=filled_status()))
    respx.post(url__regex=rf"{REPLY}.*").mock(
        return_value=httpx.Response(
            200, json={"order_id": "463681097", "oca_group_id": "oco-x"}
        )
    )

    def _working(request: httpx.Request) -> httpx.Response:
        # Before the pair is submitted nothing works; once submitted, return BOTH
        # legs as Submitted with the refs the pair submit carried.
        for call in submit_calls:
            orders = cast("list[object]", call.get("orders", []))
            if any("isSingleGroup" in json.dumps(o) for o in orders):
                refs = cast(
                    "tuple[str, str]",
                    tuple(str(cast("dict[str, object]", o)["cOID"]) for o in orders),
                )
                return httpx.Response(200, json={"orders": _pair_rows(refs)})
        return httpx.Response(200, json={"orders": []})

    respx.get(OPEN_ORDERS).mock(side_effect=_working)

    placed = await broker().place_cohort((_entry_intent(sl=95.0, tp=110.0),))
    results = cohort_ok(placed)

    stop = [r for r in results if r.message.startswith("protect AAPL stop")]
    take = [r for r in results if r.message.startswith("protect AAPL take_profit")]
    assert len(stop) == 1 and len(take) == 1
    assert stop[0].ok is True and "resting GTC" in stop[0].message
    assert take[0].ok is True and "resting GTC" in take[0].message


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
        return_value=httpx.Response(200, json={"order_id": "463681097"})
    )
    # The re-read NEVER shows the pair (always empty): the lot stays NAKED.
    mock_no_working_orders()

    placed = await broker().place_cohort((_entry_intent(sl=95.0),))
    results = cohort_ok(placed)

    unprotected = [r for r in results if r.outcome.value == "unprotected"]
    assert len(unprotected) == 1
    row = unprotected[0]
    assert row.ok is False
    assert row.error_kind == "unprotected"
