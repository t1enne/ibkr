"""Step 2 verification: the new wire fixtures parse and the client edges exist.

The four captures (spec §6) are the contract model: their FIELD NAMES are what
the parser and the placement path read, so a test parses the real files — a
hand-built mock that invents a gateway field name is exactly the bug AGENTS.md
warns about. `confirm()`/`cancel_order()` are exercised at the true HTTP edge.

REGRESSION tests: each pins a wire name the placement path depends on (a rename
in a fixture is a broken parser, not a cosmetic change).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from src.data.ibkr.client import IbkrClient
from src.live.adapters.ibkr.orders import (
    classify_reply,
    parse_working_order,
)
from src.live.identity import WorkingOrder

_FIXTURES = Path(__file__).parent / "fixtures"
BASE = "https://localhost:5000/v1/api/"
ACCOUNT = "DU452563"


def _load(name: str) -> object:
    return json.loads((_FIXTURES / name).read_text())


def test_stop_submit_challenge_fixture_classifies_as_confirm() -> None:
    body = _load("gateway_stop_submit_challenge.json")
    outcome = classify_reply(body["orders"])
    # The order does NOT rest until confirmed (spec §0); the confirm reply id is
    # the challenge's ``id``.
    assert outcome.kind == "confirm"
    assert outcome.reply_id
    assert "o10331" in outcome.message


def test_stop_submit_confirmed_fixture_classifies_as_success() -> None:
    body = _load("gateway_stop_submit_confirmed.json")
    outcome = classify_reply(body["reply"])
    assert outcome.kind == "success"
    assert outcome.order_id == "463681097"


def test_bracket_working_fixture_parses_both_legs_with_wire_names() -> None:
    body = _load("gateway_bracket_working.json")
    rows = body["orders"]
    stop = parse_working_order(rows[0])
    take = parse_working_order(rows[1])
    foreign = parse_working_order(rows[2])
    assert stop is not None and take is not None and foreign is None
    assert stop.order_type == "Stop"
    assert stop.level == 92.10
    assert stop.status == "Submitted" and stop.conid == 265598
    assert take.order_type == "Limit"
    assert take.level == 108.0
    assert stop.side == take.side == "SELL"
    assert stop.order_id == "463681097" and take.order_id == "463681098"
    # ``isSingleGroup``/OCA is reply-only: the working read carries no group id,
    # so identity is reconstructed from order_ref prefix + conid + side.
    assert all("oca_group_id" not in row for row in rows[:2])


def test_cancel_fixture_carries_order_id_and_absence_proof_shape() -> None:
    body = _load("gateway_cancel.json")
    assert body["order"]["order_id"] == "463681097"
    assert body["order"]["orderCancelSuccess"] is True
    # The confirming evidence is the follow-up read (order absent), not the reply.
    assert body["orders_after_cancel"] == []


@respx.mock
@pytest.mark.asyncio
async def test_client_confirm_posts_the_reply_confirmation() -> None:
    endpoint = respx.post(f"{BASE}iserver/reply/dlm_abc").mock(
        return_value=httpx.Response(200, json={"order_id": "463681097"})
    )
    client = IbkrClient(base_url=BASE, account=ACCOUNT)
    body = await client.confirm("dlm_abc")
    sent = json.loads(endpoint.calls[0].request.content.decode())
    assert sent == {"confirmed": True}
    assert body["order_id"] == "463681097"
    await client.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_client_cancel_order_deletes_the_order_endpoint() -> None:
    endpoint = respx.delete(f"{BASE}iserver/account/{ACCOUNT}/order/463681097").mock(
        return_value=httpx.Response(200, json={"orderCancelSuccess": True})
    )
    client = IbkrClient(base_url=BASE, account=ACCOUNT)
    body = await client.cancel_order(ACCOUNT, "463681097")
    assert body["orderCancelSuccess"] is True
    assert endpoint.calls.call_count == 1
    await client.aclose()


def test_working_order_defaults_keep_existing_callers_constructing() -> None:
    # The two NEW fields default; a pre-bracket caller that only knows the old
    # six fields still constructs a WorkingOrder.
    w = WorkingOrder("r", "1", 1, "AAPL", "BUY", "Submitted", 0.0)
    assert w.order_type == "" and w.level is None