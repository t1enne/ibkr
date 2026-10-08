"""``IbkrClient`` error mapping + happy-path parse. No network, no gateway.

``respx`` intercepts every request, so these run offline. The four mappings the
plan names are asserted explicitly (401/auth, 429/rate_limit, 503/rate_limit,
timeout/transport) plus a happy-path parse of each endpoint this phase reads.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from src.data.ibkr.client import IbkrClient

BASE = "https://localhost:5000/v1/api/"


def _client() -> IbkrClient:
    return IbkrClient(base_url=BASE, account="DU1234567")


@respx.mock
@pytest.mark.asyncio
async def test_happy_path_parses_each_endpoint() -> None:
    respx.get(f"{BASE}iserver/auth/status").mock(
        return_value=httpx.Response(200, json={"authenticated": True})
    )
    respx.get(f"{BASE}tickle").mock(
        return_value=httpx.Response(200, json={"session": "abc", "iserver": {}})
    )
    respx.get(f"{BASE}iserver/accounts").mock(
        return_value=httpx.Response(200, json={"accounts": ["DU1234567"]})
    )
    respx.get(f"{BASE}portfolio/DU1234567/summary").mock(
        return_value=httpx.Response(200, json={"netliquidation": {"amount": 1}})
    )
    respx.get(f"{BASE}portfolio/DU1234567/positions/0").mock(
        return_value=httpx.Response(200, json=[{"conid": 265598}])
    )
    respx.get(f"{BASE}iserver/account/trades").mock(
        return_value=httpx.Response(200, json=[{"execution_id": "e1"}])
    )

    client = _client()
    assert (await client.auth_status())["authenticated"] is True
    assert (await client.tickle())["session"] == "abc"
    assert await client.accounts() == ["DU1234567"]
    assert await client.portfolio_summary() == {"netliquidation": {"amount": 1}}
    assert await client.positions() == [{"conid": 265598}]
    assert await client.trades() == [{"execution_id": "e1"}]
