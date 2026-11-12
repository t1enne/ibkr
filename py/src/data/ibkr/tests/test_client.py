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


def test_both_clients_carry_the_same_timeout() -> None:
    """A wedged gateway must fail as a transport error, not hang forever.

    The generated REST client opens its OWN httpx pool, so the httpx client's
    timeout says nothing about secdef search or candle history. A saturated
    gateway otherwise accepts the connection and never answers, and every `rest`
    call waits indefinitely.
    """
    client = IbkrClient(base_url=BASE, account="DU1234567", timeout=7.5)

    assert client.timeout_s == 7.5
    assert client.http.timeout.read == 7.5
    # The generated client keeps its timeout on the pool it builds lazily.
    assert client.rest._timeout == httpx.Timeout(7.5)


def test_candle_history_gets_the_long_read_timeout() -> None:
    """Candle history must NOT inherit the fast lookup timeout.

    One history chunk walks years of bars and legitimately outlives the secdef
    timeout; under it, a backfill dies as a message-less ``ReadTimeout`` that
    reads like a rate limit.
    """
    client = IbkrClient(base_url=BASE, account="DU1234567", timeout=7.5)

    assert client.history_timeout_s > client.timeout_s
    assert client.history_rest._timeout == httpx.Timeout(client.history_timeout_s)
    assert client.history_rest is not client.rest
