"""``IbkrClient`` error mapping + happy-path parse. No network, no gateway.

``respx`` intercepts every request, so these run offline. The four mappings the
plan names are asserted explicitly (401/auth, 429/rate_limit, 503/rate_limit,
timeout/transport) plus a happy-path parse of each endpoint this phase reads.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from src.data.ibkr.client import IbkrClient, IbkrError

BASE = "https://localhost:5000/v1/api/"


def _client() -> IbkrClient:
    return IbkrClient(base_url=BASE, account="DU1234567")


@respx.mock
@pytest.mark.asyncio
async def test_401_maps_to_auth() -> None:
    respx.get(f"{BASE}iserver/auth/status").mock(
        return_value=httpx.Response(401, json={"error": "unauthenticated"})
    )
    with pytest.raises(IbkrError) as excinfo:
        await _client().auth_status()
    assert excinfo.value.kind == "auth"


@respx.mock
@pytest.mark.asyncio
async def test_429_maps_to_rate_limit() -> None:
    respx.get(f"{BASE}tickle").mock(return_value=httpx.Response(429))
    with pytest.raises(IbkrError) as excinfo:
        await _client().tickle()
    assert excinfo.value.kind == "rate_limit"


@respx.mock
@pytest.mark.asyncio
async def test_503_maps_to_rate_limit() -> None:
    respx.get(f"{BASE}iserver/account/trades").mock(return_value=httpx.Response(503))
    with pytest.raises(IbkrError) as excinfo:
        await _client().trades()
    assert excinfo.value.kind == "rate_limit"


@respx.mock
@pytest.mark.asyncio
async def test_timeout_maps_to_transport() -> None:
    respx.get(f"{BASE}iserver/auth/status").mock(
        side_effect=httpx.ConnectTimeout("timed out")
    )
    with pytest.raises(IbkrError) as excinfo:
        await _client().auth_status()
    assert excinfo.value.kind == "transport"


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


@respx.mock
@pytest.mark.asyncio
async def test_unreadable_list_body_is_an_error_not_empty() -> None:
    """A shape drift must not read as "no trades" (finding M6).

    An empty list is indistinguishable from "no trades", which would silently
    drop the fills that gate duplicate opens, so a wrong container shape raises a
    typed error the port maps to ``Err``.
    """
    respx.get(f"{BASE}iserver/account/trades").mock(
        return_value=httpx.Response(200, json={"unexpected": {"trades": []}})
    )
    with pytest.raises(IbkrError) as excinfo:
        await _client().trades()
    assert excinfo.value.kind == "transport"


@respx.mock
@pytest.mark.asyncio
async def test_unreadable_positions_body_is_an_error_not_empty() -> None:
    respx.get(f"{BASE}portfolio/DU1234567/positions/0").mock(
        return_value=httpx.Response(200, json="no positions")
    )
    with pytest.raises(IbkrError) as excinfo:
        await _client().positions()
    assert excinfo.value.kind == "transport"


@pytest.mark.asyncio
async def test_portfolio_call_without_account_is_auth_error() -> None:
    client = IbkrClient(base_url=BASE, account=None)
    with pytest.raises(IbkrError) as excinfo:
        await client.portfolio_summary()
    assert excinfo.value.kind == "auth"


@respx.mock
@pytest.mark.asyncio
async def test_resolve_account_uses_first_when_unset() -> None:
    respx.get(f"{BASE}iserver/accounts").mock(
        return_value=httpx.Response(200, json={"accounts": ["DU9", "U1"]})
    )
    client = IbkrClient(base_url=BASE, account=None)
    assert await client.resolve_account() == "DU9"
    # Cached: a second call is answered without another request.
    assert await client.resolve_account() == "DU9"


def test_base_url_default_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IBKR_GATEWAY_URL", raising=False)
    assert IbkrClient(base_url=None).base_url == BASE
    monkeypatch.setenv("IBKR_GATEWAY_URL", "https://example.test/v1/api/")
    # A remote base url must not use the localhost-only ``verify=False`` default.
    assert IbkrClient(verify=True).base_url == "https://example.test/v1/api/"


def test_client_refuses_verify_false_off_localhost() -> None:
    with pytest.raises(AssertionError):
        IbkrClient(base_url="https://example.test/v1/api/", verify=False)


def test_httpx_client_uses_the_configured_base_url() -> None:
    client = _client()
    assert str(client.http.base_url) == BASE
