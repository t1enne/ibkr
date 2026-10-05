"""``IbkrGateway`` readiness/keepalive: typed values, never raises."""

from __future__ import annotations

import httpx
import pytest
import respx

from src.data.ibkr.client import IbkrClient
from src.data.ibkr.gateway import IbkrGateway
from src.live.result import Ok

BASE = "https://localhost:5000/v1/api/"


def _gateway() -> IbkrGateway:
    return IbkrGateway(IbkrClient(base_url=BASE, account="DU1234567"))


@respx.mock
@pytest.mark.asyncio
async def test_is_ready_true_when_authenticated() -> None:
    respx.get(f"{BASE}iserver/auth/status").mock(
        return_value=httpx.Response(200, json={"authenticated": True})
    )
    result = await _gateway().is_ready()
    assert isinstance(result, Ok)
    assert result.value is True


@respx.mock
@pytest.mark.asyncio
async def test_is_ready_false_when_authenticated() -> None:
    respx.get(f"{BASE}iserver/auth/status").mock(
        return_value=httpx.Response(200, json={"authenticated": False})
    )
    result = await _gateway().is_ready()
    assert isinstance(result, Ok)
    assert result.value is False


@respx.mock
@pytest.mark.asyncio
async def test_is_ready_transport_failure_is_err_not_raise() -> None:
    respx.get(f"{BASE}iserver/auth/status").mock(
        side_effect=httpx.ConnectTimeout("down")
    )
    result = await _gateway().is_ready()
    assert not isinstance(result, Ok)
    assert result.error.kind == "transport"


@respx.mock
@pytest.mark.asyncio
async def test_ensure_ready_tickles_then_checks_auth() -> None:
    respx.get(f"{BASE}tickle").mock(return_value=httpx.Response(200, json={}))
    respx.get(f"{BASE}iserver/auth/status").mock(
        return_value=httpx.Response(200, json={"authenticated": True})
    )
    result = await _gateway().ensure_ready()
    assert isinstance(result, Ok)


@respx.mock
@pytest.mark.asyncio
async def test_ensure_ready_unauthenticated_without_login_is_auth_error() -> None:
    respx.get(f"{BASE}tickle").mock(return_value=httpx.Response(200, json={}))
    respx.get(f"{BASE}iserver/auth/status").mock(
        return_value=httpx.Response(200, json={"authenticated": False})
    )
    result = await _gateway().ensure_ready()
    assert not isinstance(result, Ok)
    assert result.error.kind == "auth"


@respx.mock
@pytest.mark.asyncio
async def test_ensure_ready_runs_login_then_reprobes() -> None:
    respx.get(f"{BASE}tickle").mock(return_value=httpx.Response(200, json={}))
    route = respx.get(f"{BASE}iserver/auth/status")
    route.side_effect = [
        httpx.Response(200, json={"authenticated": False}),
        httpx.Response(200, json={"authenticated": True}),
    ]
    calls: list[int] = []

    async def login() -> None:
        calls.append(1)

    result = await _gateway().ensure_ready(login=login)
    assert isinstance(result, Ok)
    assert calls == [1]
