"""``tickle_auth`` / ``gw_responding`` parser tests over a fake client (no network).

These two helpers decide whether the gateway is up and authenticated. They must
accept BOTH gateway shapes (flat ``authenticated`` and nested
``iserver.authStatus.authenticated``) and treat a strict JSON ``true`` as the only
authenticated value.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from src.data.ibkr.client import IbkrClient, IbkrError
from src.data.ibkr.sync import (
    GatewayStartError,
    ensure_gateway_session,
    gw_responding,
    tickle_auth,
)


class _FakeClient:
    """A minimal ``IbkrClient`` stand-in: ``tickle`` returns a body or raises."""

    def __init__(self, body: object = None, error: IbkrError | None = None) -> None:
        self._body = body
        self._error = error

    async def tickle(self) -> Any:
        if self._error is not None:
            raise self._error
        return self._body


def _c(body: object = None, error: IbkrError | None = None) -> IbkrClient:
    return cast("IbkrClient", _FakeClient(body, error))


@pytest.mark.asyncio
async def test_tickle_auth_reads_nested_shape() -> None:
    body = {"iserver": {"authStatus": {"authenticated": True}}}
    assert await tickle_auth(_c(body)) is True


@pytest.mark.asyncio
async def test_tickle_auth_reads_flat_shape() -> None:
    assert await tickle_auth(_c({"authenticated": True})) is True


@pytest.mark.asyncio
async def test_tickle_auth_false_when_logged_out() -> None:
    body = {"iserver": {"authStatus": {"authenticated": False}}}
    assert await tickle_auth(_c(body)) is False


@pytest.mark.asyncio
async def test_tickle_auth_string_false_is_not_authenticated() -> None:
    assert await tickle_auth(_c({"authenticated": "false"})) is False


@pytest.mark.asyncio
async def test_tickle_auth_error_is_false() -> None:
    client = _c(error=IbkrError("transport", "down", "tickle"))
    assert await tickle_auth(client) is False


@pytest.mark.asyncio
async def test_gw_responding_true_on_success() -> None:
    assert await gw_responding(_c({})) is True


@pytest.mark.asyncio
async def test_gw_responding_true_on_401() -> None:
    """A 401 means the server is up, just unauthenticated."""
    client = _c(error=IbkrError("auth", "401", "tickle"))
    assert await gw_responding(client) is True


@pytest.mark.asyncio
async def test_gw_responding_false_on_transport() -> None:
    client = _c(error=IbkrError("transport", "down", "tickle"))
    assert await gw_responding(client) is False


# --- ensure_gateway_session: typed bring-up, no sys.exit ---------------------


class _StubClient:
    """A stand-in whose ``tickle`` always answers *body* (or raises *error*)."""

    def __init__(self, body: object, error: IbkrError | None = None) -> None:
        self._body = body
        self._error = error
        self.closed = False

    async def tickle(self) -> Any:
        if self._error is not None:
            raise self._error
        return self._body

    async def aclose(self) -> None:
        self.closed = True


class _AsyncioStub:
    """A loop-free stand-in for the module's ``asyncio`` (its only use: sleep)."""

    async def sleep(self, _seconds: float) -> None:
        return None


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    *,
    healthy: bool,
    body: object,
    error: IbkrError | None = None,
) -> tuple[_StubClient, list[str]]:
    """Patch the bring-up's four edges; return the fake client + the call log."""
    client = _StubClient(body, error)
    calls: list[str] = []
    monkeypatch.setattr("src.data.ibkr.sync.IbkrClient", lambda *a, **k: client)
    monkeypatch.setattr("src.data.ibkr.sync._container_healthy", lambda: healthy)
    monkeypatch.setattr("src.data.ibkr.sync._compose_up", lambda: calls.append("up"))
    monkeypatch.setattr("src.data.ibkr.sync.asyncio", _AsyncioStub())

    async def fake_login(mode: object = None, env_path: object = None) -> None:
        calls.append(f"login:{mode}")

    monkeypatch.setattr("src.data.ibkr.sync.login_from_env", fake_login)
    return client, calls


@pytest.mark.asyncio
async def test_session_healthy_and_authenticated_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, calls = _wire(monkeypatch, healthy=True, body={"authenticated": True})
    assert await ensure_gateway_session("paper") == "ready"
    assert calls == []  # no compose, no login
    assert client.closed is True


@pytest.mark.asyncio
async def test_session_starts_the_container_then_finds_a_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, calls = _wire(monkeypatch, healthy=False, body={"authenticated": True})
    assert await ensure_gateway_session("paper") == "started"
    assert calls == ["up"]  # started, no login needed
    assert client.closed is True


@pytest.mark.asyncio
async def test_session_logs_in_with_the_caller_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mode handoff: a `live` caller must not open the paper session."""

    class _FlipsAuthenticated(_StubClient):
        async def tickle(self) -> Any:
            # Unauthenticated until the login seam has run (which the client
            # cannot see directly, so the call log is the flag).
            return {"authenticated": any(c.startswith("login") for c in calls)}

    client = _FlipsAuthenticated({})
    calls: list[str] = []
    monkeypatch.setattr("src.data.ibkr.sync.IbkrClient", lambda *a, **k: client)
    monkeypatch.setattr("src.data.ibkr.sync._container_healthy", lambda: False)
    monkeypatch.setattr("src.data.ibkr.sync._compose_up", lambda: calls.append("up"))
    monkeypatch.setattr("src.data.ibkr.sync.asyncio", _AsyncioStub())

    async def fake_login(mode: object = None, env_path: object = None) -> None:
        calls.append(f"login:{mode}")

    monkeypatch.setattr("src.data.ibkr.sync.login_from_env", fake_login)

    assert await ensure_gateway_session("live") == "logged-in"
    assert calls == ["up", "login:live"]
    assert client.closed is True


@pytest.mark.asyncio
async def test_session_unreachable_gateway_is_a_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never ``sys.exit``: the caller owns the exit code."""
    _client, calls = _wire(
        monkeypatch,
        healthy=False,
        body={},
        error=IbkrError("transport", "down", "tickle"),
    )
    with pytest.raises(GatewayStartError, match="did not become reachable"):
        await ensure_gateway_session("paper", timeout=0)
    assert calls == ["up"]  # login never attempted on a dead gateway
