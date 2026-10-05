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
from src.data.ibkr.sync import gw_responding, tickle_auth


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
