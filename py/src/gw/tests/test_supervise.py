"""``ibkr gw start`` supervisor: bring-up then tickle/bounce, typed on seams."""

from __future__ import annotations

import pytest

from src.data.ibkr.sync import GatewayStartError
from src.gw.supervise import supervise_gateway


@pytest.mark.asyncio
async def test_brings_up_logs_in_then_bounces_on_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Healthy tickles through the window -> one ensure, then one daily bounce."""
    ensured_mode: list[object] = []
    bounced: list[bool] = []

    async def fake_ensure(mode: object = None, *, env_path: object = None) -> str:
        ensured_mode.append(mode)
        return "ready"

    async def fake_tickle(client: object) -> bool:
        return True

    def fake_restart() -> None:
        bounced.append(True)

    monkeypatch.setattr("src.gw.supervise.ensure_gateway_session", fake_ensure)
    monkeypatch.setattr("src.gw.supervise.tickle_auth", fake_tickle)
    monkeypatch.setattr("src.gw.supervise.compose_restart", fake_restart)

    await supervise_gateway(
        None, tickle_seconds=0.001, restart_seconds=0.001, cycles=1
    )

    assert ensured_mode == [None]
    assert bounced == [True]


@pytest.mark.asyncio
async def test_dropped_session_spans_the_window_without_a_bounce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A /tickle that stops authenticating triggers a re-bring-up, not a bounce."""
    ensured = 0
    bounced: list[bool] = []

    async def fake_ensure(mode: object = None, *, env_path: object = None) -> str:
        nonlocal ensured
        ensured += 1
        return "logged-in"

    async def fake_tickle(client: object) -> bool:
        return False

    def fake_restart() -> None:
        bounced.append(True)

    monkeypatch.setattr("src.gw.supervise.ensure_gateway_session", fake_ensure)
    monkeypatch.setattr("src.gw.supervise.tickle_auth", fake_tickle)
    monkeypatch.setattr("src.gw.supervise.compose_restart", fake_restart)

    await supervise_gateway(
        None, tickle_seconds=0.001, restart_seconds=1000, cycles=2
    )

    assert ensured == 2  # re-established both cycles
    assert bounced == []  # the window never completed


@pytest.mark.asyncio
async def test_bring_up_failure_raises_gateway_start_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gateway that cannot come up surfaces the typed error (CLI owns exit 1)."""

    async def dead(mode: object = None, *, env_path: object = None) -> str:
        raise GatewayStartError("docker compose up failed")

    monkeypatch.setattr("src.gw.supervise.ensure_gateway_session", dead)
    monkeypatch.setattr(
        "src.gw.supervise.tickle_auth", lambda client: True  # unreachable
    )

    with pytest.raises(GatewayStartError):
        await supervise_gateway(None, cycles=1)