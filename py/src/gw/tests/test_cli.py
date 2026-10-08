"""``ibkr gw`` wiring: default subcommand is start; stop tears the stack down."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from src.data.ibkr.sync import GatewayStartError
from src.gw.cli import gw_group


def test_bare_gw_invokes_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """`ibkr gw` (no subcommand) runs `start`, the supervisor."""
    seen: dict[str, object] = {}

    async def fake_supervise(mode: object = None) -> None:
        seen["mode"] = mode

    monkeypatch.setattr("src.gw.cli.supervise_gateway", fake_supervise)
    out = CliRunner().invoke(gw_group, [])
    assert out.exit_code == 0, out.output
    assert seen["mode"] is None


def test_gw_start_passes_the_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    async def fake_supervise(mode: object = None) -> None:
        seen["mode"] = mode

    monkeypatch.setattr("src.gw.cli.supervise_gateway", fake_supervise)
    out = CliRunner().invoke(gw_group, ["start", "--mode", "live"])
    assert out.exit_code == 0, out.output
    assert seen["mode"] == "live"


def test_gw_start_failure_is_clickexception_exit_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def dead(mode: object = None) -> None:
        raise GatewayStartError("docker compose up failed")

    monkeypatch.setattr("src.gw.cli.supervise_gateway", dead)
    out = CliRunner().invoke(gw_group, ["start"])
    assert out.exit_code == 1, out.output
    assert "docker compose up failed" in out.output


def test_gw_stop_runs_compose_down(monkeypatch: pytest.MonkeyPatch) -> None:
    stopped: list[bool] = []

    def fake_down() -> None:
        stopped.append(True)

    monkeypatch.setattr("src.gw.cli.compose_down", fake_down)
    out = CliRunner().invoke(gw_group, ["stop"])
    assert out.exit_code == 0, out.output
    assert stopped == [True]