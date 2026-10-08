"""``ibkr gw`` — the Client Portal Gateway lifecycle.

``gw start`` (the default subcommand) brings the compose stack up in the
background, logs the session in, and then supervises the gateway in the
foreground: a /tickle keepalive plus a daily container bounce for the fresh
IBKR session. Ends on SIGINT (the container stays up); ``gw stop`` tears the
stack down.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from typing import cast

import click

from src.data.ibkr.login import TradingMode
from src.data.ibkr.sync import GatewayStartError, compose_down, log
from src.gw.supervise import supervise_gateway


@click.group(name="gw", invoke_without_command=True)
@click.pass_context
def gw_group(ctx: click.Context) -> None:
    """Client Portal Gateway — bring it up, keep it fresh, or tear it down."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(gw_start)


@click.command("start")
@click.option(
    "--mode",
    type=click.Choice(["paper", "live"]),
    default=None,
    help="Session to log in (default: $TRADING_MODE).",
)
def gw_start(mode: str | None) -> None:
    """Start docker, log in, then keep the gateway fresh (tickle + 24h bounce)."""
    try:
        asyncio.run(supervise_gateway(cast("TradingMode | None", mode)))
    except GatewayStartError as exc:
        raise click.ClickException(str(exc)) from exc
    except KeyboardInterrupt:
        log("SIGINT — supervisor stopped; the gateway container keeps running.")
        sys.exit(0)


@click.command("stop")
def gw_stop() -> None:
    """Stop the gateway compose stack."""
    try:
        compose_down()
    except (subprocess.CalledProcessError, OSError) as exc:
        raise click.ClickException(f"gateway stop failed: {exc}") from exc
    log("gateway stack stopped.")


gw_group.add_command(gw_start)
gw_group.add_command(gw_stop)