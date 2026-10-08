"""Gateway supervisor for ``ibkr gw start``.

Keeps the Client Portal Gateway up, logged in, and fresh as a foreground
process — cron/tmux/systemd schedules it; there is no daemon. Startup reuses
``ensure_gateway_session`` (compose up + Playwright login when the session is
unauthenticated), then the loop pokes ``/tickle`` on an interval so the Client
Portal session does not idle out, and bounces the container + re-logs in every
24h because IBKR expires the session daily.

The container itself keeps running (``restart: unless-stopped``) when this
process is interrupted; ``ibkr gw stop`` tears the stack down.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.data.ibkr.client import IbkrClient
from src.data.ibkr.login import TradingMode
from src.data.ibkr.sync import (
    ENV_PATH,
    GatewayStartError,
    compose_restart,
    ensure_gateway_session,
    log,
    tickle_auth,
)

#: How often the supervisor pokes /tickle to keep the session alive.
TICKLE_SECONDS = 60
#: IBKR expires the Client Portal session daily; bounce + re-login on this cadence.
RESTART_SECONDS = 24 * 60 * 60


async def _tickle_window(
    client: IbkrClient, *, interval: int, window: int
) -> bool:
    """Tickle every *interval* until *window* elapses.

    Returns True when the whole window elapsed (time for the daily bounce),
    False when /tickle stopped reporting an authenticated session mid-window
    (the supervising loop re-brings the gateway up instead of bouncing it).
    ``window <= 0`` falls through to True immediately.
    """
    waited = 0
    while waited < window:
        if interval <= 0:
            break
        await asyncio.sleep(interval)
        waited += interval
        if not await tickle_auth(client):
            return False
    return True


async def supervise_gateway(
    mode: TradingMode | None = None,
    *,
    env_path: Path | str = ENV_PATH,
    tickle_seconds: int = TICKLE_SECONDS,
    restart_seconds: int = RESTART_SECONDS,
    cycles: int | None = None,
) -> None:
    """Supervise the gateway: bring up + login, then tickle / bounce to stay fresh.

    Each cycle logs the session in, tickles on *tickle_seconds* for
    *restart_seconds*, then bounces the container for the fresh daily session.
    A /tickle that stops authenticating mid-window returns early so the next
    cycle re-establishes the session without a bounce. ``cycles`` caps the loop
    (tests); ``None`` runs forever. A bring-up that cannot become reachable
    raises :class:`~src.data.ibkr.sync.GatewayStartError`.
    """
    client = IbkrClient()
    done = 0
    try:
        while cycles is None or done < cycles:
            try:
                await ensure_gateway_session(mode, env_path=env_path)
            except GatewayStartError as exc:
                log(f"gateway bring-up failed: {exc}")
                raise
            if await _tickle_window(
                client, interval=tickle_seconds, window=restart_seconds
            ):
                log("24h window elapsed — bouncing the gateway for a fresh session.")
                compose_restart()
            done += 1
    finally:
        await client.aclose()