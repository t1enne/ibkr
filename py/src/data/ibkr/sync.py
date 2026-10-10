"""Gateway bring-up + candle sync — the logic ``sync_market_data.py`` now shims.

Moved out of the repo-root script so the gateway lifecycle lives beside the rest
of the IBKR code (plan §5). The root ``sync_market_data.py`` still owns the cron
contract (same env vars, same exit codes, same log shape) but holds no logic: it
delegates here.

Flow, unchanged: probe the session; if the container is down bring it up and wait
for it to answer; if the session is unauthenticated run the Playwright login; then
download 1h candles for every ticker in the local DB.

The bring-up is ``ensure_gateway_session``, which raises a typed
``GatewayStartError`` rather than exiting; only the cron shim
(``ensure_gateway_async``) turns that into ``sys.exit``. ``ibkr gw start``
awaits the same function in-process on its first cycle (and then supervises
the session — see ``src/gw/supervise.py``), so the gateway lifecycle needs no
sibling script.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from src.data.ibkr.client import IbkrClient, IbkrError, is_authenticated
from src.data.ibkr.login import TradingMode, login_from_env
from src.db.connection import get_connection
from src.db.path import resolve_db_path

# ── Config ────────────────────────────────────────────────────────────────
PY_DIR = Path(__file__).resolve().parents[3]  # .../py
ROOT = PY_DIR.parent  # repo root
#: The candle file, resolved by :mod:`src.db.path` (honours ``IBKR_DB_PATH``).
DB_PATH = resolve_db_path()
ENV_PATH = PY_DIR / ".env"
GATEWAY_TIMEOUT = 180  # seconds to wait for the healthcheck after up -d
DL_DAYS = 30
DL_BAR = "1h"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", file=sys.stderr)


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    log("=> " + " ".join(cmd))
    return subprocess.run(cmd, check=True, text=True)


async def tickle_auth(client: IbkrClient) -> bool:
    """True if the gateway is up AND reports an authenticated session.

    ``is_authenticated`` accepts both gateway shapes (the flat ``authenticated``
    of /iserver/auth/status and the nested ``iserver.authStatus.authenticated``
    of /tickle) and requires a strict JSON ``true``.
    """
    try:
        body = await client.tickle()
    except IbkrError:
        return False
    return is_authenticated(body)


async def gw_responding(client: IbkrClient) -> bool:
    """True if the endpoint answers at all (unauthenticated gateway returns 401)."""
    try:
        await client.tickle()
        return True
    except IbkrError as exc:
        return exc.kind == "auth"  # 401 = server up, just unauthenticated


#: What the bring-up had to do. ``ready``: nothing. ``started``: the container
#: came up. ``logged-in``: the Playwright login ran.
GatewayAction = Literal["ready", "started", "logged-in"]


class GatewayStartError(RuntimeError):
    """The gateway could not be made reachable/authenticated.

    Typed rather than raised as ``SystemExit`` so a CLI caller owns the exit code:
    the cron shim converts it to ``sys.exit``, ``ibkr gw start`` to a
    ``ClickException``. Nothing here touches the account, the ledger or an order.
    """


def _container_healthy() -> bool:
    """True when ``docker ps`` reports the compose gateway container healthy."""
    ps = subprocess.run(
        [
            "docker",
            "ps",
            "--filter",
            "name=^gateway$",
            "--format",
            "{{.Names}} {{.Status}}",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return "healthy" in ps


def _compose_up() -> None:
    """Start the compose stack detached (the one place `-d` is spelled)."""
    run(
        [
            "docker",
            "compose",
            "--ansi",
            "never",
            "--project-directory",
            str(ROOT),
            "up",
            "-d",
        ]
    )


def compose_restart() -> None:
    """Bounce the gateway container (the daily IBKR session reset).

    ``docker compose restart`` re-launches the running container in place; the
    caller (``ibkr gw start``) re-runs ``ensure_gateway_session`` afterwards to
    wait for it and re-log the session in.
    """
    run(
        [
            "docker",
            "compose",
            "--ansi",
            "never",
            "--project-directory",
            str(ROOT),
            "restart",
        ]
    )


def compose_down() -> None:
    """Tear the gateway stack down (``ibkr gw stop``)."""
    run(
        [
            "docker",
            "compose",
            "--ansi",
            "never",
            "--project-directory",
            str(ROOT),
            "down",
        ]
    )


async def ensure_gateway_session(
    mode: TradingMode | None = None,
    *,
    timeout: int = GATEWAY_TIMEOUT,
    env_path: Path | str = ENV_PATH,
) -> GatewayAction:
    """Bring the gateway up and log the session in when it is unauthenticated.

    An already-healthy + authenticated gateway is left alone. Otherwise the
    container is started (when ``docker ps`` does not call it healthy) and polled
    until it answers, and the Playwright login runs when ``/tickle`` still reports
    an unauthenticated session. ``mode`` is the session the login should open — a
    live caller passes its config's ``mode`` so a ``paper`` config never opens the
    LIVE session; ``None`` falls back to the environment's ``TRADING_MODE``.

    Raises ``GatewayStartError`` instead of ``sys.exit``: the caller owns the exit
    code. Account-side state is never touched.
    """
    client = IbkrClient()
    try:
        try:
            healthy = _container_healthy()
        except OSError as exc:  # no docker binary on the host
            raise GatewayStartError(f"docker unavailable: {exc}") from exc
        if healthy and await tickle_auth(client):
            log("Gateway healthy + authenticated — nothing to start/login.")
            return "ready"

        if not healthy:
            log("Gateway not healthy — docker compose up -d.")
            try:
                _compose_up()
            except (subprocess.CalledProcessError, OSError) as exc:
                raise GatewayStartError(f"docker compose up failed: {exc}") from exc

        # Wait for the server to answer. ``asyncio.sleep`` (not ``time.sleep``) so
        # the coroutine yields instead of blocking the loop it runs on.
        waited = 0
        while waited < timeout and not await gw_responding(client):
            await asyncio.sleep(2)
            waited += 2
        if not await gw_responding(client):
            raise GatewayStartError(
                f"Gateway did not become reachable within {timeout}s."
            )

        if await tickle_auth(client):
            log("Gateway authenticated after start — skipping login.")
            return "started"

        log("No authenticated session — running the Playwright login.")
        # ``login_from_env`` is a coroutine: AWAIT it. Calling ``asyncio.run`` here
        # would nest event loops inside this running loop and raise RuntimeError.
        await login_from_env(mode=mode, env_path=env_path)
        await asyncio.sleep(5)
        if not await tickle_auth(client):
            await asyncio.sleep(5)
            raise GatewayStartError(
                "Login ran but /tickle still reports unauthenticated."
            )
        log("Session authenticated after login.")
        return "logged-in"
    finally:
        await client.aclose()


def ensure_gateway() -> None:
    """Synchronous entry point: run the (async) gateway bring-up to completion."""
    asyncio.run(ensure_gateway_async())


async def ensure_gateway_async() -> None:
    """Cron shim: the bring-up, with a failure as a non-zero process exit.

    ``sync_market_data.py``'s contract is a non-zero exit on a failed
    gateway/login, so the typed ``GatewayStartError`` is converted HERE and
    nowhere else.
    """
    try:
        await ensure_gateway_session()
    except GatewayStartError as exc:
        sys.exit(str(exc))


def symbols_from_db() -> list[str]:
    con = get_connection()
    try:
        rows = con.execute("SELECT ticker FROM symbol ORDER BY ticker").fetchall()
    finally:
        con.close()
    if not rows:
        sys.exit(f"No symbols found in {DB_PATH} (symbol table).")
    return [r[0] for r in rows]


def download(symbols: list[str], from_d: date, to_d: date) -> None:
    today = to_d.isoformat()
    frm = from_d.isoformat()
    log(f"Downloading {len(symbols)} symbols {frm}..{today} ({DL_BAR}).")
    run(
        [
            "uv",
            "--directory",
            str(PY_DIR),
            "run",
            "ibkr",
            "data",
            "dl",
            *symbols,
            "--from",
            frm,
            "--to",
            today,
            "--bar",
            DL_BAR,
        ]
    )


def main() -> None:
    ensure_gateway()
    symbols = symbols_from_db()
    today = date.today()
    download(symbols, today - timedelta(days=DL_DAYS), today)


if __name__ == "__main__":
    # ``subprocess`` failures are already surfaced by ``run()`` (``check=True``);
    # nothing here catches ``CalledProcessError`` (the old handler was dead).
    main()
