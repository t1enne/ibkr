"""Gateway bring-up + candle sync — the logic ``sync_market_data.py`` now shims.

Moved out of the repo-root script so the gateway lifecycle lives beside the rest
of the IBKR code (plan §5). The root ``sync_market_data.py`` still owns the cron
contract (same env vars, same exit codes, same log shape) but holds no logic: it
delegates here.

Flow, unchanged: probe the session; if the container is down bring it up and wait
for it to answer; if the session is unauthenticated run the Playwright login; then
download 1h candles for every ticker in the local DB.
"""

from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from src.data.ibkr.client import IbkrClient, IbkrError
from src.data.ibkr.login import login_from_env

# ── Config ────────────────────────────────────────────────────────────────
PY_DIR = Path(__file__).resolve().parents[3]  # .../py
ROOT = PY_DIR.parent  # repo root
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "db.sqlite"
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
    """True if the gateway is up AND reports an authenticated session."""
    try:
        body = await client.tickle()
    except IbkrError:
        return False
    iserver = body.get("iserver") if isinstance(body, dict) else None
    status = iserver.get("authStatus") if isinstance(iserver, dict) else None
    return bool(status.get("authenticated")) if isinstance(status, dict) else False


async def gw_responding(client: IbkrClient) -> bool:
    """True if the endpoint answers at all (unauthenticated gateway returns 401)."""
    try:
        await client.tickle()
        return True
    except IbkrError as exc:
        return exc.kind == "auth"  # 401 = server up, just unauthenticated


def ensure_gateway() -> None:
    """Synchronous entry point: run the (async) gateway bring-up to completion."""
    asyncio.run(ensure_gateway_async())


async def ensure_gateway_async() -> None:
    client = IbkrClient()
    # Running container? "gateway   ... (healthy)" in `docker ps` output.
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
    healthy = "healthy" in ps

    if healthy and await tickle_auth(client):
        log("Gateway healthy + authenticated — nothing to start/login.")
        await client.aclose()
        return

    if not healthy:
        log("Gateway not healthy — docker compose up -d.")
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

    # Wait for the server to answer.
    waited = 0
    while waited < GATEWAY_TIMEOUT and not await gw_responding(client):
        time.sleep(2)
        waited += 2
    if not await gw_responding(client):
        sys.exit(f"Gateway did not become reachable within {GATEWAY_TIMEOUT}s.")

    if await tickle_auth(client):
        log("Gateway authenticated after start — skipping login.")
        await client.aclose()
        return

    log("No authenticated session — running the Playwright login.")
    asyncio.run(login_from_env(env_path=ENV_PATH))
    time.sleep(3)
    if not await tickle_auth(client):
        time.sleep(3)
        sys.exit("Login ran but /tickle still reports unauthenticated.")
    await client.aclose()

    log("Session authenticated after login.")


def symbols_from_db() -> list[str]:
    con = sqlite3.connect(DB_PATH)
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
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"Command failed ({e.returncode}): {e.cmd}")
