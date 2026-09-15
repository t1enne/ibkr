#!/usr/bin/env python3
"""Check gateway is healthy+authenticated (start/login if not), then download
        1h candles for every symbol in the local sqlite db from 30 days ago to today.

Flow
        1. Probe https://localhost:5000/v1/api/tickle for authenticated session.
        2. If container is down/unhealthy or not authenticated: docker compose up -d,
           wait for health, then run py/scripts/login_ibkr.py (via uv) if still
           unauthenticated.
        3. Pull all tickers from data/db.sqlite (symbol table).
        4. Run `ibkr data dl <symbols...> --from <today-30d> --to <today>`.

Run from repo root. Reads IBKR_USERNAME/IBKR_PASSWORD from .env (login script
loads it itself). Safe from cron.
"""

from __future__ import annotations

import json
import ssl
import subprocess
import sys
import time
import urllib.request
import sqlite3
from datetime import date, timedelta
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
PY_DIR = ROOT / "py"
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "db.sqlite"
GATEWAY_URL = "https://localhost:5000/v1/api/tickle"
GATEWAY_TIMEOUT = 180  # seconds to wait for the healthcheck after up -d
DL_DAYS = 30
DL_BAR = "1h"

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", file=sys.stderr)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    log("=> " + " ".join(cmd))
    return subprocess.run(cmd, check=True, **kw)


def tickle_auth() -> bool:
    """True if the gateway is up AND reports an authenticated session."""
    try:
        req = urllib.request.Request(
            GATEWAY_URL, headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(req, context=_CTX, timeout=15) as resp:
            body = json.loads(resp.read().decode())
        return bool(body.get("iserver", {}).get("authStatus", {}).get("authenticated"))
    except Exception:
        return False


def gw_responding() -> bool:
    """True if the endpoint answers at all (unauthenticated gateway returns 401)."""
    try:
        req = urllib.request.Request(GATEWAY_URL)
        urllib.request.urlopen(req, context=_CTX, timeout=8).close()
        return True
    except urllib.error.HTTPError as e:
        return e.code < 500  # 401/4xx = server up, just unauthenticated
    except Exception:
        return False


def ensure_gateway() -> None:
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

    if healthy and tickle_auth():
        log("Gateway healthy + authenticated — nothing to start/login.")
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
    while waited < GATEWAY_TIMEOUT and not gw_responding():
        time.sleep(2)
        waited += 2
    if not gw_responding():
        sys.exit(f"Gateway did not become reachable within {GATEWAY_TIMEOUT}s.")

    if tickle_auth():
        log("Gateway authenticated after start — skipping login.")
        return

    log("No authenticated session — running login_ibkr.py.")
    run(["uv", "--directory", str(PY_DIR), "run", "scripts/login_ibkr.py"])
    time.sleep(3)
    if not tickle_auth():
        time.sleep(3)
        sys.exit("Login ran but /tickle still reports unauthenticated.")

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
        ],
    )


def main() -> None:
    # Login script expects to run with the .env that holds credentials next to it.
    ensure_gateway()

    symbols = symbols_from_db()
    today = date.today()
    download(symbols, today - timedelta(days=DL_DAYS), today)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"Command failed ({e.returncode}): {e.cmd}")
