"""Gateway login via Playwright — one home for the browser-driving code.

Moved verbatim out of ``scripts/login_ibkr.py`` so the script is a thin shim and
the gateway session logic lives beside the rest of the gateway code. The browser
stays open on success so a human can finish an MFA prompt in the Client Portal
window; on failure the browser is closed and the process exits non-zero (the
cron contract ``run_pipeline.sh``/``sync_market_data.py`` rely on).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Literal, cast

from playwright.async_api import async_playwright

TradingMode = Literal["paper", "live"]

#: The gateway's own login page (NOT under /v1/api/).
GATEWAY_LOGIN_URL = "https://localhost:5000"


def load_env(path: str | Path = ".env") -> None:
    """Load a .env file into os.environ. Never overrides existing env vars."""

    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key not in os.environ:
            os.environ[key] = val


async def login_ibkr(username: str, password: str, mode: TradingMode = "paper") -> None:
    """Drive the gateway login form; raise ``SystemExit(1)`` on any failure."""
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(ignore_https_errors=True)
        page = await context.new_page()

        try:
            await page.goto(GATEWAY_LOGIN_URL)

            # Enable paper trading mode
            if mode == "paper":
                await page.click('label[for="toggle1"]')

            await page.wait_for_timeout(500)

            await page.fill(
                'input[name="username"], #username, input[type="text"]',
                username,
            )
            await page.fill(
                'input[name="password"], #password, input[type="password"]',
                password,
            )
            await page.click(
                'button[type="submit"], input[type="submit"], .login-button',
            )

            # Wait for the success text to appear
            await page.get_by_text("Client login succeeds").wait_for()
            await page.wait_for_timeout(1000)

            print("Login attempt completed")
            # Browser stays open so you can interact with the Gateway
        except Exception as exc:
            print(f"Login failed: {exc}", file=sys.stderr)
            await browser.close()
            raise SystemExit(1) from exc


async def login_from_env(
    mode: TradingMode | None = None, env_path: str | Path = ".env"
) -> None:
    """Login using ``IBKR_USERNAME``/``IBKR_PASSWORD`` (+ ``TRADING_MODE``).

    The injectable form ``IbkrGateway.ensure_ready`` expects: it loads ``env_path``
    (the caller names it — the script and the sync both live elsewhere), resolves
    the mode, and delegates to :func:`login_ibkr`.
    """
    load_env(env_path)
    username = os.environ.get("IBKR_USERNAME")
    password = os.environ.get("IBKR_PASSWORD")
    if not username or not password:
        raise SystemExit(
            "Provide IBKR_USERNAME and IBKR_PASSWORD via env vars "
            "(or a .env file beside the working directory)."
        )
    resolved = mode or os.environ.get("TRADING_MODE", "paper")
    await login_ibkr(username, password, cast("TradingMode", resolved))


def main() -> None:
    """CLI entry point for the ``scripts/login_ibkr.py`` shim."""
    import argparse

    parser = argparse.ArgumentParser(description="Login to IBKR Gateway")
    parser.add_argument("--username", default=os.environ.get("IBKR_USERNAME"))
    parser.add_argument("--password", default=os.environ.get("IBKR_PASSWORD"))
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default=os.environ.get("TRADING_MODE", "paper"),
    )
    args = parser.parse_args()

    if not args.username or not args.password:
        print(
            "Provide IBKR_USERNAME and IBKR_PASSWORD via env vars, "
            "or pass --username / --password on the command line.",
            file=sys.stderr,
        )
        sys.exit(1)

    asyncio.run(login_ibkr(args.username, args.password, args.mode))
