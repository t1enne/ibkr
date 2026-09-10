"""SEC EDGAR HTTP client — ticker -> CIK -> companyfacts, rate-limited + cached.

The only network-touching module in the fundamentals path (plus the CLI that
drives it). Two endpoints:

* ``company_tickers.json`` — the full ticker -> CIK map (fetched once, cached).
* ``api/xbrl/companyfacts/CIK##########.json`` — every XBRL fact a filer has
  reported, cached per symbol so a re-run of ``dl`` for unchanged data costs no
  requests.

SEC requires a descriptive ``User-Agent`` and asks for <= 10 req/s; ``batch_delay_s``
paces the per-symbol loop well inside that. Cache files live under
``data/fundamentals_cache/`` (a gitignored sibling of the DB) and are plain JSON
so a stale or hand-edited payload can be inspected with ``cat``.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# SEC requires a declared caller identity; a generic agent is rejected.
_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "ibkr-py fundamentals research contact@example.com"
)

_DEFAULT_CACHE_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "data"
    / "fundamentals_cache"
)

_TIMEOUT_S = 30.0


def _headers() -> dict[str, str]:
    return {"User-Agent": _USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def cache_dir(path: str | Path | None = None) -> Path:
    """Cache directory (created on demand)."""
    out = Path(path) if path is not None else _DEFAULT_CACHE_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def _read_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        # A truncated cache (interrupted download) must retry the network, not
        # poison every later run with an unparseable file.
        return None
    return payload if isinstance(payload, dict) else None


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)  # atomic: a killed process never leaves a half-written cache


def cik_for_ticker(
    ticker: str,
    *,
    client: httpx.Client | None = None,
    cache: str | Path | None = None,
) -> str:
    """SEC CIK (10-digit, zero-padded) for ``ticker``.

    Reads the ticker->CIK map from cache when present, else fetches it once and
    caches it (it changes as issuers list, but not within a run's lifetime).

    Raises:
        KeyError: when the ticker is not a registered SEC filer (ETFs, foreign
            private issuers without a US listing, crypto).
    """
    directory = cache_dir(cache)
    path = directory / "company_tickers.json"
    payload = _read_cache(path)
    if payload is None:
        with _client(client) as http:
            response = http.get(_TICKERS_URL, headers=_headers(), timeout=_TIMEOUT_S)
            response.raise_for_status()
        payload = response.json()
        _write_cache(path, payload)

    wanted = ticker.upper()
    for entry in payload.values():
        if str(entry.get("ticker", "")).upper() == wanted:
            return f"{int(entry['cik_str']):010d}"
    raise KeyError(f"no SEC CIK for ticker {ticker!r}")


def _client(client: httpx.Client | None) -> httpx.Client:
    """Use the caller's client (tests inject one) or a fresh default client."""
    return client if client is not None else httpx.Client()


def fetch_companyfacts(
    ticker: str,
    *,
    client: httpx.Client | None = None,
    cache: str | Path | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Raw ``companyfacts`` payload for ``ticker`` (cached per symbol)."""
    directory = cache_dir(cache)
    path = directory / f"{ticker.upper()}.json"
    payload = None if refresh else _read_cache(path)
    if payload is None:
        cik = cik_for_ticker(ticker, client=client, cache=directory)
        with _client(client) as http:
            response = http.get(
                _FACTS_URL.format(cik=int(cik)),
                headers=_headers(),
                timeout=_TIMEOUT_S,
            )
            response.raise_for_status()
        payload = response.json()
        _write_cache(path, payload)
    return payload


async def download_fundamentals(
    symbols: Sequence[str],
    batch_delay_s: float = 0.2,
    *,
    client: httpx.Client | None = None,
    cache: str | Path | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Fetch ``companyfacts`` payloads for ``symbols``, keyed by symbol.

    Symbols SEC has no CIK for, or whose payload fails, are **omitted** rather
    than aborting the batch: ``dl`` over a mixed universe (ETFs alongside
    issuers) should download what exists. The caller sees which keys are absent.

    HTTP runs synchronously inside the async wrapper on purpose — SEC asks for a
    serialized, paced caller, so concurrency here would only risk a ban.
    """
    out: dict[str, Any] = {}
    for i, symbol in enumerate(symbols):
        if i:
            await asyncio.sleep(batch_delay_s)
        try:
            out[symbol.upper()] = fetch_companyfacts(
                symbol, client=client, cache=cache, refresh=refresh
            )
        except httpx.HTTPError, KeyError:
            continue
    return out


__all__ = [
    "cik_for_ticker",
    "fetch_companyfacts",
    "download_fundamentals",
    "cache_dir",
]
