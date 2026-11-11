"""Symbol resolution and universe configuration loading.

Shared primitives used by the `data dl` and `data preview` commands.
These used to live in the monolithic sync module; they are split out so
each command module imports only what it needs.

Resolution is keyed by the SERIES KEY, never by the raw input string: a ticker
plus its venue derives the key that ``symbol.ticker`` holds (see
:mod:`src.data.venue`), and that key is what the DB row, the gateway search and
the candle rows all agree on. An unmapped venue raises here rather than being
stored as a bare ticker.

This module is I/O-bound (DB lookups + IBKR API fallback) but keeps the
per-ticker logic private. The public surface:

  resolve_symbols       — ticker strings → resolved ISymbol objects
  validate_tickers      — ticker strings → their series keys (raises if unmapped)
  market_for_ticker     — a stored series key's venue, for calendar selection
  load_universe_config  — load/validate a universe .json file
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from src.data.ibkr import get_contract_info, lookup
from src.data.ibkr.lookup import listing_exchange
from src.data.types import ISymbol, SymbolSchema, UniverseConf
from src.data.venue import bare_ticker, is_bare_venue, series_key, venue_for_key

logger = logging.getLogger(__name__)


class VenueMismatchError(ValueError):
    """A resolved candidate is listed on a venue the series key does not name.

    Distinct from the per-ticker failures ``resolve_symbols`` logs and skips: a
    gateway error loses one symbol for one run, whereas this would store a real
    contract's bars under another listing's key. The caller must seed the row or
    ask the venue the ticker actually lists on.
    """


# ── Per-ticker resolution ──────────────────────────────────────


def _api_lookup_ticker(ticker: str, exchange: str | None) -> str:
    """The string to hand the gateway search for *ticker*.

    A universe may name its symbols by series key (``ENI.MI``) — a local
    namespace label that keeps an Italian listing apart from a same-named US
    one. The gateway knows only IBKR's own symbol, so on a non-US venue the
    known suffix is dropped for the search (see
    :func:`src.data.venue.bare_ticker`); the series key is what gets stored. A
    US lookup keeps the string untouched, because a US ticker may legitimately
    contain a dot (``BRK.B``).

    "Non-US" is decided by the venue the key RESOLVES against
    (:func:`src.data.venue.venue_for_key`): a suffixed key is never searched on
    the US path even when the universe declares no ``exchange``.
    """
    if is_bare_venue(venue_for_key(ticker, exchange)):
        return ticker
    return bare_ticker(ticker)


def validate_tickers(tickers: list[str], exchange: str | None) -> tuple[str, ...]:
    """Every ticker's series key, in input order.

    One validator for both callers: universe loading (a bad venue is a config
    error) and resolution (a bad venue is never a per-ticker skip, which would
    silently drop the symbol from a download).

    Raises:
        UnmappedVenueError: the venue has no series-key suffix rule.
        ValueError: a ticker is empty.
    """
    return tuple(series_key(t, exchange) for t in tickers)


def _venue_derives_key(key: str, listed_venue: str | None) -> str:
    """The key *listed_venue* derives from *key*'s bare ticker.

    Comparing this with *key* is how resolution checks that the candidate a
    search returned is actually LISTED where the key says: a ``.PA`` key resolved
    against a ``BVME`` search would otherwise store the Milan conid under the
    Paris key. The venue comes from the candidate, never from the key.
    """
    return series_key(bare_ticker(key), listed_venue)


async def _get_symbol_for_key(key: str, exchange: str | None = None) -> ISymbol:
    """Resolve a series KEY to its listing (DB lookup → API fallback).

    The DB is authoritative: a stored row names the exact conid, which is the
    only way to disambiguate a ticker listed on several venues. Otherwise the key
    is resolved against the venue it NAMES — its suffix's canonical venue, or
    *exchange* for an unsuffixed key (see
    :func:`src.data.venue.venue_for_key`) — and that is the venue the new row
    records.

    Raises:
        VenueMismatchError: the gateway's candidate is listed on a venue *key*
            does not name (seed the row, or run that venue).
    """
    s = SymbolSchema.get_or_none(SymbolSchema.ticker == key)
    if s:
        return s

    venue = venue_for_key(key, exchange)
    logger.info("looking up %s via API (market=%s)", key, venue or "US")
    contract = await lookup(_api_lookup_ticker(key, venue), venue)
    listed = listing_exchange(contract)
    if _venue_derives_key(key, listed) != key:
        raise VenueMismatchError(
            f"{key} resolved to a {listed or 'US'} listing, which derives "
            f"{_venue_derives_key(key, listed)}; seed the symbol row for {key} or "
            "run the universe with the venue it lists on"
        )
    conid = int(contract.conid or "-")
    symbol_info = await get_contract_info(conid, market=venue, ticker=key)
    return symbol_info


async def resolve_symbols(
    tickers: list[str],
    exchange: str | None = None,
) -> list[ISymbol]:
    """Resolve ticker strings to ISymbol objects (DB lookup + API fallback).

    *exchange* is the default venue; a ticker that already carries a known
    suffix resolves against its OWN venue (``ENI.PA`` in a ``BVME`` universe
    asks ``SBF``), which is what makes a pan-European universe a plain list.
    Without *exchange* the historical US-only path applies, except where a suffix
    names a venue.

    Returns only successfully resolved symbols: a per-ticker API failure is
    logged and skipped. Two failures are NOT skippable, because both would
    quietly produce a series nobody asked for — an unmapped venue (a config
    error, see :func:`validate_tickers`) and a candidate listed on the wrong
    venue (:class:`VenueMismatchError`).

    Raises:
        UnmappedVenueError: *exchange* (or a ticker's own venue) has no rule.
        VenueMismatchError: the gateway answered from a venue the key disowns.
    """
    keys = validate_tickers(tickers, exchange)
    semaphore = asyncio.Semaphore(1)

    async def bounded(key: str) -> Optional[ISymbol]:
        async with semaphore:
            try:
                return await _get_symbol_for_key(key, exchange)
            except VenueMismatchError:
                raise
            except Exception as e:
                logger.warning("Error resolving %s: %s", key, e)
                return None

    results = await asyncio.gather(*[bounded(k) for k in keys])
    return [s for s in results if s is not None]


def market_for_ticker(ticker: str) -> str | None:
    """The stored venue for series key *ticker*, or ``None`` when not in the DB.

    A read-only seam for the calendar paths: whether a date owes bars depends on
    the venue, and the DB row is the only place the venue is recorded.
    """
    row = SymbolSchema.get_or_none(SymbolSchema.ticker == ticker.upper())
    return row.market if row is not None else None


# ── Universe configuration ─────────────────────────────────────


def load_universe_config(file_path: str) -> UniverseConf:
    """Load and validate a universe configuration JSON file.

    Validates:
      - File exists and is readable
      - Content is a JSON object (not array, scalar, or empty)
      - Required 'symbols' key is present and non-empty
      - Optional 'exchange', when present, is a non-empty string
      - Every symbol derives a series key from its venue

    Args:
        file_path: Path to the universe .json file

    Returns:
        Frozen UniverseConf with validated fields.

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the content is malformed or missing required fields
        UnmappedVenueError: If 'exchange' has no series-key rule
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Universe config not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid universe config: expected a JSON object, "
            f"got {type(data).__name__}"
        )

    if "symbols" not in data:
        raise ValueError(
            "Universe config missing required 'symbols' field. "
            "Expected format:\n"
            '  { "symbols": ["AAPL", "MSFT"] }\n'
        )

    symbols = data.get("symbols", [])
    if not isinstance(symbols, list) or len(symbols) == 0:
        raise ValueError(
            "Universe config 'symbols' must be a non-empty list of tickers"
        )

    exchange = data.get("exchange")
    if exchange is not None and (not isinstance(exchange, str) or not exchange.strip()):
        raise ValueError(
            f"Universe config 'exchange' must be a non-empty venue code "
            f"(e.g. 'BVME'), got {exchange!r}"
        )

    venue_code = exchange.strip().upper() if exchange else None
    validate_tickers(symbols, venue_code)

    return UniverseConf(
        symbols=[sym.upper() for sym in symbols],
        from_date=data.get("from_date"),
        to_date=data.get("to_date"),
        bar=data.get("bar", "1h"),
        exchange=venue_code,
    )
