"""Contract search over the gateway's secdef endpoint.

``search_contracts`` is the shared candidate fetch; callers pick a listing:

- no ``exchange`` — the historical US-stock filter (the backtest data path for a
  bare US ticker);
- ``exchange`` — an IBKR venue code (``BVME``, ``IBIS``…), so a non-US market can
  be resolved by ticker instead of only by a conid seeded into the DB.

A venue code is required for non-US markets because one ticker can be listed on
several exchanges at once and the gateway returns all of them; nothing in the
symbol string says which one the caller meant.
"""

from __future__ import annotations

import re

from ib_rest_api_client.models import (
    ErrorOnlyResponse,
    SecdefSearchResponseItem,
)
from ib_rest_api_client.api.trading_contracts import get_iserver_secdef_search
from src.data.ibkr.client import default_client


US_EXCHANGES: frozenset[str] = frozenset(
    {
        "ARCA",
        "NASDAQ",
        "NYSE",
        "AMEX",
        "BATS",
        "IEX",
        "BATY",
        "ARCAEDGE",
        "EDGEA",
        "NYSEAMERICAN",
        "NASDAQNM",
    }
)

#: Trailing venue token of a name-search header — ``ENI SPA (BVME)`` or
#: ``ENI SPA - BVME``.
_HEADER_VENUE = re.compile(r"[-(]\s*([A-Z][A-Z0-9.]{1,7})\s*\)?\s*$")


def _is_usd_stock(entry: SecdefSearchResponseItem) -> bool:
    """True when the candidate is a US-listed stock.

    A named venue decides: ``VALUE``, ``BVME`` and friends are listings too, so
    falling through to the sections check for THOSE would let a foreign listing
    answer a US lookup. The sections fallback exists only for a row that names
    no venue at all.
    """
    desc = entry.description
    if isinstance(desc, str) and desc.strip():
        return desc in US_EXCHANGES
    # Venue-less row: the STK section is the only evidence available.
    if entry.sections:
        return any(getattr(s, "sec_type", None) == "STK" for s in entry.sections)
    return False


def listing_exchange(entry: SecdefSearchResponseItem) -> str | None:
    """The venue a secdef candidate is LISTED on.

    The gateway serves two shapes: a symbol-search row names the venue in
    ``description`` (``BVME``, ``NYSE``…), while a name-search row leaves it null
    and names it in the header's trailing token (``ENI SPA (BVME)``). Reading
    only the first shape would silently drop every name-search hit.
    """
    desc = entry.description
    if isinstance(desc, str) and desc.strip():
        return desc.split(";")[0].strip()
    header = entry.company_header
    if isinstance(header, str):
        match = _HEADER_VENUE.search(header)
        if match:
            return match.group(1)
    return None


def select_listings(
    items: list[SecdefSearchResponseItem],
    exchange: str | None = None,
) -> tuple[SecdefSearchResponseItem, ...]:
    """Filter secdef candidates to one venue (the US set when *exchange* is unset).

    Empty when nothing matches — the caller decides whether that is a refusal.
    """
    if exchange is None:
        return tuple(item for item in items if _is_usd_stock(item))
    wanted = exchange.strip().upper()
    return tuple(
        item for item in items if (listing_exchange(item) or "").upper() == wanted
    )


async def search_contracts(
    ticker: str,
    exchange: str | None = None,
) -> tuple[SecdefSearchResponseItem, ...]:
    """Every stock secdef candidate for *ticker* (0..n, listing-filtered).

    With *exchange* only that venue's listing is returned; without it the
    historical US-only filter applies. ``lookup`` takes the first candidate; the
    live conid resolver refuses an ambiguous set. Raises ``ValueError`` on a
    transport error or an empty candidate set — never returns a partial result.
    """
    try:
        r = await get_iserver_secdef_search.asyncio(
            client=default_client().rest,
            symbol=ticker,
        )
        if isinstance(r, ErrorOnlyResponse):
            raise ValueError(f"Failed to search contract for {ticker}")
        if not isinstance(r, list):
            raise ValueError(f"Failed to search contract for {ticker}")

        data = select_listings(r, exchange)
        if not data:
            wanted = (
                "US stock contract"
                if exchange is None
                else f"{exchange.upper()} listing"
            )
            raise ValueError(f"No {wanted} found for {ticker}")
        return data
    except ValueError:
        raise  # our own refusal (empty/ambiguous set) already carries the reason
    except Exception as e:
        raise ValueError(f"Failed to search contract for {ticker}: {e}") from e


async def lookup(
    ticker: str,
    exchange: str | None = None,
) -> SecdefSearchResponseItem:
    """Resolve *ticker* to its first candidate on *exchange* (US when unset)."""
    return (await search_contracts(ticker, exchange))[0]


__all__ = ["lookup", "search_contracts", "listing_exchange", "select_listings"]
