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


def _is_usd_stock(entry: SecdefSearchResponseItem) -> bool:
    desc = entry.description
    if isinstance(desc, str) and desc in US_EXCHANGES:
        return True
    # Also match via sections: US stocks have 'STK' section
    if entry.sections:
        for s in entry.sections:
            if hasattr(s, "sec_type") and s.sec_type == "STK":
                return True
    return False


async def search_contracts(ticker: str) -> tuple[SecdefSearchResponseItem, ...]:
    """Every US-stock secdef candidate for *ticker* (0..n, exchange-filtered).

    The shared gateway search: ``lookup`` takes the first candidate; the live
    conid resolver refuses an ambiguous set. Raises ``ValueError`` on a transport
    error or an empty candidate set — never returns a partial result.
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

        data = tuple(item for item in r if _is_usd_stock(item))
        if not data:
            raise ValueError(f"No US stock contract found for {ticker}")

        return data
    except Exception as e:
        raise ValueError(f"Failed to search contract for {ticker}: {e}")


async def lookup(ticker: str) -> SecdefSearchResponseItem:
    """Resolve *ticker* to its first US-stock candidate (backtest data path)."""
    return (await search_contracts(ticker))[0]


__all__ = ["lookup", "search_contracts"]
