"""Contract lookup helpers — routed through the one gateway client.

Historically this module held its OWN module-level ``httpx.AsyncClient`` (plus an
``ib_rest_api_client.Client``) with a hardcoded base url — a second client with
its own policy next to the one in ``candles.py``. Both now come from
``src.data.ibkr.client.default_client()``: one base url, one verify policy, one
error mapping.

:func:`get_contract_info` is the ONLY path that inserts a ``symbol`` row, so it
is where the series-key invariants are enforced: a venue with no suffix rule and
a key already owned by another conid are both refused rather than stored. A
second listing under one key would leave its calendar, its currency and its bars
ambiguous, and nothing downstream could tell which one won.
"""

from __future__ import annotations

from typing import Any, Dict

from src.data.ibkr.client import IbkrError, default_client
from src.data.types import ISymbol, SymbolSchema
from src.data.venue import require_keyable, series_key


async def fetch_contract_info(conid: int) -> Dict[str, Any]:
    ep = f"iserver/contract/{conid}/info"
    try:
        return await default_client().get(ep)
    except IbkrError as e:
        raise ValueError(f"Failed call to {ep}: {e}") from e


async def get_contract_info(
    conid: int,
    market: str | None = None,
    ticker: str | None = None,
) -> ISymbol:
    """Resolve *conid* to its stored symbol row, fetching and persisting on miss.

    *market* is the venue the caller resolved the conid FOR (a universe's
    ``exchange``). It is stored in preference to the contract's own ``exchange``
    field, which reports the routing venue (``SMART``) and so never says where a
    non-US stock actually lists — that column drives the trading calendar, and
    ``SMART`` would silently mean New York.

    *ticker* overrides the label stored in the row, so a caller that names its
    symbols by series key (a universe listing ``ENI.MI``) keeps that name and the
    row stays findable by the string its own config uses. The stored value is
    normalised through :func:`src.data.venue.series_key`, so ``ENI`` on ``BVME``
    and ``ENI.MI`` on ``BVME`` land on the same key.

    Raises:
        UnmappedVenueError: the venue has no series-key suffix rule.
        ValueError: the derived key already belongs to a different conid.
    """
    symbol = SymbolSchema.get_or_none(SymbolSchema.conid == conid)
    if symbol is not None:
        return symbol

    require_keyable(market)
    cinfo = await fetch_contract_info(conid)

    venue = (market or cinfo["exchange"]).strip().upper()
    key = series_key(ticker or cinfo["symbol"], venue)

    owner = SymbolSchema.get_or_none(SymbolSchema.ticker == key)
    if owner is not None:
        raise ValueError(
            f"series key {key} already belongs to conid {owner.conid} "
            f"(venue {owner.market}); refusing to point it at conid {conid}"
        )

    return SymbolSchema.create(
        conid=conid,
        ticker=key,
        name=cinfo.get("company_name"),
        market=venue,
        currency=cinfo["currency"],
    )
