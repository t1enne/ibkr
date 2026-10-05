"""Contract lookup helpers — routed through the one gateway client.

Historically this module held its OWN module-level ``httpx.AsyncClient`` (plus an
``ib_rest_api_client.Client``) with a hardcoded base url — a second client with
its own policy next to the one in ``candles.py``. Both now come from
``src.data.ibkr.client.default_client()``: one base url, one verify policy, one
error mapping.
"""

from __future__ import annotations

from typing import Any, Dict

from src.data.ibkr.client import IbkrError, default_client
from src.data.types import ISymbol, SymbolSchema


async def fetch_contract_info(conid: int) -> Dict[str, Any]:
    ep = f"iserver/contract/{conid}/info"
    try:
        return await default_client().get(ep)
    except IbkrError as e:
        raise ValueError(f"Failed call to {ep}: {e}") from e


async def get_contract_info(conid: int) -> ISymbol:
    # Query DB first
    try:
        symbol = SymbolSchema.get(SymbolSchema.conid == conid)
        return symbol
    except SymbolSchema.DoesNotExist:
        pass

    # Fetch from API
    cinfo = await fetch_contract_info(conid)

    # Insert into DB
    symbol = SymbolSchema.create(
        conid=conid,
        ticker=cinfo["symbol"],
        name=cinfo.get("company_name"),
        market=cinfo["exchange"],
        currency=cinfo["currency"],
    )
    return symbol
