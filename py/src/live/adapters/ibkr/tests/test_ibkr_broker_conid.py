"""``_default_conid_lookup`` verification: never order on an unverified contract.

The secdef search routinely returns multiple listings of one ticker; the lookup
must refuse ambiguity and refuse a contract that does not name the ticker or is
restricted — a wrong conid is a wrong instrument.
"""

from __future__ import annotations

import pytest
from ib_rest_api_client.models import SecdefSearchResponseItem

from src.live.adapters.ibkr import broker as broker_mod


def _contract(
    conid: str, symbol: str, *, restricted: bool | None = None
) -> SecdefSearchResponseItem:
    """A US-stock search candidate (primary exchange NASDAQ passes ``_is_usd_stock``)."""
    return SecdefSearchResponseItem(
        conid=conid, symbol=symbol, description="NASDAQ", restricted=restricted
    )


async def _candidates(
    *items: SecdefSearchResponseItem,
) -> tuple[SecdefSearchResponseItem, ...]:
    return items


@pytest.mark.asyncio
async def test_default_conid_lookup_refuses_ambiguity_without_a_db_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two distinct conids also naming the ticker and NEITHER matches the DB conid:
    # refuse, naming each candidate's description, rather than pick one.
    monkeypatch.setattr(
        broker_mod,
        "search_contracts",
        lambda _t: _candidates(_contract("1", "AAPL"), _contract("2", "AAPL")),
    )
    monkeypatch.setattr(broker_mod, "_db_conid_for_ticker", lambda t: None)
    with pytest.raises(ValueError, match="ambiguous"):
        await broker_mod._default_conid_lookup("AAPL")


@pytest.mark.asyncio
async def test_default_conid_lookup_refuses_a_mismatched_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The search for AAPL returned a contract for MSFT: never order on it.
    monkeypatch.setattr(
        broker_mod, "search_contracts", lambda _t: _candidates(_contract("1", "MSFT"))
    )
    with pytest.raises(ValueError, match="naming AAPL"):
        await broker_mod._default_conid_lookup("AAPL")


@pytest.mark.asyncio
async def test_default_conid_lookup_refuses_a_restricted_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        broker_mod,
        "search_contracts",
        lambda _t: _candidates(_contract("1", "AAPL", restricted=True)),
    )
    with pytest.raises(ValueError, match="restricted"):
        await broker_mod._default_conid_lookup("AAPL")
