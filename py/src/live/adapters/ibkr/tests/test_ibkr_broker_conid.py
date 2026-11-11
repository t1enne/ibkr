"""``_default_conid_lookup`` verification: never order on an unverified contract.

The secdef search routinely returns multiple listings of one ticker; the lookup
must refuse ambiguity and refuse a contract that does not name the ticker or is
restricted — a wrong conid is a wrong instrument. It is also VENUE-scoped: one
ticker can list on several exchanges (``ENI`` on Milan and Paris), so the venue is
half of the question and the candle DB's ``symbol`` row (looked up by SERIES KEY)
is the tie-break between two rows on the same venue.

``search_contracts`` is stubbed with the SDK's own candidate objects — the same
shape the real search returns, never a hand-built gateway dict — and the DB tests
bind ``SymbolSchema`` to a temp sqlite file, so the repo's ``data/ibkr.db`` is
never read or written.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import peewee
import pytest
from ib_rest_api_client.models import SecdefSearchResponseItem

from src.data.types import SymbolSchema
from src.live.adapters.ibkr import broker as broker_mod


def _contract(
    conid: str,
    symbol: str,
    *,
    venue: str = "NASDAQ",
    restricted: bool | None = None,
) -> SecdefSearchResponseItem:
    """A search candidate: ``description`` is the venue the row is listed on."""
    return SecdefSearchResponseItem(
        conid=conid, symbol=symbol, description=venue, restricted=restricted
    )


async def _candidates(
    *items: SecdefSearchResponseItem,
) -> tuple[SecdefSearchResponseItem, ...]:
    return items


def _stub_search(
    monkeypatch: pytest.MonkeyPatch,
    candidates: tuple[SecdefSearchResponseItem, ...],
    seen: list[str | None] | None = None,
) -> None:
    """Serve *candidates*, filtered by the requested venue, recording each ask.

    Filtering here mirrors the real ``search_contracts``: it applies the venue
    filter and raises the same "No <venue> listing found" refusal on an empty set,
    so a lookup that asks the wrong venue gets a wrong (or empty) answer and the
    test can see it.
    """

    async def _search(
        ticker: str, venue: str | None = None
    ) -> tuple[SecdefSearchResponseItem, ...]:
        if seen is not None:
            seen.append(venue)
        wanted = "US stock contract" if venue is None else f"{venue.upper()} listing"
        found = tuple(
            c
            for c in candidates
            if c.description == ("NASDAQ" if venue is None else venue)
        )
        if not found:
            raise ValueError(f"No {wanted} found for {ticker}")
        return found

    monkeypatch.setattr(broker_mod, "search_contracts", _search)


@pytest.fixture()
def symbol_db(tmp_path: Path) -> Iterator[peewee.SqliteDatabase]:
    """A temp sqlite file with the ``symbol`` table bound to the real model."""
    database = peewee.SqliteDatabase(str(tmp_path / "symbols.sqlite"))
    with database.bind_ctx((SymbolSchema,)):
        database.create_tables((SymbolSchema,))
        yield database


@pytest.mark.asyncio
async def test_default_conid_lookup_refuses_ambiguity_without_a_db_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two distinct conids also naming the ticker and NEITHER matches the DB conid:
    # refuse, naming each candidate's description, rather than pick one.
    monkeypatch.setattr(
        broker_mod,
        "search_contracts",
        lambda _t, _venue=None: _candidates(
            _contract("1", "AAPL"), _contract("2", "AAPL")
        ),
    )
    monkeypatch.setattr(broker_mod, "_db_conid_for_listing", lambda _t, _v: None)
    with pytest.raises(ValueError, match="ambiguous"):
        await broker_mod._default_conid_lookup("AAPL")


@pytest.mark.asyncio
async def test_default_conid_lookup_refuses_a_mismatched_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The search for AAPL returned a contract for MSFT: never order on it.
    monkeypatch.setattr(
        broker_mod,
        "search_contracts",
        lambda _t, _venue=None: _candidates(_contract("1", "MSFT")),
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
        lambda _t, _venue=None: _candidates(_contract("1", "AAPL", restricted=True)),
    )
    with pytest.raises(ValueError, match="restricted"):
        await broker_mod._default_conid_lookup("AAPL")


@pytest.mark.asyncio
async def test_a_cross_listed_ticker_resolves_the_requested_venue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ENI`` lists in Milan and Paris: the VENUE decides which conid is meant."""
    seen: list[str | None] = []
    _stub_search(
        monkeypatch,
        (
            _contract("100", "ENI", venue="NYSE"),
            _contract("200", "ENI", venue="BVME"),
            _contract("300", "ENI", venue="SBF"),
        ),
        seen,
    )

    assert await broker_mod._default_conid_lookup("ENI", "BVME") == 200
    assert await broker_mod._default_conid_lookup("ENI", "SBF") == 300
    assert await broker_mod._default_conid_lookup("ENI", "NYSE") == 100
    assert seen == ["BVME", "SBF", "NYSE"]


@pytest.mark.asyncio
async def test_a_venue_with_no_listing_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No listing on the asked venue is a refusal, never another venue's conid."""
    _stub_search(monkeypatch, (_contract("100", "ENI", venue="NYSE"),))

    with pytest.raises(ValueError, match="No BVME listing"):
        await broker_mod._default_conid_lookup("ENI", "BVME")


@pytest.mark.db
@pytest.mark.asyncio
async def test_a_same_venue_duplicate_still_resolves_via_the_db_tie_break(
    symbol_db: peewee.SqliteDatabase,
) -> None:
    """Two US rows for ``SHOP``: the candle DB's listing row picks the contract.

    This is the tie-break the plan keeps. It now matches the LISTING row (the
    series key ``SHOP``) rather than the bare ticker, so a cross-listed ticker
    cannot be tie-broken onto another venue's conid.
    """
    SymbolSchema.create(conid=2, ticker="SHOP", market="SMART", currency="USD")

    with pytest.MonkeyPatch.context() as patch:
        _stub_search(
            patch,
            (
                _contract("1", "SHOP", venue="NASDAQ"),
                _contract("2", "SHOP", venue="NASDAQ"),
            ),
        )
        assert await broker_mod._default_conid_lookup("SHOP") == 2


@pytest.mark.db
def test_the_db_tie_break_reads_the_venue_series_key(
    symbol_db: peewee.SqliteDatabase,
) -> None:
    """``ENI`` on ``BVME`` is looked up as ``ENI.MI`` — never as the bare ticker."""
    SymbolSchema.create(conid=200, ticker="ENI.MI", market="BVME", currency="EUR")
    SymbolSchema.create(conid=100, ticker="ENI", market="SMART", currency="USD")

    assert broker_mod._db_conid_for_listing("ENI", "BVME") == 200
    assert broker_mod._db_conid_for_listing("ENI", None) == 100
    # A venue with no series-key rule is no preference, not a wrong row.
    assert broker_mod._db_conid_for_listing("ENI", "TSX") is None
