"""Venue-scoped ticker resolution: series key first, then the exchange-filtered API.

Resolution is keyed by the SERIES KEY, not by the input string: ``ENI`` on
``BVME`` and ``ENI.MI`` are one key, and the DB row is found by that key. The API
path is where a universe stops depending on a pre-seeded `symbol` row. ``lookup``
and ``get_contract_info`` are stubbed in the pure tests — those pin the argument
wiring (which venue is asked for, which key is stored), not the gateway. The
DB-backed tests bind the real models to a temp sqlite file, never the repo DB.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import peewee
import pytest

from src.data import symbols as symbols_mod
from src.data.types import ISymbol, SymbolSchema
from src.data.venue import UnmappedVenueError


@pytest.mark.asyncio
async def test_known_ticker_comes_from_the_db_without_a_gateway_call(monkeypatch):
    stored = ISymbol(conid=1, ticker="ENI.MI", market="BVME", currency="EUR")

    monkeypatch.setattr(symbols_mod.SymbolSchema, "get_or_none", lambda *_: stored)

    async def _boom(*_args, **_kwargs):
        raise AssertionError("the API must not be consulted for a stored ticker")

    monkeypatch.setattr(symbols_mod, "lookup", _boom)
    resolved = await symbols_mod.resolve_symbols(["ENI.MI"], "BVME")

    assert resolved == [stored]


@pytest.mark.asyncio
async def test_unknown_ticker_is_resolved_on_the_requested_venue(monkeypatch):
    seen: dict[str, object] = {}

    monkeypatch.setattr(symbols_mod.SymbolSchema, "get_or_none", lambda *_: None)

    class _Contract:
        conid = "55764652"
        description = "BVME"

    async def _lookup(ticker: str, exchange: str | None):
        seen["lookup"] = (ticker, exchange)
        return _Contract()

    async def _contract_info(
        conid: int, market: str | None = None, ticker: str | None = None
    ):
        seen["contract_info"] = (conid, market, ticker)
        return ISymbol(
            conid=conid, ticker=ticker or "", market=market or "", currency="EUR"
        )

    monkeypatch.setattr(symbols_mod, "lookup", _lookup)
    monkeypatch.setattr(symbols_mod, "get_contract_info", _contract_info)

    resolved = await symbols_mod.resolve_symbols(["AMP2.MI"], "BVME")

    assert seen["lookup"] == ("AMP2", "BVME")  # venue path drops the namespace suffix
    assert seen["contract_info"] == (55764652, "BVME", "AMP2.MI")
    assert resolved[0].market == "BVME"


def test_us_ticker_keeps_its_dotted_name():
    # No venue declared: a dot is part of the US symbol, never a suffix to strip.
    assert symbols_mod._api_lookup_ticker("BRK.B", None) == "BRK.B"
    assert symbols_mod._api_lookup_ticker("BRK.B", "SMART") == "BRK.B"


def test_a_suffixed_key_searches_for_the_bare_gateway_ticker():
    assert symbols_mod._api_lookup_ticker("ENI.MI", "BVME") == "ENI"
    assert symbols_mod._api_lookup_ticker("ENI", "BVME") == "ENI"


@dataclass(frozen=True)
class _Contract:
    """A minimal secdef candidate: the two fields resolution reads."""

    conid: str
    description: str


@pytest.mark.asyncio
async def test_a_pan_european_universe_resolves_two_listings(monkeypatch):
    """``ENI.MI`` and ``ENI.PA`` in ONE ``BVME`` universe are two conids.

    The blocker this replaces: asking the universe's exchange for both made the
    ``.PA`` key resolve the Milan conid and raised ``VenueMismatchError``, so a
    pan-European universe could not resolve at all. Each key now asks the venue
    its own suffix names.
    """
    seen: list[tuple[str, str | None]] = []
    monkeypatch.setattr(symbols_mod.SymbolSchema, "get_or_none", lambda *_: None)

    async def _lookup(ticker: str, exchange: str | None):
        seen.append((ticker, exchange))
        return {"BVME": _Contract("1", "BVME"), "SBF": _Contract("2", "SBF")}[
            str(exchange)
        ]

    async def _contract_info(
        conid: int, market: str | None = None, ticker: str | None = None
    ):
        return ISymbol(
            conid=conid, ticker=ticker or "", market=market or "", currency="EUR"
        )

    monkeypatch.setattr(symbols_mod, "lookup", _lookup)
    monkeypatch.setattr(symbols_mod, "get_contract_info", _contract_info)

    resolved = await symbols_mod.resolve_symbols(["ENI.MI", "ENI.PA"], "BVME")

    assert [(s.ticker, s.conid, s.market) for s in resolved] == [
        ("ENI.MI", 1, "BVME"),
        ("ENI.PA", 2, "SBF"),
    ]
    assert seen == [("ENI", "BVME"), ("ENI", "SBF")]


@pytest.mark.asyncio
async def test_an_unsuffixed_ticker_still_resolves_on_the_universe_exchange(
    monkeypatch,
):
    """A key with no venue of its own falls back to the declared ``exchange``."""
    seen: list[tuple[str, str | None]] = []
    monkeypatch.setattr(symbols_mod.SymbolSchema, "get_or_none", lambda *_: None)

    async def _lookup(ticker: str, exchange: str | None):
        seen.append((ticker, exchange))
        return _Contract("1", "BVME")

    async def _contract_info(
        conid: int, market: str | None = None, ticker: str | None = None
    ):
        return ISymbol(
            conid=conid, ticker=ticker or "", market=market or "", currency="EUR"
        )

    monkeypatch.setattr(symbols_mod, "lookup", _lookup)
    monkeypatch.setattr(symbols_mod, "get_contract_info", _contract_info)

    resolved = await symbols_mod.resolve_symbols(["ENI"], "BVME")

    assert [(s.ticker, s.market) for s in resolved] == [("ENI.MI", "BVME")]
    assert seen == [("ENI", "BVME")]


@pytest.mark.asyncio
async def test_a_candidate_listed_elsewhere_is_refused_not_stored(monkeypatch):
    """A ``.PA`` key answered by a Milan listing must not store the Milan conid.

    ``ENI.PA`` is asked of ``SBF``; a gateway that answers with a ``BVME``
    candidate anyway derives ``ENI.MI`` from it, so the candidate is refused
    rather than stored under the wrong key. Refusing beats storing foreign bars.
    """
    monkeypatch.setattr(symbols_mod.SymbolSchema, "get_or_none", lambda *_: None)
    asked: list[str | None] = []

    async def _lookup(_ticker: str, exchange: str | None):
        asked.append(exchange)
        return _Contract("1", "BVME")

    async def _boom(*_args, **_kwargs):
        raise AssertionError(
            "a mis-venue candidate must be refused before it is stored"
        )

    monkeypatch.setattr(symbols_mod, "lookup", _lookup)
    monkeypatch.setattr(symbols_mod, "get_contract_info", _boom)

    with pytest.raises(ValueError, match="ENI.PA resolved to a BVME listing"):
        await symbols_mod.resolve_symbols(["ENI.PA"], "BVME")
    assert asked == ["SBF"], "the suffix's own venue is what gets asked"


@pytest.mark.asyncio
async def test_a_gateway_failure_loses_only_that_symbol(monkeypatch):
    """One unresolvable ticker is logged and skipped, never fatal for the run."""
    monkeypatch.setattr(symbols_mod.SymbolSchema, "get_or_none", lambda *_: None)

    async def _no_listing(_ticker: str, _exchange: str | None):
        raise ValueError("No US stock contract found for ZZZZ")

    monkeypatch.setattr(symbols_mod, "lookup", _no_listing)

    assert await symbols_mod.resolve_symbols(["ZZZZ"]) == []


@pytest.mark.asyncio
async def test_an_unmapped_venue_raises_instead_of_resolving_bare():
    # A bad venue is a configuration error, not one bad symbol to log and skip:
    # a silently bare key is how foreign bars end up under a US name.
    with pytest.raises(UnmappedVenueError, match="TSX"):
        await symbols_mod.resolve_symbols(["SHOP"], "TSX")


@pytest.fixture()
def symbol_db(tmp_path: Path) -> Iterator[peewee.SqliteDatabase]:
    """A temp sqlite file with the ``symbol`` table bound to the real model."""
    database = peewee.SqliteDatabase(str(tmp_path / "symbols.sqlite"))
    with database.bind_ctx((SymbolSchema,)):
        database.create_tables((SymbolSchema,))
        yield database


@pytest.mark.db
@pytest.mark.asyncio
async def test_two_venues_of_one_ticker_resolve_to_two_listings(symbol_db):
    """One ticker, two venues, two conids: the DB row is what disambiguates."""
    SymbolSchema.create(conid=1, ticker="ENI.MI", market="BVME", currency="EUR")
    SymbolSchema.create(conid=2, ticker="ENI.PA", market="SBF", currency="EUR")

    resolved = await symbols_mod.resolve_symbols(["ENI.MI", "ENI.PA"], "BVME")

    assert [(s.ticker, s.conid, s.market) for s in resolved] == [
        ("ENI.MI", 1, "BVME"),
        ("ENI.PA", 2, "SBF"),
    ]


@pytest.mark.db
@pytest.mark.asyncio
async def test_a_bare_ticker_resolves_through_its_venue_key(symbol_db):
    """``ENI`` in a ``BVME`` universe is the ``ENI.MI`` listing, not a US one."""
    SymbolSchema.create(conid=1, ticker="ENI.MI", market="BVME", currency="EUR")
    SymbolSchema.create(conid=3, ticker="ENI", market="SMART", currency="USD")

    resolved = await symbols_mod.resolve_symbols(["ENI"], "BVME")

    assert [(s.ticker, s.conid) for s in resolved] == [("ENI.MI", 1)]
