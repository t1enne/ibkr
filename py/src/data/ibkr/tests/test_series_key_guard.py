"""The series-key write guards: one key, one contract.

Both write paths are tested here, because each has a different way to lie:

* ``get_contract_info`` inserts the ``symbol`` row, so it refuses a key another
  conid already owns and a venue with no key rule;
* ``_chunked_insert`` writes candle rows with ``on_conflict_ignore``, which turns
  a cross-conid collision into a silent no-op — exactly the shape of a
  "successful" re-download whose bars never arrive.

Every test binds the real models to a temp sqlite file; the repo DB is never
touched.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import peewee
import pytest

from src.data.ibkr import shared
from src.data.ibkr.candles import _chunked_insert
from src.data.types import CandleDict, CandleSchema, SymbolSchema
from src.data.venue import UnmappedVenueError

pytestmark = pytest.mark.db


@pytest.fixture()
def temp_db(tmp_path: Path) -> Iterator[peewee.SqliteDatabase]:
    """A temp sqlite file with ``symbol`` and ``candle`` bound to their models."""
    database = peewee.SqliteDatabase(str(tmp_path / "guard.sqlite"))
    models = (SymbolSchema, CandleSchema)
    with database.bind_ctx(models):
        database.create_tables(models)
        yield database


def _gateway_info(conid: int, exchange: str, currency: str) -> dict[str, object]:
    """The gateway payload ``get_contract_info`` consumes (no HTTP)."""
    return {
        "conid": conid,
        "symbol": "EA",
        "exchange": exchange,
        "currency": currency,
        "company_name": "ELECTRONIC ARTS INC",
    }


def test_a_key_owned_by_another_conid_is_refused(temp_db, monkeypatch):
    """The MEXI listing already owns ``EA.MX``; a second conid must not take it."""
    SymbolSchema.create(conid=295338948, ticker="EA.MX", market="MEXI", currency="MXN")

    async def _fetch(conid: int) -> dict[str, object]:
        return _gateway_info(conid, "MEXI", "MXN")

    monkeypatch.setattr(shared, "fetch_contract_info", _fetch)

    with pytest.raises(ValueError, match="EA.MX already belongs to conid 295338948"):
        asyncio.run(
            shared.get_contract_info(conid=268995, market="MEXI", ticker="EA.MX")
        )

    assert SymbolSchema.select().count() == 1, "the refused row must not be written"


def test_a_venue_with_no_key_rule_is_refused_before_the_fetch(temp_db, monkeypatch):
    """An unmapped venue is a config error: refuse it without a gateway call."""

    async def _boom(_conid: int) -> dict[str, object]:
        raise AssertionError("an unmapped venue must be refused before any I/O")

    monkeypatch.setattr(shared, "fetch_contract_info", _boom)

    with pytest.raises(UnmappedVenueError, match="TSX"):
        asyncio.run(shared.get_contract_info(conid=1, market="TSX", ticker="SHOP"))

    assert SymbolSchema.select().count() == 0


def test_a_stored_conid_is_returned_without_a_fetch(temp_db, monkeypatch):
    """Re-resolving a known conid is a read: an idempotent re-download must not
    pay for the contract info call again."""
    SymbolSchema.create(conid=1, ticker="AAPL", market="SMART", currency="USD")

    async def _boom(_conid: int) -> dict[str, object]:
        raise AssertionError("a stored conid needs no contract lookup")

    monkeypatch.setattr(shared, "fetch_contract_info", _boom)

    resolved = asyncio.run(shared.get_contract_info(conid=1))

    assert (resolved.ticker, resolved.market) == ("AAPL", "SMART")


def test_the_inserted_key_is_the_venues_key_not_the_contracts_exchange(
    temp_db, monkeypatch
):
    """The stored venue is the caller's, and the key follows it.

    The contract's own ``exchange`` field reports the routing venue (``SMART``),
    which would make every foreign series look American.
    """

    async def _fetch(conid: int) -> dict[str, object]:
        return _gateway_info(conid, "SMART", "EUR")

    monkeypatch.setattr(shared, "fetch_contract_info", _fetch)

    stored = asyncio.run(shared.get_contract_info(conid=7, market="BVME", ticker="ENI"))

    assert (stored.ticker, stored.market) == ("ENI.MI", "BVME")


def _candle(conid: int, timestamp: int) -> CandleDict:
    return {
        "conid": conid,
        "ticker": "EA.MX",
        "timestamp": timestamp,
        "open": 1.0,
        "high": 2.0,
        "low": 0.5,
        "close": 1.5,
        "volume": 100.0,
    }


def test_a_candle_write_refuses_another_conids_series(temp_db):
    """``on_conflict_ignore`` would swallow this batch and look like success."""
    _chunked_insert([_candle(295338948, 1_700_000_000_000)])

    with pytest.raises(ValueError, match="EA.MX already belongs to conid 295338948"):
        _chunked_insert([_candle(268995, 1_700_003_600_000)])

    assert CandleSchema.select().count() == 1


def test_a_candle_write_for_the_same_conid_stays_silent(temp_db):
    """The normal re-download path writes the same conid again — no refusal."""
    _chunked_insert([_candle(295338948, 1_700_000_000_000)])
    _chunked_insert([_candle(295338948, 1_700_000_000_000)])

    assert CandleSchema.select().where(CandleSchema.conid == 295338948).count() >= 1
