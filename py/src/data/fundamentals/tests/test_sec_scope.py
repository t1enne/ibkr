"""SEC EDGAR scope: a fundamentals run submits US listings only.

EDGAR has no CIK for a foreign listing, so submitting ``ENI.MI`` can only answer
"Not found" — which reads like a broken download instead of an inapplicable
source. The venue decides: the stored ``symbol`` row when one exists (the
authority), else the series key's own suffix. A bare key with neither is the US
path every legacy symbol takes.
"""

from __future__ import annotations

import pytest

from src.data.fundamentals import dl as dl_mod
from src.data.fundamentals.dl import is_sec_listed


def test_a_bare_us_key_is_sec_listed():
    assert is_sec_listed("AAPL")
    assert is_sec_listed("BRK.B")  # a dot is not a venue suffix


def test_a_suffixed_key_is_not_sec_listed():
    assert not is_sec_listed("ENI.MI")
    assert not is_sec_listed("AI.PA")


def test_a_stored_us_venue_is_sec_listed_even_when_suffixed():
    # The row is the authority: a series the DB records on a US venue is US.
    assert is_sec_listed("ENI.MI", "SMART")
    assert is_sec_listed("AAPL", "NASDAQ")


def test_a_stored_foreign_venue_is_not_sec_listed():
    assert not is_sec_listed("ENI", "BVME")
    assert not is_sec_listed("ENI.MI", "BVME")


def test_the_partition_keeps_input_order_and_names_the_skipped(monkeypatch):
    """One call splits a mixed universe: submitted first, skipped upper-cased."""
    monkeypatch.setattr(
        dl_mod, "_stored_market", lambda symbol: {"ENI": "BVME"}.get(symbol)
    )

    asked, skipped = dl_mod._sec_listed_symbols(["AAPL", "eni.mi", "ENI", "MSFT"])

    assert asked == ["AAPL", "MSFT"]
    assert skipped == ["ENI.MI", "ENI"]


def test_an_unreadable_symbol_table_is_no_venue(monkeypatch):
    # A fundamentals run must not die on a candle-DB hint it can live without.
    def _boom(_symbol: str) -> str | None:
        raise RuntimeError("no such table: symbol")

    monkeypatch.setattr(dl_mod, "market_for_ticker", _boom)

    assert dl_mod._stored_market("ENI.MI") is None
    assert not is_sec_listed("ENI.MI", dl_mod._stored_market("ENI.MI"))


@pytest.mark.asyncio
async def test_download_submits_only_the_us_listings(monkeypatch):
    """A foreign series never reaches SEC, and the summary says so once."""
    monkeypatch.setattr(dl_mod, "_stored_market", lambda _symbol: None)
    submitted: list[str] = []

    async def _fetch(
        symbols: list[str], cache: object = None, refresh: bool = False
    ) -> dict[str, object]:
        submitted.extend(symbols)
        return {}

    monkeypatch.setattr(dl_mod, "download_fundamentals", _fetch)
    monkeypatch.setattr(dl_mod, "bootstrap", lambda: None)
    lines: list[str] = []
    monkeypatch.setattr(dl_mod.click, "echo", lambda msg, err=False: lines.append(msg))

    recaps = await dl_mod.download(["AAPL", "ENI.MI"])

    assert submitted == ["AAPL"]
    assert [r.symbol for r in recaps] == ["AAPL"]
    assert any("Skipped (not US-listed): 1" in line for line in lines)
