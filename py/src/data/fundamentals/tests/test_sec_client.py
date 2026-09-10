"""SEC EDGAR HTTP tests (respx-mocked — no live network)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from src.data.fundamentals import sec_client
from src.data.fundamentals.sec_client import (
    cik_for_ticker,
    download_fundamentals,
    fetch_companyfacts,
)

_TICKERS = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
}
_FACTS = {"cik": 320193, "facts": {"us-gaap": {}}}


def _mock_tickers() -> respx.Route:
    return respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKERS)
    )


def _mock_facts(cik: int = 320193) -> respx.Route:
    return respx.get(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    ).mock(return_value=httpx.Response(200, json=_FACTS))


def test_cik_for_ticker_resolves_and_pads(tmp_path: Path) -> None:
    with respx.mock:
        _mock_tickers()
        assert cik_for_ticker("AAPL", cache=tmp_path) == "0000320193"
        assert cik_for_ticker("aapl", cache=tmp_path) == "0000320193"


def test_cik_for_ticker_unknown_is_loud(tmp_path: Path) -> None:
    with respx.mock:
        _mock_tickers()
        with pytest.raises(KeyError):
            cik_for_ticker("SPY", cache=tmp_path)


def test_ticker_map_is_fetched_once_then_cached(tmp_path: Path) -> None:
    with respx.mock:
        route = _mock_tickers()
        cik_for_ticker("AAPL", cache=tmp_path)
        cik_for_ticker("MSFT", cache=tmp_path)
        assert route.call_count == 1


def test_companyfacts_uses_cache_on_second_call(tmp_path: Path) -> None:
    with respx.mock:
        _mock_tickers()
        route = _mock_facts()
        first = fetch_companyfacts("AAPL", cache=tmp_path)
        second = fetch_companyfacts("AAPL", cache=tmp_path)
        assert first == _FACTS == second
        assert route.call_count == 1


def test_refresh_bypasses_the_cache(tmp_path: Path) -> None:
    with respx.mock:
        _mock_tickers()
        route = _mock_facts()
        fetch_companyfacts("AAPL", cache=tmp_path)
        fetch_companyfacts("AAPL", cache=tmp_path, refresh=True)
        assert route.call_count == 2


def test_corrupt_cache_falls_back_to_the_network(tmp_path: Path) -> None:
    """A truncated cache file must self-heal, not poison every later run."""
    (tmp_path / "AAPL.json").write_text("{not json")
    with respx.mock:
        _mock_tickers()
        route = _mock_facts()
        assert fetch_companyfacts("AAPL", cache=tmp_path) == _FACTS
        assert route.call_count == 1


def test_sec_requires_a_user_agent_header(tmp_path: Path) -> None:
    with respx.mock:
        _mock_tickers()
        route = _mock_facts()
        fetch_companyfacts("AAPL", cache=tmp_path)
        ua = route.calls[0].request.headers.get("User-Agent")
        assert ua and "ibkr-py" in ua


@pytest.mark.asyncio
async def test_download_includes_only_resolvable_symbols(tmp_path: Path) -> None:
    """A mixed universe downloads the issuers that exist, skips the rest."""
    with respx.mock:
        _mock_tickers()
        _mock_facts(320193)
        _mock_facts(789019)
        respx.get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000789019.json").mock(
            return_value=httpx.Response(500)
        )
        out = await download_fundamentals(
            ["AAPL", "MSFT", "SPY"], batch_delay_s=0.0, cache=tmp_path
        )
    assert sorted(out) == ["AAPL"]  # MSFT errored, SPY has no CIK
    assert out["AAPL"] == _FACTS


@pytest.mark.asyncio
async def test_download_paces_between_symbols(tmp_path: Path, monkeypatch) -> None:
    """SEC asks for a paced caller — the batch must not hammer the endpoint."""
    slept: list[float] = []

    async def _sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(sec_client.asyncio, "sleep", _sleep)
    with respx.mock:
        _mock_tickers()
        _mock_facts(320193)
        await download_fundamentals(["AAPL", "AAPL"], batch_delay_s=0.2, cache=tmp_path)
    # Two symbols -> exactly one inter-request delay (none before the first).
    assert slept == [0.2]


def test_cache_roundtrips_as_plain_json(tmp_path: Path) -> None:
    """The cache stays human-inspectable (the reason it isn't a pickle)."""
    with respx.mock:
        _mock_tickers()
        _mock_facts()
        fetch_companyfacts("AAPL", cache=tmp_path)
    assert json.loads((tmp_path / "AAPL.json").read_text()) == _FACTS
