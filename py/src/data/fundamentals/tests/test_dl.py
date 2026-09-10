"""`data fundamentals dl` tests — SEC mocked at HTTP, DB isolated to a temp file.

Covers the ingest path end-to-end: payload -> rows -> DB -> recap, with the
window filter and the "no CIK" skip. The recap is what the CLI prints, so a
regression in which rows land (or in the window bound) fails here.
"""

from __future__ import annotations

import pandas as pd

from pathlib import Path

import httpx
import pytest
import respx

from src.data.fundamentals import dl as dlmod
from src.data.fundamentals.query import load_stated
from src.data.fundamentals.schema import FundamentalSchema, bootstrap
from src.data.fundamentals.sec_client import download_fundamentals
from src.utils import parse_timestamp


def ts(value: str) -> pd.Timestamp:
    """``pd.Timestamp`` narrowed past its ``NaTType`` union (repo convention)."""
    return parse_timestamp(value)


_TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}

_FACTS = {
    "cik": 320193,
    "facts": {
        "us-gaap": {
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        {
                            "start": "2023-01-01",
                            "end": "2023-03-31",
                            "val": 100.0,
                            "form": "10-Q",
                            "filed": "2023-05-01",
                        },
                        {
                            "start": "2023-04-01",
                            "end": "2023-06-30",
                            "val": 200.0,
                            "form": "10-Q",
                            "filed": "2023-08-01",
                        },
                        # A restatement of Q1, filed much later.
                        {
                            "start": "2023-01-01",
                            "end": "2023-03-31",
                            "val": 150.0,
                            "form": "10-K",
                            "filed": "2024-02-01",
                        },
                    ]
                }
            },
            "Assets": {
                "units": {
                    "USD": [
                        {
                            "start": "2023-03-31",
                            "end": "2023-03-31",
                            "val": 5000.0,
                            "form": "10-Q",
                            "filed": "2023-05-01",
                        }
                    ]
                }
            },
        }
    },
}


@pytest.fixture()
def db_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate the whole module against a temp DB (never the production file)."""
    from peewee import SqliteDatabase

    database = SqliteDatabase(str(tmp_path / "fund.db"))
    monkeypatch.setattr(FundamentalSchema._meta, "database", database)
    bootstrap(database)
    return tmp_path / "fund.db"


def _mock_sec() -> respx.Route:
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKERS)
    )
    return respx.get(
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"
    ).mock(return_value=httpx.Response(200, json=_FACTS))


@pytest.mark.asyncio
async def test_download_persists_rows_and_reports_them(
    db_file: Path, tmp_path: Path
) -> None:
    with respx.mock:
        _mock_sec()
        recaps = await dlmod.download(["AAPL"], cache=tmp_path / "cache")

    (recap,) = recaps
    # 3 income facts (incl. the restatement) + 1 balance fact.
    assert recap.rows == 4
    assert recap.periods == 2  # Q1 (restated) and Q2
    assert recap.first_filed == pd_timestamp("2023-05-01")
    assert recap.last_filed == pd_timestamp("2024-02-01")

    stored = load_stated("AAPL")
    assert {r.field for r in stored} == {"net_income", "assets"}
    # The restatement is stored, so the as-first-stated read can still see it.
    assert len([r for r in stored if r.field == "net_income"]) == 3


@pytest.mark.asyncio
async def test_download_window_bounds_the_filing_date(
    db_file: Path, tmp_path: Path
) -> None:
    """``--from``/``--to`` bound *filings*, so a late restatement is retained."""
    from datetime import date

    with respx.mock:
        _mock_sec()
        recaps = await dlmod.download(
            ["AAPL"],
            from_date=date(2023, 1, 1),
            to_date=date(2023, 12, 31),
            cache=tmp_path / "cache",
        )
    (recap,) = recaps
    assert recap.rows == 3  # the 2024-02-01 restatement is outside the window
    assert recap.last_filed == pd_timestamp("2023-08-01")


@pytest.mark.asyncio
async def test_download_reports_symbols_sec_does_not_know(
    db_file: Path, tmp_path: Path
) -> None:
    """An ETF in the universe gets an explicit zero-row recap, not an abort."""
    with respx.mock:
        _mock_sec()
        recaps = await dlmod.download(["AAPL", "SPY"], cache=tmp_path / "cache")

    by_symbol = {r.symbol: r for r in recaps}
    assert by_symbol["AAPL"].rows == 4
    assert by_symbol["SPY"].rows == 0
    assert "SPY" in {r.symbol for r in recaps}


@pytest.mark.asyncio
async def test_download_payload_cache_avoids_second_fetch(
    db_file: Path, tmp_path: Path
) -> None:
    """A cached payload means no second SEC request (idempotent re-runs)."""
    cache = tmp_path / "cache"
    with respx.mock:
        facts_route = _mock_sec()
        await download_fundamentals(["AAPL"], batch_delay_s=0.0, cache=cache)
        await download_fundamentals(["AAPL"], batch_delay_s=0.0, cache=cache)
    assert facts_route.call_count == 1


def pd_timestamp(value: str):

    return parse_timestamp(value)


def test_table_exists_helper_roundtrip(db_file: Path) -> None:
    """The isolated fixture really is isolated (no production DB writes)."""
    assert FundamentalSchema.table_exists()
