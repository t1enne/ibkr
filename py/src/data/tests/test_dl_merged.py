"""`data dl` CLI — one command fetches candles *and* SEC fundamentals.

The merge is the contract under test: both passes always run for one resolved
symbol list, and the fundamentals-only flags reach the SEC ingest unchanged.
Candle download, preview and the SEC fetch are all monkeypatched, so this
exercises the wiring and the flag plumbing, never the network.
"""

from __future__ import annotations

import sys

from dataclasses import dataclass, field
from datetime import date

import pytest

from click.testing import CliRunner

from src.data import dl as dlmod
from src.data.fundamentals.dl import DownloadRecap
from src.data.types import PreviewResult


@dataclass
class Recorder:
    """What the CLI asked each transport for (one entry per pass)."""

    candles: list[tuple[list[str], date, date | None, str]] = field(
        default_factory=list
    )
    fundamentals: list[tuple[list[str], date | None, date | None, bool, str | None]] = (
        field(default_factory=list)
    )


@pytest.fixture()
def wired(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """Replace the three I/O edges of `data dl` with recorders."""
    calls = Recorder()

    async def fake_candles(
        tickers: list[str],
        from_date: date,
        to_date: date | None = None,
        bar: str = "1h",
        on_progress=None,
    ) -> None:
        calls.candles.append((tickers, from_date, to_date, bar))

    async def fake_preview(
        tickers: list[str], from_date: date, to_date: date | None = None
    ) -> PreviewResult:
        return PreviewResult(resolved=len(tickers), total_gaps=0, plans=[])

    async def fake_fundamentals(
        symbols: list[str],
        from_date: date | None = None,
        to_date: date | None = None,
        refresh: bool = False,
        cache=None,
    ) -> tuple[DownloadRecap, ...]:
        calls.fundamentals.append((symbols, from_date, to_date, refresh, cache))
        return tuple(DownloadRecap(s.upper(), 0, 0, None, None) for s in symbols)

    monkeypatch.setattr(dlmod, "download", fake_candles)
    monkeypatch.setattr(dlmod, "download_fundamentals", fake_fundamentals)
    monkeypatch.setattr(dlmod, "display_preview", lambda *a, **k: None)
    monkeypatch.setattr(sys.modules["src.data.preview"], "preview", fake_preview)
    return calls


def test_dl_fetches_fundamentals_alongside_candles(wired: Recorder) -> None:
    from src.data.cli import data_group

    result = CliRunner().invoke(data_group, ["dl", "AAPL", "--from", "2019-01-01"])
    assert result.exit_code == 0, result.output
    (tickers, _from, _to, bar) = wired.candles[0]
    assert tickers == ["AAPL"]
    assert bar == "1h"
    (fundamentals,) = wired.fundamentals
    symbols, f_from, f_to, refresh, _cache = fundamentals
    # The candle window is *not* imposed on the filings: as-first-stated reads
    # need the earliest filings, so the default window is unbounded.
    assert symbols == ["AAPL"]
    assert (f_from, f_to) == (None, None)
    assert refresh is False


def test_dl_passes_filing_window_and_refresh(wired: Recorder) -> None:
    from src.data.cli import data_group

    result = CliRunner().invoke(
        data_group,
        [
            "dl",
            "AAPL",
            "--from",
            "2019-01-01",
            "--fundamentals-from",
            "2023-01-01",
            "--fundamentals-to",
            "2023-12-31",
            "--refresh-fundamentals",
            "--fundamentals-cache",
            "/tmp/fund-cache",
        ],
    )
    assert result.exit_code == 0, result.output
    (fundamentals,) = wired.fundamentals
    _symbols, f_from, f_to, refresh, cache = fundamentals
    assert f_from is not None and f_from.isoformat() == "2023-01-01"
    assert f_to is not None and f_to.isoformat() == "2023-12-31"
    assert refresh is True
    assert cache == "/tmp/fund-cache"
