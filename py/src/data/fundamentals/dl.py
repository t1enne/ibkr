"""SEC EDGAR fundamentals ingest — fetch payloads, persist rows, report counts.

Driven by the merged `data dl` command (positional SYMBOLS or --universe); the
`--from`/`--to` bounds here apply to the *filing* date, and `--refresh`
bypasses the on-disk payload cache. ``download`` returns per-symbol counts and
``_recap_line`` renders one line each, so "did anything arrive?" is answerable
from the command output rather than a second query.

Fetching and persistence are separate concerns here: ``download`` returns the
counts (pure-ish, HTTP + DB edges only), the caller renders them.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from types import MappingProxyType
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import click
import pandas as pd

from src.data.fundamentals.normalize import sec_payload_to_rows
from src.data.fundamentals.schema import FundamentalRow, bootstrap, insert_fundamentals
from src.data.fundamentals.sec_client import download_fundamentals
from src.data.symbols import market_for_ticker
from src.data.venue import is_bare_venue, venue_of


@dataclass(frozen=True)
class DownloadRecap:
    """Per-symbol outcome of a fundamentals download.

    ``rows`` is the number of rows actually written (0 on an idempotent re-run);
    ``periods`` describes the payload, so the two together distinguish "nothing
    new" from "nothing at all".
    """

    symbol: str
    rows: int
    periods: int
    first_filed: pd.Timestamp | None
    last_filed: pd.Timestamp | None


def is_sec_listed(symbol: str, market: str | None = None) -> bool:
    """True when SEC EDGAR can have fundamentals for *symbol* — a US listing.

    SEC EDGAR registers US listings only, so a foreign series (``ENI.MI``) has no
    CIK at all: submitting it can only ever answer "Not found", which reads like a
    broken download rather than an inapplicable source. Skipping says so once,
    instead of naming every foreign symbol in the run.

    The venue comes from the ``symbol`` row when one is stored (the authority on
    where a series lists) and from the key's own suffix otherwise; a bare key with
    no venue is the US path, which every legacy symbol is.
    """
    if market is not None:
        return is_bare_venue(market)
    return venue_of(symbol) is None


def _stored_market(symbol: str) -> str | None:
    """The stored venue of *symbol*, or ``None`` when the DB cannot answer.

    A missing ``symbol`` table is no venue, not a failed run: the suffix fallback
    in :func:`is_sec_listed` still classifies the symbol, and a fundamentals
    download must not die on a candle-DB read it only uses as a hint.
    """
    try:
        return market_for_ticker(symbol)
    except Exception:
        return None


def _sec_listed_symbols(symbols: list[str]) -> tuple[list[str], list[str]]:
    """Partition *symbols* into (what SEC may know, what it cannot) — one order.

    Returned in input order so the progress bar, the recap and the skip summary
    all list a universe the way the operator wrote it.
    """
    asked: list[str] = []
    foreign: list[str] = []
    for symbol in symbols:
        if is_sec_listed(symbol, _stored_market(symbol)):
            asked.append(symbol)
        else:
            foreign.append(symbol.upper())
    return asked, foreign


def _window(
    rows: list[FundamentalRow],
    from_date: date | None,
    to_date: date | None,
) -> list[FundamentalRow]:
    """Filter rows to filings inside ``[from_date, to_date]`` (inclusive).

    Bounds the *filed* date, not the fiscal period: a download window is about
    which filings to pull, and a 10-K filed in 2024 restates 2022's period —
    excluding it by period date would drop exactly the PIT facts we keep.
    """
    if from_date is None and to_date is None:
        return rows
    low = pd.Timestamp(from_date) if from_date else None
    high = pd.Timestamp(to_date) if to_date else None
    return [
        r
        for r in rows
        if (low is None or r.filed >= low) and (high is None or r.filed <= high)
    ]


async def download(
    symbols: list[str],
    from_date: date | None = None,
    to_date: date | None = None,
    refresh: bool = False,
    cache: str | Path | None = None,
) -> tuple[DownloadRecap, ...]:
    """Fetch and persist fundamentals for ``symbols``; returns per-symbol recap.

    Shows its own ``fundamentals`` progress bar on stderr. The merged ``data dl``
    command runs ``dl.run_fundamentals`` instead, which owns a bar for the whole
    run — this entry point is the standalone form and keeps its own.

    A symbol SEC has no CIK for (ETF) or whose payload errors is reported with
    ``rows=0`` rather than aborting the batch — a mixed universe should still
    download the issuers that exist. A NON-US series is never submitted: SEC
    cannot have a CIK for it, so it is skipped by venue (see
    :func:`is_sec_listed`) and named once in the summary instead of appearing as
    a phantom "Not found".

    Only the rows actually written are reported, and symbols that produced no new
    rows are skipped at the DB, so the recap matches what a read would find.

    No history is re-downloaded: ``companyfacts`` is cached per symbol on disk,
    so a warm run makes no HTTP call at all and an unchanged payload writes
    nothing (``Unchanged``). ``--refresh`` is the only way to bypass the cache.
    """
    bootstrap()

    asked, foreign = _sec_listed_symbols(symbols)
    payloads = await download_fundamentals(asked, cache=cache, refresh=refresh)
    tallies: dict[str, list[str]] = {status: [] for status in _STATUSES}

    def _rows(advance: Callable[[int, str], None]) -> list[DownloadRecap]:
        """Persist every payload, calling ``advance(i, ticker)`` per symbol."""
        recaps: list[DownloadRecap] = []
        for i, symbol in enumerate(asked, 1):
            recap, status = _persist(
                symbol.upper(), payloads.get(symbol.upper()), from_date, to_date
            )
            tallies[status].append(recap.symbol)
            recaps.append(recap)
            advance(i, recap.symbol)
        return recaps

    with click.progressbar(
        length=len(asked), label="fundamentals", width=40, file=sys.stderr
    ) as pb:
        recaps = _rows(lambda _done, _ticker: pb.update(1))

    _report(tallies, foreign)
    return tuple(recaps)


#: Every outcome a symbol's persist can have, in report order.
_STATUSES: tuple[str, ...] = ("updated", "unchanged", "not_found", "failed")

#: The summary labels, so the tally keys and the printed words cannot drift.
_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "updated": "Updated",
        "unchanged": "Unchanged",
        "not_found": "Not found",
        "failed": "Failed",
    }
)


def _persist(
    ticker: str,
    payload: dict[str, object] | None,
    from_date: date | None,
    to_date: date | None,
) -> tuple[DownloadRecap, str]:
    """Persist one symbol's payload; return its recap and one of :data:`_STATUSES`.

    ``not_found`` and ``unchanged`` are different answers: the first means SEC
    answered with nothing usable, the second that the facts are already stored (a
    no-op re-run). A payload that raises is ``failed`` — one unreadable symbol
    never aborts the batch.
    """
    if payload is None:
        return DownloadRecap(ticker, 0, 0, None, None), "not_found"
    try:
        rows = _window(sec_payload_to_rows(ticker, payload), from_date, to_date)
        written = insert_fundamentals(rows)
        status = "updated" if written else ("unchanged" if rows else "not_found")
        return (
            DownloadRecap(
                symbol=ticker,
                rows=written,
                periods=len({r.period_end for r in rows}),
                first_filed=min((r.filed for r in rows), default=None),
                last_filed=max((r.filed for r in rows), default=None),
            ),
            status,
        )
    except Exception:
        return DownloadRecap(ticker, 0, 0, None, None), "failed"


def _report(tallies: Mapping[str, list[str]], foreign: list[str]) -> None:
    """Print final summary line to stderr.

    ``Unchanged`` and ``Not found`` are different answers: the first holds
    symbols whose facts are already stored (a re-run wrote nothing because
    nothing was new), the second holds symbols SEC has no data for at all.
    ``Skipped`` holds the non-US series never submitted — SEC registers US
    listings only, so their absence is the source's scope, not a miss.
    """
    parts: list[str] = []
    if foreign:
        parts.append(f"Skipped (not US-listed): {len(foreign)}")
    for status in _STATUSES:
        names = tallies[status]
        if not names:
            continue
        named = (
            f": {len(names)}"
            if status == "updated"
            else f": {len(names)} ({', '.join(names)})"
        )
        parts.append(f"{_LABELS[status]}{named}")
    click.echo("  " + ", ".join(parts), err=True)


def _recap_line(recap: DownloadRecap) -> str:
    """One recap line: what was written, and over which fiscal periods.

    ``periods`` counts the periods seen in the payload, so a no-op re-run reads
    "0 rows ... up to date" (the filings are stored, nothing was new) rather
    than the misleading "no SEC fundamentals", which is reserved for symbols we
    genuinely have nothing for.
    """
    if recap.periods == 0:
        return f"{recap.symbol}: no SEC fundamentals"
    span = (
        f"{recap.first_filed.strftime('%Y-%m-%d')} -> "
        f"{recap.last_filed.strftime('%Y-%m-%d')}"
        if recap.first_filed is not None and recap.last_filed is not None
        else "n/a"
    )
    written = f"{recap.rows} rows" if recap.rows else "up to date"
    return f"{recap.symbol}: {written}  {recap.periods} fiscal periods  filed {span}"


def report(
    recaps: tuple[DownloadRecap, ...],
    echo: Callable[[str], None],
) -> None:
    """Write the batch total plus one recap line per symbol via ``echo``.

    ``echo`` is injected so the merged ``data dl`` command owns the destination
    (stderr) and this module never imports ``click``.
    """
    total = sum(r.rows for r in recaps)
    echo(f"{len(recaps)} symbols, {total} rows written to the local DB")
    for recap in recaps:
        echo(_recap_line(recap))


__all__ = ["download", "is_sec_listed", "DownloadRecap", "report"]
