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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import click
import pandas as pd

from src.data.fundamentals.normalize import sec_payload_to_rows
from src.data.fundamentals.schema import FundamentalRow, bootstrap, insert_fundamentals
from src.data.fundamentals.sec_client import download_fundamentals


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

    A symbol SEC has no CIK for (ETF, non-US registrant) or whose payload errors
    is reported with ``rows=0`` rather than aborting the batch — a mixed
    universe should still download the issuers that exist.

    Only the rows actually written are reported, and symbols that produced no new
    rows are skipped at the DB, so the recap matches what a read would find.

    No history is re-downloaded: ``companyfacts`` is cached per symbol on disk,
    so a warm run makes no HTTP call at all and an unchanged payload writes
    nothing (``Unchanged``). ``--refresh`` is the only way to bypass the cache.
    """
    bootstrap()

    updated: list[str] = []
    failed: list[str] = []
    not_found: list[str] = []
    unchanged: list[str] = []

    payloads = await download_fundamentals(symbols, cache=cache, refresh=refresh)

    total = len(symbols)

    def _rows(advance: Callable[[int, str], None]) -> list[DownloadRecap]:
        """Persist every payload, calling ``advance(i, ticker)`` per symbol."""
        recaps: list[DownloadRecap] = []
        for i, symbol in enumerate(symbols, 1):
            ticker = symbol.upper()
            payload = payloads.get(ticker)
            if payload is None:
                not_found.append(ticker)
                recaps.append(DownloadRecap(ticker, 0, 0, None, None))
                advance(i, ticker)
                continue
            try:
                rows = _window(sec_payload_to_rows(ticker, payload), from_date, to_date)
                written = insert_fundamentals(rows)
                if written:
                    updated.append(ticker)
                elif rows:
                    # Facts already stored — a no-op re-run, not a miss.
                    unchanged.append(ticker)
                else:
                    not_found.append(ticker)
                recaps.append(
                    DownloadRecap(
                        symbol=ticker,
                        rows=written,
                        periods=len({r.period_end for r in rows}),
                        first_filed=min((r.filed for r in rows), default=None),
                        last_filed=max((r.filed for r in rows), default=None),
                    )
                )
            except Exception:
                failed.append(ticker)
                recaps.append(DownloadRecap(ticker, 0, 0, None, None))
            advance(i, ticker)
        return recaps

    with click.progressbar(
        length=total,
        label="fundamentals",
        width=40,
        file=sys.stderr,
    ) as pb:
        recaps = _rows(lambda _done, _ticker: pb.update(1))

    _report(updated, failed, not_found, unchanged)
    return tuple(recaps)


def _report(
    updated: list[str],
    failed: list[str],
    not_found: list[str],
    unchanged: list[str],
) -> None:
    """Print final summary line to stderr.

    ``Unchanged`` and ``Not found`` are different answers: the first holds
    symbols whose facts are already stored (a re-run wrote nothing because
    nothing was new), the second holds symbols SEC has no data for at all.
    """
    parts = []
    if updated:
        parts.append(f"Updated: {len(updated)}")
    if unchanged:
        parts.append(f"Unchanged: {len(unchanged)}")
    if failed:
        parts.append(f"Failed: {len(failed)} ({', '.join(failed)})")
    if not_found:
        parts.append(f"Not found: {len(not_found)} ({', '.join(not_found)})")
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


__all__ = ["download", "DownloadRecap", "report"]
