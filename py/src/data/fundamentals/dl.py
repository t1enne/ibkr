"""`data fundamentals dl` — download SEC EDGAR fundamentals for symbols.

Mirrors the `data dl` UX (positional SYMBOLS or --universe, `--from`/`--to`
bounds the *filing* window, `--refresh` bypasses the on-disk payload cache) and
prints a per-symbol recap like `data query` does for candles: how many periods
landed and over what filing span, so "did anything arrive?" is answerable from
the command output rather than a second query.

Fetching and persistence are separate concerns here: ``download`` returns the
counts (pure-ish, HTTP + DB edges only), the command renders them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from src.data._shared import resolve_symbol_list
from src.data.fundamentals.normalize import sec_payload_to_rows
from src.data.fundamentals.schema import FundamentalRow, bootstrap, insert_fundamentals
from src.data.fundamentals.sec_client import download_fundamentals


@dataclass(frozen=True)
class DownloadRecap:
    """Per-symbol outcome of a fundamentals download."""

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

    A symbol SEC has no CIK for (ETF, non-US registrant) or whose payload errors
    is reported with ``rows=0`` rather than aborting the batch — a mixed
    universe should still download the issuers that exist.

    Only the rows actually written are reported, and symbols that produced no new
    rows are skipped at the DB, so the recap matches what a read would find.
    """
    bootstrap()
    payloads = await download_fundamentals(symbols, cache=cache, refresh=refresh)
    recaps: list[DownloadRecap] = []
    for symbol in symbols:
        ticker = symbol.upper()
        payload = payloads.get(ticker)
        if payload is None:
            recaps.append(DownloadRecap(ticker, 0, 0, None, None))
            continue
        rows = _window(sec_payload_to_rows(ticker, payload), from_date, to_date)
        insert_fundamentals(rows)
        recaps.append(
            DownloadRecap(
                symbol=ticker,
                rows=len(rows),
                periods=len({r.period_end for r in rows}),
                first_filed=min((r.filed for r in rows), default=None),
                last_filed=max((r.filed for r in rows), default=None),
            )
        )
    return tuple(recaps)


def _recap_line(recap: DownloadRecap) -> str:
    if recap.rows == 0:
        return f"{recap.symbol}: no SEC fundamentals"
    span = (
        f"{recap.first_filed.strftime('%Y-%m-%d')} -> "
        f"{recap.last_filed.strftime('%Y-%m-%d')}"
        if recap.first_filed is not None and recap.last_filed is not None
        else "n/a"
    )
    return (
        f"{recap.symbol}: {recap.rows} rows  {recap.periods} fiscal periods"
        f"  filed {span}"
    )


@click.command(name="dl")
@click.argument("symbols", nargs=-1, required=False)
@click.option(
    "--universe",
    "-U",
    help="Universe file PATH (e.g. 'universes/nsdq.json'). Overrides positional SYMBOLS.",
)
@click.option(
    "--from",
    "-f",
    "from_date",
    help="Only filings on/after this date (YYYY-MM-DD)",
)
@click.option(
    "--to", "-t", "to_date", help="Only filings on/before this date (YYYY-MM-DD)"
)
@click.option(
    "--refresh",
    is_flag=True,
    help="Bypass the on-disk payload cache and re-fetch from SEC.",
)
@click.option(
    "--cache",
    "cache_path",
    default=None,
    help="Payload cache directory (default: ../data/fundamentals_cache).",
)
def dl_cmd(
    symbols: tuple[str, ...],
    universe: Optional[str],
    from_date: Optional[str],
    to_date: Optional[str],
    refresh: bool,
    cache_path: Optional[str],
):
    """Download SEC EDGAR fundamentals for SYMBOLS or --universe.

    Stores sparse fiscal rows in the local DB; re-run after new filings appear.
    """
    import asyncio

    symbols_list = resolve_symbol_list(symbols, universe)
    f_date = date.fromisoformat(from_date) if from_date else None
    t_date = date.fromisoformat(to_date) if to_date else None

    recaps = asyncio.run(
        download(
            symbols_list,
            from_date=f_date,
            to_date=t_date,
            refresh=refresh,
            cache=cache_path,
        )
    )

    total = sum(r.rows for r in recaps)
    click.echo(
        f"{len(recaps)} symbols, {total} rows written to the local DB\n", err=True
    )
    for recap in recaps:
        click.echo(_recap_line(recap), err=True)


def register(group: click.Group) -> None:
    """Register this command onto the data group."""
    group.add_command(dl_cmd)


__all__ = ["download", "DownloadRecap", "dl_cmd", "register"]
