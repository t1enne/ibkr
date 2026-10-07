"""`data dl` command — download historical candles and SEC fundamentals.

Owns the download orchestration (previously buried in the `sync` module).
Imports the actual fetching primitives from the ibkr layer, symbol resolution
from `symbols` and the SEC ingest from `fundamentals.dl` — it never imports
from `sync`.

The two sources are one operator action but independent transports: candles
come from the IBKR Gateway, fundamentals from SEC EDGAR (cached on disk). A
symbol list is resolved once and feeds both passes.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Optional

import click

from src.data._shared import display_preview, resolve_symbol_list
from src.data.fundamentals.dl import report as report_fundamentals
from src.data.fundamentals.dl import download as download_fundamentals
from src.data.ibkr.candles import candles_batch
from src.data.symbols import resolve_symbols
from src.data.types import ISymbol, ProgressFn, SyncResult


# ── Download orchestration ─────────────────────────────────────


async def _get_candles(
    symbols: list[ISymbol],
    from_date: date,
    to_date: Optional[date] = None,
    bar: str = "1h",
) -> list[int]:
    """Fetch missing candle data for resolved symbols.

    Returns list of conids that were successfully fetched.
    Errors propagate — no silent swallowing.
    """
    from_dt = datetime.combine(from_date, datetime.min.time())
    to_dt = datetime.combine(to_date, datetime.max.time()) if to_date else None

    return await candles_batch(
        [symbol.conid for symbol in symbols],
        from_datetime=from_dt,
        to_datetime=to_dt,
        bar=bar,
    )


async def download(
    tickers: list[str],
    from_date: date,
    to_date: Optional[date] = None,
    bar: str = "1h",
    on_progress: Optional[ProgressFn] = None,
) -> SyncResult:
    """Sync historical candle data for the given tickers.

    Args:
        tickers: List of ticker symbols to sync
        from_date: Start date for historical data (required)
        to_date: End date for historical data (defaults to now)
        bar: Bar size (e.g. "1h", "1d")
        on_progress: Optional progress callback (I/O boundary)

    Returns:
        SyncResult with resolved count, fetched conids, and gap count.

    Raises:
        ValueError: If no symbols could be resolved
    """
    if on_progress:
        on_progress("Resolving symbols...", 0, len(tickers))

    symbols = await resolve_symbols(tickers)

    if on_progress:
        on_progress(
            f"Resolved {len(symbols)}/{len(tickers)} symbols",
            len(symbols),
            len(tickers),
        )

    if not symbols:
        raise ValueError(f"No symbols resolved for tickers: {tickers}")

    if on_progress:
        on_progress("Fetching candles...", 0, len(symbols))

    fetched = await _get_candles(symbols, from_date, to_date, bar)

    if on_progress:
        on_progress("Sync complete", len(symbols), len(symbols))

    return SyncResult(
        resolved=len(symbols),
        fetched=fetched,
        gaps_found=len(fetched),
    )


# ── Command ────────────────────────────────────────────────────


@click.command(name="dl")
@click.argument("symbols", nargs=-1, required=False)
@click.option(
    "--universe",
    "-U",
    help="Universe file PATH (e.g. 'universes/nsdq.json'). Overrides positional SYMBOLS.",
)
@click.option(
    "--from", "-f", "from_date", required=True, help="Start date (YYYY-MM-DD)"
)
@click.option("--to", "-t", "to_date", help="End date (YYYY-MM-DD)")
@click.option("--bar", default="1h", help="Bar size (1h, 1d, etc.)")
@click.option(
    "--fundamentals-from",
    "fundamentals_from",
    default=None,
    help="Only SEC filings on/after this date (YYYY-MM-DD). Default: all filings.",
)
@click.option(
    "--fundamentals-to",
    "fundamentals_to",
    default=None,
    help="Only SEC filings on/before this date (YYYY-MM-DD). Default: all filings.",
)
@click.option(
    "--refresh-fundamentals",
    is_flag=True,
    help="Bypass the SEC payload cache and re-fetch from EDGAR.",
)
@click.option(
    "--fundamentals-cache",
    "fundamentals_cache",
    default=None,
    help="SEC payload cache dir (default: ../data/fundamentals_cache).",
)
def dl_cmd(
    symbols: tuple[str, ...],
    universe: Optional[str],
    from_date: str,
    to_date: Optional[str],
    bar: str,
    fundamentals_from: Optional[str],
    fundamentals_to: Optional[str],
    refresh_fundamentals: bool,
    fundamentals_cache: Optional[str],
):
    """Download IBKR candles + SEC fundamentals for SYMBOLS or --universe.

    Candles come from the Gateway, bounded by `--from`/`--to`; fundamentals come
    from EDGAR as sparse fiscal rows, unbounded by default (a strategy's
    as-first-stated history needs the earliest filings, so the candle window is
    never silently imposed on it). Shows remaining candle gaps after download
    using same format as preview.
    """
    symbols_list = resolve_symbol_list(symbols, universe)
    f_date = date.fromisoformat(from_date)
    t_date = date.fromisoformat(to_date) if to_date else None

    def _echo(line: str) -> None:
        click.echo(line, err=True)

    from src.data.preview import preview as preview_gaps

    async def _run():
        # The SEC pass runs first and independently of the Gateway: it needs no
        # broker session, so a Gateway outage must not cost us the filings, and
        # a Candles failure still leaves the recap on screen.
        recaps = await download_fundamentals(
            symbols_list,
            from_date=(
                date.fromisoformat(fundamentals_from) if fundamentals_from else None
            ),
            to_date=(date.fromisoformat(fundamentals_to) if fundamentals_to else None),
            refresh=refresh_fundamentals,
            cache=fundamentals_cache,
        )
        _echo("fundamentals:")
        report_fundamentals(recaps, _echo)
        await download(symbols_list, from_date=f_date, to_date=t_date, bar=bar)
        return await preview_gaps(symbols_list, from_date=f_date, to_date=t_date)

    remaining = asyncio.run(_run())
    display_preview(remaining, from_date, to_date)


def register(group: click.Group) -> None:
    """Register this command onto the data group."""
    group.add_command(dl_cmd)
