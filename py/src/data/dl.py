"""`data dl` command — download historical candles and SEC fundamentals.

Owns the download orchestration (previously buried in the `sync` module).
Imports the actual fetching primitives from the ibkr layer, symbol resolution
from `symbols` and the SEC ingest from `fundamentals.dl` — it never imports
from `sync`.

Two independent phases, each owning its own progress bar and its own summary:
:func:`run_fundamentals` (SEC EDGAR) then :func:`run_candles` (IBKR Gateway).
The command is just those two calls in order, so neither phase knows about the
other and either can be run alone. Inside a phase the per-chunk INFO stream is
muted (:func:`set_verbose`) because a bar and timestamped progress lines cannot
share one terminal; warn/error/retry stay loud, and the phase's report table
carries the outcome — failures and remaining gaps alike.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable
from datetime import date, datetime
from typing import Optional, TextIO

import click

from src.bt.table import Col, Table, render as render_table
from src.data._shared import resolve_symbol_list
from src.data.fundamentals.dl import download as download_fundamentals
from src.data.ibkr.candles import candles_batch, gap_range
from src.data.symbols import resolve_symbols
from src.data.types import CandleDict, ISymbol, ProgressFn, SymbolOutcome, SyncResult
from src.shared.runtime_log import info as _log_info
from src.shared.runtime_log import set_verbose as _set_verbose
from src.shared.style import Styler

#: Sink for a fetched chunk of candles (I/O boundary — default writes to the DB).
type ChunkSink = Callable[[list[CandleDict]], None]
#: Called once per conid with its gaps still missing, or ``None`` on failure.
type ItemFn = Callable[[int, Optional[list[tuple[datetime, datetime]]]], None]
#: Runs one phase under a progress bar; returns the lines it wants printed.
type PhaseFn[T] = Callable[[ProgressFn], Awaitable[T]]


# ── Phase runner ───────────────────────────────────────────────


async def run_phase[T](
    label: str,
    total: int,
    body: PhaseFn[T],
) -> T:
    """Run *body* under one progress bar labelled *label*.

    The phase reports progress through the ``ProgressFn`` it is handed; the bar
    only ever moves forward, so a phase that re-reports an earlier count cannot
    rewind it. INFO is muted for the duration (see the module docstring) and
    restored afterwards, even on failure.
    """
    with click.progressbar(
        length=max(1, total), label=label, width=40, file=sys.stderr
    ) as bar:

        def _progress(status: str, current: int, total: int) -> None:
            del status  # the bar reports counts; its label is fixed
            bar.length = max(1, total)
            if current > bar.pos:
                bar.update(current - bar.pos)

        _set_verbose(False)
        try:
            return await body(_progress)
        finally:
            _set_verbose(True)


# ── Candle phase ───────────────────────────────────────────────


async def _get_candles(
    symbols: list[ISymbol],
    from_date: date,
    to_date: Optional[date] = None,
    bar: str = "1h",
    on_chunk: Optional[ChunkSink] = None,
    on_item: Optional[ItemFn] = None,
) -> dict[int, list[tuple[datetime, datetime]] | None]:
    """Fetch missing candle data for resolved symbols.

    Returns ``{conid: gaps still missing or None if the fetch raised}`` — the
    gaps ``candles()`` already computed, so no caller re-scans the DB.
    """
    from_dt = datetime.combine(from_date, datetime.min.time())
    to_dt = datetime.combine(to_date, datetime.max.time()) if to_date else None

    return await candles_batch(
        [symbol.conid for symbol in symbols],
        from_datetime=from_dt,
        to_datetime=to_dt,
        bar=bar,
        on_chunk=on_chunk,
        on_item=on_item,
    )


def _report(result: SyncResult, *, file: TextIO | None = None) -> None:
    """Render the run's final table: totals, then one row per symbol.

    A symbol is a problem only when it failed or still has gaps; clean symbols
    collapse into the header count so a 300-symbol universe stays readable.
    """
    out = file or sys.stderr
    styler = Styler(enabled=out.isatty())
    broken = [s for s in result.symbols if not (s.ok and not s.gaps)]

    complete = sum(1 for s in result.symbols if s.ok and not s.gaps)
    heading = (
        f"resolved {result.resolved}  "
        f"complete {complete}  "
        f"failed {sum(1 for s in result.symbols if not s.ok)}  "
        f"gaps {result.gaps_found}  "
        f"unresolved {len(result.unresolved)}"
    )
    click.echo(heading, err=out is sys.stderr, file=out)

    if not broken:
        click.echo("  no failures, no remaining gaps", err=out is sys.stderr, file=out)
        return

    rows = tuple(
        (
            s.ticker,
            "failed" if not s.ok else ("gaps" if s.gaps else "ok"),
            str(len(s.gaps)) if s.gaps else "-",
            gap_range(s.gaps),
            (s.error or "")[:80],
        )
        for s in broken
    )
    cols = (
        Col("symbol", role="strong"),
        Col("state"),
        Col("gaps", ">"),
        Col("range"),
        Col("detail"),
    )
    lines = render_table(Table(columns=cols, rows=rows), sep="  ", styler=styler)
    click.echo("\n".join(lines), err=out is sys.stderr, file=out)


async def download(
    tickers: list[str],
    from_date: date,
    to_date: Optional[date] = None,
    bar: str = "1h",
    on_progress: Optional[ProgressFn] = None,
    on_chunk: Optional[ChunkSink] = None,
) -> SyncResult:
    """Sync historical candle data for the given tickers.

    Args:
        tickers: List of ticker symbols to sync
        from_date: Start date for historical data (required)
        to_date: End date for historical data (defaults to now)
        bar: Bar size (e.g. "1h", "1d")
        on_progress: Optional progress callback (I/O boundary)
        on_chunk: Optional sink for fetched candles (default: write to the DB)

    Returns:
        SyncResult with per-symbol outcomes — failures, remaining gaps and
        unresolved tickers all land here rather than in the log stream.

    Raises:
        ValueError: If no symbols could be resolved
    """
    if on_progress:
        on_progress("Resolving symbols...", 0, len(tickers))

    symbols = await resolve_symbols(tickers)
    resolved_tickers = {s.ticker for s in symbols}
    unresolved = tuple(
        t for t in (x.upper() for x in tickers) if t not in resolved_tickers
    )

    if on_progress:
        on_progress(
            f"Resolved {len(symbols)}/{len(tickers)} symbols",
            len(symbols),
            len(tickers),
        )

    if not symbols:
        raise ValueError(f"No symbols resolved for tickers: {tickers}")

    # One completed symbol is one unit of the bar: ``on_item`` fires per symbol
    # (success or failure), so the bar advances exactly ``len(symbols)`` times.
    done = 0

    def _item(_conid: int, _gaps: list[tuple[datetime, datetime]] | None) -> None:
        nonlocal done
        done += 1
        if on_progress:
            on_progress("Candles", done, len(symbols))

    fetched = await _get_candles(
        symbols, from_date, to_date, bar=bar, on_chunk=on_chunk, on_item=_item
    )

    outcomes = tuple(
        SymbolOutcome(sym.ticker, sym.conid, error="fetch failed")
        if fetched.get(sym.conid) is None
        else SymbolOutcome(sym.ticker, sym.conid, gaps=tuple(fetched[sym.conid] or ()))
        for sym in symbols
    )

    result = SyncResult(
        resolved=len(symbols),
        fetched=[c for c, gaps in fetched.items() if gaps is not None],
        gaps_found=sum(len(o.gaps) for o in outcomes),
        symbols=outcomes,
        unresolved=unresolved,
    )

    if on_progress:
        on_progress("Sync complete", len(result.fetched), len(symbols))

    return result


async def run_candles(
    tickers: list[str],
    from_date: date,
    to_date: Optional[date] = None,
    bar: str = "1h",
) -> SyncResult:
    """Candle phase: one ``candles`` bar, then the outcome table.

    Owns everything the merged command needs, so the command itself stays a
    call list: bar, quiet INFO, report. Returns the :class:`SyncResult` for a
    caller that wants the numbers (a test, a later phase).
    """

    async def _body(progress: ProgressFn) -> SyncResult:
        result = await download(
            tickers, from_date=from_date, to_date=to_date, bar=bar, on_progress=progress
        )
        _report(result)
        return result

    return await run_phase("candles", len(tickers), _body)


async def run_fundamentals(
    tickers: list[str],
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    refresh: bool = False,
    cache: Optional[str] = None,
) -> int:
    """Fundamentals phase: one ``fundamentals`` bar, then the written-row count.

    Reads as the candle phase's twin, but the bar is SEC's own — its download
    already reports progress per symbol, so this adds no second one. Returns the
    total rows written so a caller need not parse the printed summary.
    """
    recaps = await download_fundamentals(
        tickers, from_date=from_date, to_date=to_date, refresh=refresh, cache=cache
    )
    return sum(r.rows for r in recaps)


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
    never silently imposed on it).
    """
    symbols_list = resolve_symbol_list(symbols, universe)

    async def _run() -> None:
        await run_fundamentals(
            symbols_list,
            from_date=_opt_date(fundamentals_from),
            to_date=_opt_date(fundamentals_to),
            refresh=refresh_fundamentals,
            cache=fundamentals_cache,
        )
        result = await run_candles(
            symbols_list,
            from_date=date.fromisoformat(from_date),
            to_date=_opt_date(to_date),
            bar=bar,
        )
        if result.unresolved:
            _log_info(f"unresolved: {', '.join(result.unresolved)}")

    asyncio.run(_run())


def _opt_date(value: Optional[str]) -> Optional[date]:
    """Parse an optional ``YYYY-MM-DD`` flag value."""
    return date.fromisoformat(value) if value else None


def register(group: click.Group) -> None:
    """Register this command onto the data group."""
    group.add_command(dl_cmd)
