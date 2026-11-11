"""`data preview` command — dry-run gap analysis without downloading.

Owns the preview orchestration (previously buried in the `sync` module).
Imports gap primitives from the ibkr layer and symbol resolution from
`symbols` — it never imports from `sync`.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Optional

import click

from src.data._shared import display_preview, resolve_symbol_list
from src.data.ibkr.candles import (
    _merge_and_sort_gaps,
    calculate_gaps,
    find_internal_gaps,
    get_existing_range,
)
from src.data.symbols import resolve_symbols
from src.data.types import ISymbol, FetchPlan, PreviewResult


# ── Preview orchestration ──────────────────────────────────────


async def preview(
    tickers: list[str],
    from_date: date,
    to_date: Optional[date] = None,
    exchange: Optional[str] = None,
) -> PreviewResult:
    """Dry-run: show what gaps exist without making any API fetch calls.

    Resolves symbols, checks existing DB data, calculates gap plans.
    No candles are fetched — this is a pure preview.

    Args:
        tickers: List of ticker symbols to preview
        from_date: Start date for gap analysis
        to_date: End date for gap analysis (defaults to now)
        exchange: IBKR venue code the tickers list on (``None`` = US path)

    Returns:
        PreviewResult with per-symbol FetchPlans and total gap count.
    """
    symbols = await resolve_symbols(tickers, exchange)

    to_dt = (
        datetime.combine(to_date, datetime.max.time()) if to_date else datetime.now()
    )
    from_dt = datetime.combine(from_date, datetime.min.time())

    async def _plan_for(symbol: ISymbol) -> FetchPlan:
        oldest, newest = await get_existing_range(symbol.ticker)
        edge_gaps = calculate_gaps(from_dt, to_dt, oldest, newest, symbol.market)
        internal_gaps = await asyncio.to_thread(
            find_internal_gaps, symbol.ticker, from_dt, to_dt, 48.0, symbol.market
        )
        all_gaps = _merge_and_sort_gaps(edge_gaps + internal_gaps)
        return FetchPlan(
            ticker=symbol.ticker,
            conid=symbol.conid,
            gaps=all_gaps,
        )

    plans = await asyncio.gather(*[_plan_for(s) for s in symbols])

    total_gaps = sum(len(p.gaps) for p in plans)
    return PreviewResult(
        resolved=len(symbols),
        total_gaps=total_gaps,
        plans=plans,
    )


# ── Command ────────────────────────────────────────────────────


@click.command(name="preview")
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
def preview_cmd(
    symbols: tuple[str, ...],
    universe: Optional[str],
    from_date: str,
    to_date: Optional[str],
):
    """Preview what gaps exist without downloading.

    Shows both fetch gaps (missing data that would be downloaded)
    and gaps >48h in existing data.
    """
    symbols_list = resolve_symbol_list(symbols, universe)
    f_date = date.fromisoformat(from_date)
    t_date = date.fromisoformat(to_date) if to_date else None

    async def _run() -> PreviewResult:
        return await preview(
            list(symbols_list.tickers),
            from_date=f_date,
            to_date=t_date,
            exchange=symbols_list.exchange,
        )

    result = asyncio.run(_run())
    display_preview(result, from_date, to_date)


def register(group: click.Group) -> None:
    """Register this command onto the data group."""
    group.add_command(preview_cmd)
