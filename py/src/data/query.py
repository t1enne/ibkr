"""`data query` command — inspect OHLCV candles from the local database."""

from __future__ import annotations

from typing import Optional

import click

from src.data._shared import print_recap, resolve_symbol_list


@click.command(name="query")
@click.argument("symbols", nargs=-1, required=False)
@click.option(
    "--universe",
    "-U",
    help="Universe file name (e.g. 'nsdq', 'sector'). Overrides positional SYMBOLS.",
)
@click.option("--from", "-f", "from_date", help="Start date (YYYY-MM-DD)")
@click.option("--to", "-t", "to_date", help="End date (YYYY-MM-DD)")
@click.option("--bar", default="1h", help="Bar size (1h, 1d, etc.)")
def query_cmd(
    symbols: tuple[str, ...],
    universe: Optional[str],
    from_date: Optional[str],
    to_date: Optional[str],
    bar: str,
):
    """Query OHLCV candles for SYMBOL from the local database.

    Shows a recap with date range, row count, and gaps >48h.
    """
    symbols_list = resolve_symbol_list(symbols, universe)
    print_recap(symbols_list.tickers, from_date or "", to_date, bar)


def register(group: click.Group) -> None:
    """Register this command onto the data group."""
    group.add_command(query_cmd)
