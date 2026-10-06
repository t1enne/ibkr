"""Shared helpers for data CLI commands.

Pure display/gap helpers shared across the `data` command modules
(query, dl, preview). No side effects, no ORM access.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from typing import Optional

import click
import json
import pandas as pd

from src.timestamps import to_optional_ts
from src.data.symbols import load_universe_config
from src.data.db import query_candles
from src.data.types import PreviewResult
from src.data.xcal import is_non_trading_day


def _all_non_trading(gap_start: datetime, gap_end: datetime) -> bool:
    """Check if every date in (gap_start, gap_end) is a non-trading day."""
    d = gap_start.date() + timedelta(days=1)
    end = gap_end.date()
    while d < end:
        dt = datetime.combine(d, datetime.min.time())
        if not is_non_trading_day(dt):
            return False
        d += timedelta(days=1)
    return True


def find_gaps_48h(df: pd.DataFrame) -> list[tuple[datetime, datetime]]:
    """Find gaps >48h between consecutive candles in a DataFrame.

    Filters out gaps that consist entirely of non-trading days
    (weekends + NYSE holidays).

    Args:
        df: DataFrame with DatetimeIndex, sorted chronologically.

    Returns:
        List of (gap_start, gap_end) tuples for real data gaps.
    """
    if df.empty or len(df) < 2:
        return []

    gaps: list[tuple[datetime, datetime]] = []
    idx = df.index
    for i in range(len(idx) - 1):
        delta = idx[i + 1] - idx[i]
        if delta > timedelta(hours=48):
            gap_start = idx[i].to_pydatetime()
            gap_end = idx[i + 1].to_pydatetime()
            if not _all_non_trading(gap_start, gap_end):
                gaps.append((gap_start, gap_end))
    return gaps


def recap(df: pd.DataFrame, label: str) -> str:
    """Build a recap string for a symbol's data.

    Shows: start, end, total rows, and any gaps >48h.
    """
    if df.empty:
        return f"{label}: no data"

    gaps = find_gaps_48h(df)
    gap_lines = ""
    if gaps:
        pieces = "\n  ".join(
            f"{g[0].strftime('%Y-%m-%d %H:%M')} → {g[1].strftime('%Y-%m-%d %H:%M')}"
            for g in gaps
        )
        gap_lines = f"\n  gaps >48h ({len(gaps)}):\n  {pieces}"

    return (
        f"{label}: {len(df)} rows"
        f"  {df.index[0].strftime('%Y-%m-%d %H:%M')}"
        f" → {df.index[-1].strftime('%Y-%m-%d %H:%M')}"
        f"{gap_lines}"
    )


def print_recap(
    symbols: tuple[str, ...],
    from_date: str,
    to_date: Optional[str],
    bar: str = "1h",
) -> None:
    """Query each symbol from DB and print its recap to stderr."""
    for sym in symbols:
        start_ts = to_optional_ts(from_date)
        end_ts = to_optional_ts(to_date)
        df = query_candles(sym.upper(), start_ts, end_ts, bar)
        click.echo(recap(df, sym.upper()), err=True)


def read_ohlcv_stdin() -> pd.DataFrame:
    """Read OHLCV JSON lines from stdin and return a DataFrame.

    Expects: {"t":"2026-01-05T...","o":100.5,"h":...,"l":...,"c":...,"v":...}
    """
    if sys.stdin.isatty():
        return pd.DataFrame()

    records: list[dict] = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            click.echo(f"Warning: skipping invalid JSON line: {line[:80]}", err=True)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["t"])
    df = df.set_index("timestamp").drop(columns=["t"])

    renames = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    df = df.rename(columns=renames)
    return df


def resolve_symbol_list(symbols: tuple[str, ...], universe: Optional[str]) -> list[str]:
    """Resolve CLI symbol args — universe name or inline list."""
    if universe:
        return load_universe_config(universe).symbols
    if symbols:
        return list(map(str.upper, symbols))
    raise click.UsageError("provide SYMBOLS or --universe/-U")


def display_preview(
    result: PreviewResult,
    from_date: str,
    to_date: Optional[str],
) -> None:
    """Display a PreviewResult to stderr — shared by preview and dl commands."""
    click.echo(
        f"resolved {result.resolved}, {result.total_gaps} fetch gaps across all symbols\n",
        err=True,
    )

    for plan in result.plans:
        ticker = plan.ticker
        click.echo(f"{ticker} (conid {plan.conid}):", err=True)

        if plan.gaps:
            for g in plan.gaps:
                click.echo(
                    f"  fetch: {g[0].strftime('%Y-%m-%d %H:%M')}"
                    f" → {g[1].strftime('%Y-%m-%d %H:%M')}",
                    err=True,
                )
        else:
            click.echo("  fetch: up to date", err=True)

        start_ts = to_optional_ts(from_date)
        end_ts = to_optional_ts(to_date)
        df = query_candles(ticker.upper(), start_ts, end_ts)
        if not df.empty:
            gaps = find_gaps_48h(df)
            if gaps:
                for g in gaps:
                    click.echo(
                        f"  gap >48h: {g[0].strftime('%Y-%m-%d %H:%M')}"
                        f" → {g[1].strftime('%Y-%m-%d %H:%M')}",
                        err=True,
                    )
            else:
                click.echo("  gap >48h: none", err=True)
        else:
            click.echo("  gap >48h: no existing data to check", err=True)

        click.echo("", err=True)
