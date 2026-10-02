"""Functional data feed module.

Provides pure functions for loading and transforming market data into Candles.
Can be injected into Backtest engine via EngineDependencies.

Usage:
    from src.bt.data_feed import load_candles, candles_from_dataframe

    # Load data
    df = load_candles(["AAPL", "MSFT"], start, end, "1h")

    # Generate candles
    async for candle in candles_from_dataframe(df, ["AAPL", "MSFT"]):
        process(candle)
"""

import pandas as pd
from dataclasses import dataclass
from pandas import Timestamp
from typing import List, cast

from src.bt.state import Candle
from src.utils import get_local_candles
from src.data import sync_data
from src.data.xcal import is_non_trading_day

# Maximum tolerated discontinuity between consecutive bars of a symbol.
# Any gap strictly greater than this is treated as missing/corrupt data and
# aborts the load — indicators across an unresolved hole are meaningless.
DEFAULT_MAX_GAP: pd.Timedelta = cast(pd.Timedelta, pd.Timedelta(hours=96))


@dataclass(frozen=True)
class GapBreak:
    """A single discontinuity in a symbol's bars exceeding a threshold."""

    symbol: str
    prev_ts: pd.Timestamp
    next_ts: pd.Timestamp
    duration: pd.Timedelta


def _trading_days_between(prev_ts: pd.Timestamp, next_ts: pd.Timestamp) -> int:
    """NYSE trading days strictly between two bar timestamps.

    A multi-day hole that lands entirely on weekends/holidays (e.g. the
    Christmas/Thanksgiving long weekend) is a normal market closure, not
    missing data, so the gap guard must not flag it.
    """
    day = cast(pd.Timestamp, prev_ts + pd.Timedelta(days=1)).normalize()
    end = next_ts.normalize()
    missing = 0
    while day < end:
        if not is_non_trading_day(day.to_pydatetime()):
            missing += 1
        day = cast(pd.Timestamp, day + pd.Timedelta(days=1))
    return missing


def detect_gaps(
    df: pd.DataFrame,
    symbols: List[str],
    max_gap: pd.Timedelta = DEFAULT_MAX_GAP,
) -> dict[str, tuple[GapBreak, ...]]:
    """Return, per symbol, the gaps (delta between consecutive bars) > max_gap.

    A gap is the time elapsed between a bar's timestamp and the *previous*
    bar's timestamp. A pre-existing first bar is not a gap (no prior bar).

    Args:
        df: MultiIndex-column frame (symbol, field) as produced by load_candles.
        symbols: Symbols whose columns to inspect.
        max_gap: Threshold; only gaps strictly greater than this are reported.

    Returns:
        Mapping ``symbol -> tuple[GapBreak, ...]``. Symbols with no gaps do not
        appear as keys (use ``.get(symbol, ())`` for a safe default).
    """
    report: dict[str, tuple[GapBreak, ...]] = {}
    for symbol in symbols:
        try:
            block = df.xs(symbol, axis=1)
        except KeyError:
            continue
        # The MultiIndex-column frame is outer-joined across symbols, so other
        # symbols' timestamps appear as all-NaN rows in this block. Only the
        # symbol's own (non-NaN) bars are real — drop the padding before
        # measuring continuity.
        time_idx = pd.DatetimeIndex(block.dropna(subset=["close"]).index)
        if len(time_idx) < 2:
            continue
        deltas = time_idx[1:] - time_idx[:-1]
        breaks: list[GapBreak] = []
        for idx, delta in enumerate(deltas):
            assert isinstance(delta, pd.Timedelta)
            prev_ts, next_ts = time_idx[idx], time_idx[idx + 1]
            # Time threshold AND a genuinely skipped trading day: a long
            # weekend (or holiday cluster) can exceed max_gap without any data
            # being missing.
            if delta > max_gap and _trading_days_between(prev_ts, next_ts) > 0:
                breaks.append(
                    GapBreak(
                        symbol=symbol,
                        prev_ts=prev_ts,
                        next_ts=next_ts,
                        duration=delta,
                    )
                )
        if breaks:
            report[symbol] = tuple(breaks)
    return report


class DataIntegrityError(Exception):
    """Raised when a symbol's candle series contains a gap > the limit."""


def load_candles(
    symbols: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    bar: str,
    max_gap: pd.Timedelta = DEFAULT_MAX_GAP,
) -> pd.DataFrame:
    """Pure function - loads candles for given symbols and date range.

    Data integrity: after assembling the MultiIndex frame, every symbol's bar
    sequence is checked for discontinuities. If any symbol contains a gap
    strictly greater than ``max_gap`` (default 96h) between consecutive bars,
    a :class:`DataIntegrityError` is raised and loading aborts — a hole that
    wide is presumed to be missing/corrupt data, and indicators computed across
    it would be meaningless. Pass ``max_gap`` as ``pd.Timedelta.max`` (or a
    large sentinel) to skip the check.

    Args:
        symbols: List of ticker symbols
        start: Start timestamp
        end: End timestamp
        bar: Bar size (e.g., "1h", "1d")
        max_gap: Maximum tolerated discontinuity between consecutive bars.

    Returns:
        DataFrame with MultiIndex (symbol, timestamp) containing OHLCV data

    Raises:
        DataIntegrityError: If any symbol has a gap > ``max_gap``.
    """
    time_series = [get_local_candles(s, start, end, bar) for s in symbols]
    df = pd.concat(time_series, axis=1, keys=symbols, sort=False)
    # The engine iterates candles by index order (per-symbol arrays aligned by
    # row). Guarantee chronological order regardless of source row order so
    # downstream consumers (benchmarks, generators) never see reversed/shuffled
    # timestamps.
    df = df.sort_index()

    # ``pd.Timedelta.max`` is the documented "skip the check" sentinel — don't
    # scan for gaps when the caller explicitly opted out. (NB: pandas returns a
    # fresh ``Timedelta.max`` object per access, so compare by value, not id.)
    if max_gap == pd.Timedelta.max:
        return df

    report = detect_gaps(df, list(symbols), max_gap=max_gap)
    if report:
        lines = [
            f"{sym}: {', '.join(f'{b.prev_ts} -> {b.next_ts} (gap {b.duration})' for b in breaks)}"
            for sym, breaks in report.items()
        ]
        raise DataIntegrityError(
            f"Candle data contains gaps > {max_gap}: {'; '.join(lines)}"
        )
    return df


def candles_from_dataframe(
    df: pd.DataFrame,
    symbols: List[str],
):
    """Pure async generator - yields Candles (OHLCV bars) from a DataFrame.

    This function is stateless - given the same DataFrame and symbols,
    it will always yield the same sequence of Candles.

    Args:
        df: DataFrame with MultiIndex (symbol, timestamp) containing OHLCV
        symbols: List of symbols to generate candles for

    Yields:
        Candle objects for each symbol at each timestamp
    """
    if df.empty or len(df.index) == 0:
        return

    all_timestamps = sorted(
        set().union(*[set(df.xs(s, axis=1).index) for s in symbols])
    )

    for ts in all_timestamps:
        for s in symbols:
            try:
                row = df.xs(s, axis=1).loc[ts]
                pdt = cast(pd.Timestamp, Timestamp(ts))
                assert not pd.isna(pdt)
                yield Candle(
                    timestamp=pdt,
                    symbol=s,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            except KeyError:
                continue


def sync_and_load(
    symbols: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    bar: str,
) -> pd.DataFrame:
    """Sync data from IBKR and load into DataFrame.

    This is the main entry point for production use.
    """
    import asyncio

    async def _sync():
        await sync_data(symbols, from_date=start.date())

    asyncio.run(_sync())
    return load_candles(symbols, start, end, bar)
