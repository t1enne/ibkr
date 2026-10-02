"""Pure functional OHLCV resampling utilities.

Provides bt-agnostic resampling with lookahead protection.
"""

import re
from typing import Any, Optional, cast
import pandas as pd


OHLCV_COLS: list[str] = ["open", "high", "low", "close", "volume"]


_HOUR_FREQ_RE = re.compile(r"^(\d+)h$")


def _intraday_hours(freq: str) -> int | None:
    """Hours for an intraday frequency like ``"4h"``; ``None`` for ``"1h"``
    (native granularity, left untouched) and for anything not ``<n>h``."""
    match = _HOUR_FREQ_RE.match(freq)
    if match is None:
        return None
    hours = int(match.group(1))
    return hours if hours > 1 else None


def _session_anchored_resample(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    """Aggregate a day's bars into ``hours``-wide buckets anchored at that
    day's first bar.

    Bucket count is ``ceil(session_hours / hours)`` -- independent of the
    midnight offset -- so a 6.5h US session gives exactly 2 ``"4h"`` bars
    instead of the midnight-floored 12:00 stub + 16:00 bucket + 20:00 singleton
    (which fragmented the old series into ~2.3 bars/day). Partial sessions
    (holidays, half days) legitimately yield fewer buckets.
    """
    idx = pd.DatetimeIndex(df.index)
    stamps = idx.to_series()
    starts = stamps.groupby(pd.Grouper(freq="D")).transform("min")
    width = pd.Timedelta(hours=hours)
    offset = ((stamps - starts) // width).astype("int64")
    agg = (
        df.assign(__bucket=starts + offset * width)
        .groupby("__bucket")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
    )
    agg.index.name = idx.name
    return agg


def _normalize_freq(freq: str) -> str:
    """Normalize a pandas offset string to avoid deprecated aliases.

    pandas 4.x deprecates the lowercase ``'d'`` daily offset in favor of
    ``'D'``. Rewrite a ``'d'`` frequency suffix to ``'D'`` (e.g. "1d" → "1D")
    so ``resample``/``floor`` calls don't emit the Pandas4Warning.
    Idempotent for already-normalized inputs.
    """
    if freq and freq[-1] == "d":
        return freq[:-1] + "D"
    return freq


def resample_ohlcv(
    df: pd.DataFrame,
    freq: str,
    *,
    completed_only: bool = True,
    current_ts: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Resample OHLCV data to a higher timeframe.

    Args:
        df: DataFrame with timestamp index and OHLCV columns
        freq: Resample frequency (e.g., "1h", "4h", "1D")
        completed_only: If True, exclude the current incomplete bucket (no lookahead)
        current_ts: Current timestamp for completed_only filtering

    Returns:
        Resampled DataFrame with OHLCV columns
    """
    freq = _normalize_freq(freq)
    if df.empty:
        return pd.DataFrame(columns=cast(Any, OHLCV_COLS))

    # Intraday hour buckets are session-anchored (see _session_anchored_resample)
    # so an N-hour frequency tiles the trading session evenly; daily and coarser
    # frequencies keep calendar alignment via pandas resample.
    hours = _intraday_hours(freq)
    if hours is not None:
        resampled = _session_anchored_resample(df, hours)
        if completed_only and current_ts is not None:
            width = pd.Timedelta(hours=hours)
            cur = pd.Timestamp(current_ts)
            forming = (resampled.index <= cur) & (cur < resampled.index + width)
            resampled = resampled[~forming]
        return resampled

    resampled = df.resample(freq).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    resampled = resampled.dropna()

    if completed_only and current_ts is not None:
        bucket_end = pd.Timestamp(current_ts.floor(freq))
        resampled = resampled[resampled.index < bucket_end]

    return resampled


def resample_multiindex(
    df: pd.DataFrame,
    freq: str,
    *,
    completed_only: bool = True,
    current_ts: Optional[pd.Timestamp] = None,
    symbol_col: str = "symbol",
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """Resample MultiIndex (symbol, timestamp) OHLCV data.

    Args:
        df: MultiIndex DataFrame with (symbol, timestamp) index and OHLCV columns
        freq: Resample frequency (e.g., "1h", "4h", "1D")
        completed_only: If True, exclude the current incomplete bucket (no lookahead)
        current_ts: Current timestamp for completed_only filtering
        symbol_col: Column name for symbol (if df is not MultiIndex)
        ts_col: Column name for timestamp (if df is not MultiIndex)

    Returns:
        Resampled DataFrame with MultiIndex (symbol, timestamp)
    """
    freq = _normalize_freq(freq)
    if df.empty:
        return pd.DataFrame(
            columns=cast(Any, OHLCV_COLS),
            index=pd.MultiIndex.from_tuples([], names=["symbol", "timestamp"]),
        )

    if isinstance(df.index, pd.MultiIndex):
        symbols = df.index.get_level_values("symbol").unique()
        resampled_frames = []

        for symbol in symbols:
            sym_df = df.xs(symbol, level="symbol")
            sym_resampled = resample_ohlcv(
                sym_df, freq, completed_only=False, current_ts=None
            )
            if not sym_resampled.empty:
                sym_resampled = sym_resampled.reset_index()
                sym_resampled["symbol"] = symbol
                sym_resampled = sym_resampled.set_index(["symbol", "timestamp"])
                resampled_frames.append(sym_resampled)

        if not resampled_frames:
            return pd.DataFrame(
                columns=cast(Any, OHLCV_COLS),
                index=pd.MultiIndex.from_tuples([], names=["symbol", "timestamp"]),
            )

        result = pd.concat(resampled_frames)

        if completed_only and current_ts is not None:
            bucket_end = pd.Timestamp(current_ts.floor(freq))
            result = result[result.index.get_level_values("timestamp") < bucket_end]

        return cast(pd.DataFrame, result)
    else:
        return resample_ohlcv(
            df, freq, completed_only=completed_only, current_ts=current_ts
        )
