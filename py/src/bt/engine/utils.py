from src.bt.state import Candle, BacktestState
from typing import Generator, Mapping, Sequence, cast
from src.data.resample import resample_multiindex
import pandas as pd
import numpy as np


def _df_to_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """Convert MultiIndex columns df to MultiIndex (symbol, timestamp) format."""
    if df.empty:
        return df

    # Check if columns are MultiIndex (symbol, field)
    if isinstance(df.columns, pd.MultiIndex):
        # Convert to (symbol, timestamp) format
        frames = []
        for symbol in df.columns.get_level_values(0).unique():
            sym_df = df.xs(symbol, axis=1, level=0).copy()
            # Reset the DatetimeIndex to get timestamp column
            sym_df = sym_df.reset_index()
            # First column is the timestamp (original index)
            ts_col = sym_df.columns[0]
            sym_df = sym_df.rename(columns={ts_col: "timestamp"})
            # Add symbol column
            sym_df["symbol"] = symbol
            sym_df = sym_df.set_index(["symbol", "timestamp"])
            frames.append(sym_df)

        if frames:
            return pd.concat(frames)

    return df


def build_htf_feeds(
    df: pd.DataFrame,
    htf_intervals: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Resample the base feed to every higher timeframe exactly once.

    Pure function. The returned mapping is handed to BOTH ``candle_generator``
    (which yields HTF candles) and ``TaContext.from_data`` (which prefetches
    HTF indicator arrays), so both consumers derive from one resample and can
    never diverge. Keys are the frequency strings (e.g. ``"4h"``); values are
    ``(symbol, timestamp)``-MultiIndex OHLCV frames from
    :func:`resample_multiindex` (session-anchored for intraday hour buckets,
    calendar-aligned otherwise).

    ``dropna()`` is applied uniformly so the session-anchored and calendar
    paths agree on which buckets are dropped; without it a NaN-close bucket
    would survive one path and not the other, desynchronising the store's HTF
    rows from the feed's HTF rows.

    Args:
        df: Base MultiIndex-column OHLCV DataFrame (as consumed by the engine).
        htf_intervals: Higher-timeframe frequencies; empty yields an empty dict.

    Returns:
        ``{freq: resampled_frame}`` — one entry per requested interval.
    """
    if df.empty or not htf_intervals:
        return {}
    df_mi = _df_to_multiindex(df)
    return {
        freq: resample_multiindex(df_mi, freq, completed_only=False).dropna()
        for freq in htf_intervals
    }


def candle_generator(
    df: pd.DataFrame,
    config,
    htf_feeds: Mapping[str, pd.DataFrame] | None = None,
) -> Generator[Candle, None, None]:
    """Create a generator that yields Candles (OHLCV bars) from a DataFrame.

    This is a pure function - given the same inputs, it always yields
    the same sequence of Candles. Yields base timeframe and HTF Candles
    (if configured) interleaved by timestamp.

    Uses pre-extracted numpy arrays to avoid per-candle DataFrame lookups.

    Args:
        df: DataFrame with MultiIndex (symbol, timestamp) containing OHLCV
        config: StrategyConfig with symbols, bar, and htf settings, or list of symbols (legacy)
        htf_feeds: Pre-resampled HTF frames from :func:`build_htf_feeds`. When
            ``None`` the feeds are built here from ``config.bars[1:]``
            (back-compat for direct callers/tests); production passes the same
            mapping instance also handed to ``TaContext.from_data``.

    Yields:
        Candle objects for each symbol at each timestamp
    """
    # Handle both StrategyConfig and legacy list of symbols
    if hasattr(config, "symbols"):
        symbols = config.symbols
        base_interval = getattr(config, "bars", ["1h"])[0]
        htf_intervals = (
            getattr(config, "bars", ["1h"])[1:] if hasattr(config, "bars") else []
        )
    else:
        symbols = config
        base_interval = "1h"
        htf_intervals = []

    if df.empty or len(df.index) == 0:
        return

    # Pre-extract base OHLCV into numpy arrays for fast iteration.
    # df has MultiIndex columns: (symbol, field). Extract each (sym, field)
    # into a plain numpy array indexed by row position.
    fields = ["open", "high", "low", "close", "volume"]
    base_arrays: dict[tuple[str, str], np.ndarray] = {}
    for s in symbols:
        for f in fields:
            col = (s, f)
            if col in df.columns:
                base_arrays[(s, f)] = df[col].to_numpy(dtype=float)

    timestamps = df.index

    # HTF frames: the caller's shared mapping when supplied, else resample once
    # here (back-compat). Never both — one resample per HTF interval per run.
    htf_dfs: Mapping[str, pd.DataFrame] = (
        htf_feeds if htf_feeds is not None else build_htf_feeds(df, htf_intervals)
    )

    # Get all unique timestamps across base and HTF
    all_timestamps: set = set()
    for s in symbols:
        all_timestamps.update(timestamps)
    for freq, htf_df in htf_dfs.items():
        if not htf_df.empty:
            for s in symbols:
                try:
                    all_timestamps.update(htf_df.xs(s, level="symbol").index)
                except KeyError:
                    pass

    sorted_timestamps = sorted(all_timestamps)
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}  # fast lookup

    # Pre-compute HTF: { (freq, sym): {timestamp_ns: (open, high, low, close, volume)} }
    htf_points: dict[
        tuple[str, str], dict[int, tuple[float, float, float, float, float]]
    ] = {}
    for freq, htf_df in htf_dfs.items():
        if htf_df.empty:
            continue
        for s in symbols:
            try:
                sym_htf = htf_df.xs(s, level="symbol")
                htf_ts = sym_htf.index.to_numpy()
                htf_vals = {}
                for i, ts in enumerate(htf_ts):
                    htf_vals[hash(ts)] = (
                        float(sym_htf["open"].iloc[i]),
                        float(sym_htf["high"].iloc[i]),
                        float(sym_htf["low"].iloc[i]),
                        float(sym_htf["close"].iloc[i]),
                        float(sym_htf["volume"].iloc[i]),
                    )
                htf_points[(freq, s)] = htf_vals
            except KeyError:
                continue

    for ts in sorted_timestamps:
        pdt = cast(pd.Timestamp, pd.Timestamp(ts))
        if pd.isna(pdt):
            continue

        idx = ts_to_idx.get(ts)
        if idx is not None:
            for s in symbols:
                # Multi-symbol loads outer-join on the union of timestamps, so a
                # symbol missing data at this bar has NaN rows (closed market /
                # data gap). Skipping that symbol's candle keeps NaN out of the
                # engine (mark-to-market positions_value and equity would
                # otherwise go NaN permanently).
                if not np.isfinite(base_arrays[(s, "close")][idx]):
                    continue
                yield Candle(
                    timestamp=pdt,
                    symbol=s,
                    open=float(base_arrays[(s, "open")][idx]),
                    high=float(base_arrays[(s, "high")][idx]),
                    low=float(base_arrays[(s, "low")][idx]),
                    close=float(base_arrays[(s, "close")][idx]),
                    volume=float(base_arrays[(s, "volume")][idx]),
                    interval=base_interval,
                )

        # Yield HTF ticks at their boundaries — O(1) dict lookup
        ts_h = hash(ts)
        for freq in htf_intervals:
            for s in symbols:
                htf_v = htf_points.get((freq, s), {}).get(ts_h)
                if htf_v is not None:
                    yield Candle(
                        timestamp=pdt,
                        symbol=s,
                        open=htf_v[0],
                        high=htf_v[1],
                        low=htf_v[2],
                        close=htf_v[3],
                        volume=htf_v[4],
                        interval=freq,
                    )


def merge_bt_state(a: BacktestState, b: dict):
    return BacktestState(
        portfolio=b.get("portfolio", a.portfolio),
        timestamp=b.get("timestamp", a.timestamp),
        pending_signals=b.get("pending_signals", a.pending_signals),
        risk_events=b.get("risk_events", a.risk_events),
        candles=b.get("candles", a.candles),
    )
