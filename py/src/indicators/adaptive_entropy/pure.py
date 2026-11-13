"""Batch adaptive entropy trend indicator.

Pure, vectorized translation of the Pine Script in ``adaptive_entropy.pine``.
Given close / high / low series it returns the entropy-adaptive EMA, the
trend-trigger bands, and the quantised trend direction for every bar.

The core idea: Shannon entropy of the recent log-return distribution tells us
how *structured* (directional, low entropy) vs *noisy* (choppy, high entropy)
the market currently is.  That entropy drives (a) the EMA smoothing factor and
(b) the width of the ATR-scaled bands, so both the mean and the envelope adapt
to the regime detected on the fly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.adaptive_entropy.types import AdaptiveEntropyConfig
from src.indicators.adaptive_entropy.validate import require_finite_series


def _rolling_entropy(
    log_return: np.ndarray,
    lookback: int,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling normalized Shannon entropy + trend strength over the log-return
    history, computed window-by-window.  Returns ``(normalized_entropy,
    trend_strength)`` arrays of length ``N`` with ``nan`` until ``lookback``
    returns have accumulated.

    The Pine reference bins each window's log returns into ``num_bins`` equal
    width bins, converts counts to probabilities, and sums ``-p·log2(p)``,
    normalized by the maximum possible entropy ``log2(num_bins)``.  Trend
    strength is ``1 - entropy``.
    """
    n = log_return.shape[0]
    norm_entropy = np.full(n, np.nan)
    trend_strength = np.full(n, np.nan)
    max_entropy = np.log2(num_bins)

    # The first return is NaN (close[0] / undefined).  A full window of
    # ``lookback`` returns only exists from index ``lookback`` onward; earlier
    # bars stay NaN (indicator not warm).
    if lookback <= n:
        # slot ``j`` of sliding_window_view covers ``log_return[j:j+lookback]``,
        # which is exactly the window the original loop used for output bar
        # ``j + lookback - 1``.  Slot 0 covers ``log_return[0:lookback]`` which
        # starts at the NaN close[0] return and is discarded to match the batch
        # loop's warmup (it began at bar ``lookback``).  So only slots ``j >= 1``
        # are used, mapping to output bars ``[lookback, n)`` -- the same bars
        # the online class reports once ``ready``.
        windows = np.lib.stride_tricks.sliding_window_view(log_return, lookback)
        windows = windows[
            1:
        ]  # (n - lookback, lookback); relative slot j -> bar j + lookback
        lows = windows.min(axis=1)
        highs = windows.max(axis=1)
        rng = highs - lows

        bins = np.full(windows.shape, -1, dtype=np.int64)
        good_rows = rng > 0
        good = windows[good_rows]
        # Map the full window range to ``[0, num_bins-1]`` (max lands in the top
        # bin), matching the Pine histogram.
        offset = (good - lows[good_rows, None]) / rng[good_rows, None] * (num_bins - 1)
        floor = np.floor(offset).astype(np.int64)
        bins[good_rows] = np.clip(floor, 0, num_bins - 1)

        # Per-window histogram via broadcasting equality against bin labels.
        labels = np.arange(num_bins)[:, None]  # (num_bins, 1)
        hist = (bins[:, None, :] == labels).sum(axis=2)  # (slots, num_bins)

        probs = hist[good_rows] / lookback
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.where(probs > 0, np.log2(probs), 0.0)
        entropy = -(probs * logp).sum(axis=1)
        normalized = entropy / max_entropy if max_entropy > 0 else 0.5
        strength = 1.0 - normalized

        # Constant-return windows (zero range) -> Pine fallback of 0.5.
        slots_good = np.flatnonzero(good_rows)
        slots_flat = np.flatnonzero(~good_rows)
        norm_entropy[slots_good + lookback] = normalized
        trend_strength[slots_good + lookback] = strength
        norm_entropy[slots_flat + lookback] = 0.5
        trend_strength[slots_flat + lookback] = 0.5

    return norm_entropy, trend_strength


def adaptive_entropy(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    config: AdaptiveEntropyConfig | None = None,
) -> pd.DataFrame:
    """Compute the adaptive entropy trend over a price history (batch mode).

    Parameters
    ----------
    close:
        Close prices series.
    high:
        High prices series (for ATR).
    low:
        Low prices series (for ATR).
    config:
        Indicator hyper-parameters.  Defaults to ``AdaptiveEntropyConfig()``.

    Returns
    -------
    Raises
    ------
    NonFiniteBarError
        If any input price is not finite.  The post-warmup columns are derived
        from a running ATR/EMA, so a single blank row (a union-calendar feed
        where the symbol did not trade) would silently freeze ``trend``; the
        rejection is what keeps batch and online in step.

    Returns
    -------
    pd.DataFrame indexed as ``close`` with one row per bar.  Rows before the
    lookback window has filled carry ``nan`` for the rolling quantities, and
    the band/trend columns are ``nan`` / 0 until the EMA and ATR are defined.
    Columns: ``close``, ``entropy``, ``normalized_entropy``, ``trend_strength``,
    ``adaptive_ema``, ``atr``, ``fast_band_width``, ``slow_band_width``,
    ``inner_upper``, ``inner_lower``, ``outer_upper``, ``outer_lower``,
    ``trend``.
    """
    cfg = config or AdaptiveEntropyConfig()
    w = cfg.lookback

    close_arr = np.asarray(close, dtype=np.float64)
    high_arr = np.asarray(high, dtype=np.float64)
    low_arr = np.asarray(low, dtype=np.float64)
    n = close_arr.shape[0]
    require_finite_series(close_arr, high_arr, low_arr)

    if n == 0:
        return pd.DataFrame(
            columns=[
                "close",
                "entropy",
                "normalized_entropy",
                "trend_strength",
                "adaptive_ema",
                "atr",
                "fast_band_width",
                "slow_band_width",
                "inner_upper",
                "inner_lower",
                "outer_upper",
                "outer_lower",
                "trend",
            ],
            index=close.index,
        )

    # ---- log returns / Shannon entropy -----------------------------------
    prev_close = np.empty_like(close_arr)
    prev_close[0] = np.nan
    prev_close[1:] = close_arr[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_return = np.log(close_arr / prev_close)

    norm_entropy, trend_strength = _rolling_entropy(log_return, w, cfg.num_bins)

    # ---- ATR (Wilder RMA over the lookback) -------------------------------
    # Pine's ta.atr(lookback) uses alpha = 1/lookback regardless of how many
    # samples have elapsed; seed at the first finite TR for parity with
    # ``src.indicators.ta.atr`` which seeds the RMA identically.
    global_alpha = 1.0 / w
    tr = np.maximum(
        high_arr - low_arr,
        np.maximum(np.abs(high_arr - prev_close), np.abs(low_arr - prev_close)),
    ).astype(np.float64)
    atr_vec = np.full(n, np.nan)
    prev_atr: float | None = None
    for i in range(n):
        value = tr[i]
        if np.isnan(value):
            continue
        prev_atr = (
            value
            if prev_atr is None
            else (global_alpha * value + (1.0 - global_alpha) * prev_atr)
        )
        atr_vec[i] = prev_atr

    # ---- adaptive EMA ------------------------------------------------------
    adaptive_ema = np.full(n, np.nan)
    trend = np.zeros(n, dtype=np.int64)

    for i in range(n):
        if i == 0:
            adaptive_ema[i] = close_arr[i]
            continue
        ent = norm_entropy[i]
        if np.isnan(ent):
            # Not enough history yet: hold the previous EMA.
            adaptive_ema[i] = adaptive_ema[i - 1]
        else:
            adaptive_alpha = 2.0 / (w * (0.3 + ent * 1.4) + 1.0)
            adaptive_ema[i] = adaptive_ema[i - 1] + adaptive_alpha * (
                close_arr[i] - adaptive_ema[i - 1]
            )
        if np.isnan(adaptive_ema[i]) or np.isnan(atr_vec[i]):
            continue
        # Hold earlier trend (bar within the bands keeps its direction).
        prior = int(trend[i - 1]) if i > 0 else 0
        strength = trend_strength[i] if not np.isnan(trend_strength[i]) else 0.5
        fast_band_width = atr_vec[i] * cfg.fast_multiplier * (0.5 + strength)
        slow_band_width = atr_vec[i] * cfg.slow_multiplier * (0.5 + strength)
        inner_upper = adaptive_ema[i] + fast_band_width
        inner_lower = adaptive_ema[i] - fast_band_width
        if close_arr[i] > inner_upper:
            trend[i] = 1
        elif close_arr[i] < inner_lower:
            trend[i] = -1
        else:
            trend[i] = prior

    # ---- result frame ------------------------------------------------------
    fast_band_width = atr_vec * cfg.fast_multiplier * (0.5 + trend_strength)
    slow_band_width = atr_vec * cfg.slow_multiplier * (0.5 + trend_strength)
    return pd.DataFrame(
        {
            "close": close_arr,
            "entropy": norm_entropy,
            "normalized_entropy": norm_entropy,
            "trend_strength": trend_strength,
            "adaptive_ema": adaptive_ema,
            "atr": atr_vec,
            "fast_band_width": fast_band_width,
            "slow_band_width": slow_band_width,
            "inner_upper": adaptive_ema + fast_band_width,
            "inner_lower": adaptive_ema - fast_band_width,
            "outer_upper": adaptive_ema + slow_band_width,
            "outer_lower": adaptive_ema - slow_band_width,
            "trend": trend,
        },
        index=close.index,
    )
