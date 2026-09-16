"""Indicator helper functions for use inside trading strategies.

Similar to Pine Script functions, these operate on price series/DataFrames
and can be used from strategies with data from self.model.

Usage from strategy:
    from src.indicators.ta import ema, rsi, atr

    def on_candle(self, candle, open_trade):
        # Get last 14 bars of close prices for symbol
        closes = self.model.market_data[-14:].close["AAPL"]

        # Apply indicators
        ema_9 = ema(closes, 9)
        rsi_14 = rsi(closes, 14)

        # For ATR (needs OHLC)
        bars = self.model.market_data[-14:].for_symbol("AAPL")
        atr_14 = atr(bars["high"], bars["low"], bars["close"], 14)
"""

from typing import Tuple
import pandas as pd
import numpy as np


def ema(data: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average(s)."""
    return data.ewm(span=span, adjust=False).mean()


def sma(data: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average.

    Args:
        data: Price series or DataFrame (columns=symbols, index=time)
        window: MA period

    Returns:
        Series (if input was Series) or DataFrame (if input was DataFrame)
    """
    return data.rolling(window=window).mean()


def rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index.

    Args:
        data: Price series
        window: RSI period (default 14)

    Returns:
        Series with RSI values (0-100)
    """
    delta = data.diff()

    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    # Calculate RS and RSI
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Calculate Average True Range.

    Args:
        high: High prices series
        low: Low prices series
        close: Close prices series
        window: ATR period (default 14)

    Returns:
        Series with ATR values
    """
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate ATR using Wilder's smoothing (RMA)
    atr = tr.ewm(alpha=1 / window, adjust=False).mean()

    return atr


def bollinger_bands(
    data: pd.Series, window: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands.

    Args:
        data: Price series
        window: MA period (default 20)
        num_std: Number of standard deviations (default 2.0)

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle: pd.Series = sma(data, window)
    std_val = data.rolling(window=window).std().iloc[-1]

    upper = middle + (std_val * num_std)
    lower = middle - (std_val * num_std)

    return upper, middle, lower


def macd(
    data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Calculate MACD (Moving Average Convergence Divergence).

    Args:
        data: Price series
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)

    Returns:
        DataFrame with columns: macd_line, signal_line, histogram
    """
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {
            "macd_line": macd_line,
            "signal_line": signal_line,
            "histogram": histogram,
        }
    )


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_window: int = 14,
    d_window: int = 3,
) -> pd.DataFrame:
    """Calculate Stochastic Oscillator.

    Args:
        high: High prices series
        low: Low prices series
        close: Close prices series
        k_window: %K period (default 14)
        d_window: %D period (default 3)

    Returns:
        DataFrame with columns: k, d
    """
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()

    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_window).mean()

    return pd.DataFrame(
        {
            "k": k,
            "d": d,
        }
    )


def momentum(data: pd.Series, window: int = 10) -> pd.Series:
    """Calculate Momentum.

    Args:
        data: Price series
        window: Momentum period (default 10)

    Returns:
        Series with momentum values (current - N periods ago)
    """
    return data - data.shift(window)


def volatility(data: pd.Series, window: int = 20, annualized: bool = True) -> pd.Series:
    """Calculate rolling volatility (standard deviation of returns).

    Args:
        data: Price series
        window: Rolling window (default 20)
        annualized: Whether to annualize (multiply by sqrt(252))

    Returns:
        Series with volatility values
    """
    returns = np.log(data / data.shift(1))
    vol = returns.rolling(window=window).std()

    if annualized:
        vol = vol * np.sqrt(252)

    return vol


def vwma(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Volume-Weighted Moving Average.

    Args:
        close: Close prices series
        volume: Volume series
        window: MA period (default 20)

    Returns:
        Series with VWMA values
    """
    return (close * volume).rolling(window=window).sum() / volume.rolling(
        window=window
    ).sum()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume.

    Args:
        close: Close prices series
        volume: Volume series

    Returns:
        Series with OBV values (cumulative)
    """
    direction = np.sign(close.diff())
    obv = volume * direction
    return obv.fillna(0).cumsum()


def obv_z(close: pd.Series, volume: pd.Series, window: int = 40) -> pd.Series:
    """OBV level z-score against its own trailing ``window`` (mean/std).

    The direct cumulative-flow read: sign = money-flow direction, magnitude =
    strength off the name's own noise floor (``~+1.5/-1.5`` starts to be
    notable). Rows inside the first ``window`` bars (and any row with a zero
    rolling std) are ``NaN`` so callers fail closed.

    Args:
        close: Close prices series
        volume: Volume series
        window: trailing window for the OBV level's mean/std (bars)

    Returns:
        Series with the OBV z-score, ``NaN`` where it cannot be computed.
    """
    if window <= 0:
        return pd.Series(np.nan, index=close.index, dtype=float)
    level = obv(close, volume)
    roll = level.rolling(window=window)
    std = roll.std()
    z = (level - roll.mean()) / std.replace(0, np.nan)
    return z.where(np.isfinite(z))


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Calculate Money Flow Index.

    Args:
        high: High prices series
        low: Low prices series
        close: Close prices series
        volume: Volume series
        window: MFI period (default 14)

    Returns:
        Series with MFI values (0-100)
    """
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume

    money_flow_sign = np.sign(typical_price.diff())
    signed_money_flow = raw_money_flow * money_flow_sign

    # Positive and negative flows are both expressed as *magnitudes* (>= 0);
    # negative flow sums the absolute size of the money-flow that left on
    # down-bars. Negating before the mask keeps the denominator positive so
    # ``money_ratio`` stays in [0, inf) and MFI is bounded to 0..100.
    positive_flow = (
        signed_money_flow.where(signed_money_flow > 0, 0).rolling(window=window).sum()
    )
    negative_flow = (
        (-signed_money_flow)
        .where(signed_money_flow < 0, 0)
        .rolling(window=window)
        .sum()
    )

    money_ratio = positive_flow / negative_flow
    mfi = 100 - (100 / (1 + money_ratio))

    return mfi


def efficiency_ratio(data: pd.Series, window: int = 10) -> pd.Series:
    """Kaufman Efficiency Ratio: net path vs total path over ``window`` bars.

    ER = |close[t] - close[t-window]| / sum(|close[i] - close[i-1]|).

    * ER close to 1  -> a clean directional move (high signal-to-noise).
    * ER close to 0  -> a choppy/oscillating move (net displacement far
      smaller than the cumulative bar-to-bar travel) — the textbook
      mean-reversion backdrop.

    Scale-free and bounded [0, 1]; NaN through the first ``window`` bars and
    whenever the denominator (gross travel) is exactly 0.
    """
    net = (data - data.shift(window)).abs()
    gross = data.diff().abs().rolling(window).sum()
    er = net / gross.replace(0, np.nan)
    return er


def normalized_slope(
    price: pd.Series,
    scale: pd.Series,
    lookback: int,
) -> pd.Series:
    """OLS end-to-end slope of ``price`` over ``lookback``, / ``scale``.

    The leading indicator of whether price is *drifting directionally*. It
    fits a least-squares line to the last ``lookback`` price points and reports
    the per-bar OLS slope, normalized by a scale series (e.g. an ATR, a rolling
    std, or the mean price)::

        normalized_slope = ols_slope(price, lookback) / scale

    * |value| large  -> a persisting directional trend (mean-reversion hostile).
    * |value| small  -> a flat/sideways equilibrium (fadeable).

    Both ``price`` and ``scale`` share the same index/alignment and must have
    at least ``lookback`` valid rows before the head (the trailing ``NaN`` head
    is preserved). ``scale`` is applied element-wise at the line's end point,
    so the result is per-bar and cursor-truncation safe.
    """
    prices = np.asarray(price, dtype=np.float64)
    scales = np.asarray(scale, dtype=np.float64)
    n = len(prices)
    out = np.full(n, np.nan, dtype=np.float64)
    if lookback < 2 or n < lookback:
        return pd.Series(out, index=price.index)

    from numpy.lib.stride_tricks import sliding_window_view

    # Window ``i`` spans prices[i : i+lookback]; its per-bar OLS slope belongs
    # to that window's final bar (index i+lookback-1). ``slopes`` is aligned so
    # ``slopes[k]`` = the slope of the lookback window ending at bar ``k``.
    windows = sliding_window_view(prices, lookback)  # shape (n-lb+1, lb)
    x = np.arange(lookback, dtype=np.float64)
    xc = x - x.mean()
    denom = float(xc @ xc)
    y_c = windows - windows.mean(axis=1, keepdims=True)
    per_slope = (y_c @ xc) / denom  # (n-lb+1,)

    dst = np.full(n, np.nan, dtype=np.float64)
    dst[lookback - 1 :] = per_slope  # per_slope[i] ends at bar lookback-1+i

    sc = scales[lookback - 1 :]
    safe_sc = np.where(sc == 0, np.nan, sc)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = dst[lookback - 1 :] / safe_sc
    out = np.full(n, np.nan, dtype=np.float64)
    out[lookback - 1 :] = result
    return pd.Series(out, index=price.index)


def lsma(data: pd.Series, window: int = 14, offset: int = 0) -> pd.Series:
    """Calculate Least Squares Moving Average (LSMA).

    Args:
        data: Price series
        window: Lookback period
        offset: Forecast offset (0 = current bar)

    Returns:
        Series with LSMA values
    """
    if window <= 1:
        return data

    x = np.arange(window)
    x_mean = x.mean()
    x_denom = ((x - x_mean) ** 2).sum()

    def _calc(values: np.ndarray) -> float:
        y = values
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / x_denom
        intercept = y_mean - slope * x_mean
        return slope * (window - 1 + offset) + intercept

    return data.rolling(window=window).apply(_calc, raw=True)


def _dmi_components(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff()
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    tr_smooth = tr.ewm(alpha=1 / window, adjust=False).mean()
    plus_dm_smooth = (
        pd.Series(plus_dm, index=high.index).ewm(alpha=1 / window, adjust=False).mean()
    )
    minus_dm_smooth = (
        pd.Series(minus_dm, index=high.index).ewm(alpha=1 / window, adjust=False).mean()
    )

    tr_safe = tr_smooth.replace(0, np.nan)
    plus_di = 100 * (plus_dm_smooth / tr_safe)
    minus_di = 100 * (minus_dm_smooth / tr_safe)

    return plus_di.fillna(0), minus_di.fillna(0), tr_safe.fillna(0)


def plus_di(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Calculate +DI (Directional Indicator)."""
    plus, _minus, _tr = _dmi_components(high, low, close, window)
    return plus


def minus_di(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Calculate -DI (Directional Indicator)."""
    _plus, minus, _tr = _dmi_components(high, low, close, window)
    return minus


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Calculate ADX (Average Directional Index)."""
    plus, minus, _tr = _dmi_components(high, low, close, window)
    denom = (plus + minus).replace(0, np.nan)
    dx = 100 * (plus - minus).abs() / denom
    return dx.ewm(alpha=1 / window, adjust=False).mean().fillna(0)
