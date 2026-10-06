"""Shared utility functions. Kept minimal — the heavy lifting lives in domain modules.

Only functions with >= 1 consumer remain. Visualization, live-trading, and ORM helpers
were moved to their respective modules or deleted.
"""

from typing import Any, Dict, List, Optional, cast
import pandas as pd
import numpy as np

from src.data.resample import resample_ohlcv

# ``to_optional_ts``/``parse_timestamp`` moved to the ``src.timestamps`` leaf so
# ``src.data._shared`` can import them without reaching back into ``src.utils``
# (which imports ``src.data.resample`` and closed an import cycle). Re-exported
# here so every existing ``from src.utils import ...`` keeps working.
from src.timestamps import parse_timestamp, to_optional_ts  # noqa: F401


_DEFAULT_START = pd.Timestamp("2020-01-01")
_DEFAULT_END = pd.Timestamp("2099-01-01")


def get_local_candles(
    symbol: str,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    bar: str = "1h",
) -> pd.DataFrame:
    from src.data.db import get_connection  # lazy: avoid src.data pkg-init cycle

    con = get_connection()
    cur = con.cursor()
    _start = start_date if start_date else _DEFAULT_START
    _end = end_date if end_date else _DEFAULT_END
    _start = cast(pd.Timestamp, _start)
    _end = cast(pd.Timestamp, _end)
    _sd = int(_start.timestamp() * 1000) if _start else 0
    _ed = int(_end.timestamp() * 1000)

    q = """
        select ticker as symbol,
               timestamp,
               open,
               high,
               low,
               close,
               volume
        from candle
        where ticker = UPPER(?)
          and timestamp >= ?
          and timestamp <= ?
        order by timestamp asc
    """
    res = cur.execute(q, (symbol, _sd, _ed))
    data = res.fetchall()
    con.close()
    columns = pd.Index(
        ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    )
    df = pd.DataFrame(data, columns=columns)
    df = df.assign(Date=pd.to_datetime(df["timestamp"], unit="ms"))
    df = df.set_index("Date").drop(columns=["timestamp"])
    if bar != "1h":
        return resample_ohlcv(df, bar, completed_only=True)
    return df


def calculate_zscore_spread(
    s1: pd.Series, s2: pd.Series, rolling_window: Optional[int] = None
):
    """Calculate z-score normalized spread between two price series."""
    if s1.empty or s2.empty or len(s1) != len(s2):
        return pd.Series(dtype=float)
    if rolling_window is not None and rolling_window > 0:
        return _calculate_rolling_zscore_spread(s1, s2, rolling_window)

    model = _get_ols_fit_model(s1, s2)
    alpha, beta = model.params
    scaled_s2 = alpha + beta * s2
    spread_series = s1 - scaled_s2
    if spread_series.empty:
        return pd.Series(dtype=float)
    return (spread_series - spread_series.mean()) / spread_series.std()


def symmetric_cointegration_p(price1, price2):
    """Symmetric cointegration p-value — average of both directions."""
    p1 = float(_eg_pvalue(price1, price2))
    p2 = float(_eg_pvalue(price2, price1))
    return float((p1 + p2) / 2)


def pick(d: Dict[str, Any], keys: List[str]):
    return {k: v for k, v in d.items() if k in keys}


def omit(d: Dict[str, Any], keys: List[str]):
    return {k: v for k, v in d.items() if k not in keys}


def get_ts(ds: str) -> pd.Timestamp:
    _ts = pd.Timestamp(ds)
    if isinstance(_ts, pd.Timestamp):
        return _ts
    raise ValueError(f"Failed when creating timestamp for {ds}")


# ── Private helpers ──────────────────────────────────────────────────────────


def _get_ols_fit_model(y, x):
    import statsmodels.api as sm

    y_arr = np.asarray(y)
    x_arr = np.asarray(x)
    if len(y_arr) == 0 or len(x_arr) == 0 or len(y_arr) != len(x_arr):
        raise ValueError("Empty or mismatched data for OLS")
    y_arr = np.log(y_arr).astype(float)
    x_arr = np.log(x_arr).astype(float)
    X = sm.add_constant(x_arr)
    return sm.OLS(y_arr, X).fit()


def _calculate_rolling_zscore_spread(s1: pd.Series, s2: pd.Series, rolling_window: int):
    if len(s1) < rolling_window:
        return pd.Series(dtype=float)

    import statsmodels.api as sm

    def calc_spread(window_idx):
        idx = s1.index[window_idx]
        if window_idx < rolling_window - 1:
            return np.nan
        start_idx = window_idx - rolling_window + 1
        end_idx = window_idx + 1
        s1_window = s1.iloc[start_idx:end_idx]
        s2_window = s2.iloc[start_idx:end_idx]
        if s1_window.empty or s2_window.empty:
            return np.nan
        X = sm.add_constant(s2_window)
        model = sm.OLS(s1_window, X).fit()
        alpha, beta = model.params
        return s1.loc[idx] - (alpha + beta * s2.loc[idx])

    spread_series = pd.Series([calc_spread(i) for i in range(len(s1))], index=s1.index)
    rolling_mean = spread_series.rolling(
        window=rolling_window, min_periods=rolling_window
    ).mean()
    rolling_std = spread_series.rolling(
        window=rolling_window, min_periods=rolling_window
    ).std()
    return (spread_series - rolling_mean) / rolling_std


def _hedge_ratio_residuals(y, x) -> pd.DataFrame:
    """Returns residuals from OLS(y ~ x) using log prices."""
    model = _get_ols_fit_model(y, x)
    alpha, beta = model.params
    return y - (alpha + beta * x)


def _eg_pvalue(price1, price2):
    """Engle-Granger p-value in one direction: price1 ~ price2"""
    from statsmodels.tsa.stattools import adfuller

    resid = _hedge_ratio_residuals(price1, price2)
    return adfuller(resid)[1]
