"""Tests for the `src.indicators.ta` indicator helpers (pure functions).

Primarily regression coverage for the MFI fix (the negative-money-flow
denominator bug that produced unbounded MFI) plus a sanity check that the
core momentum indicators stay bounded/well-formed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.ta import mfi, obv, obv_z, rsi


def _frame_from_closes(closes: pd.Series, volume: float = 1e6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": volume,
        }
    )


def test_mfi_bounded_0_100_on_mixed_data():
    """A realistic up/down/sideways series stays within 0..100 (was the bug:
    negative flows made MFI explode to +/- thousands)."""
    rng = np.random.default_rng(42)
    rets = rng.choice([-0.02, 0.0, 0.02], size=300)
    closes = 100.0 * np.exp(np.cumsum(rets))
    f = _frame_from_closes(pd.Series(closes))
    m = mfi(f["high"], f["low"], f["close"], f["volume"], window=14).dropna()
    assert len(m) > 0
    assert float(m.min()) >= 0.0
    assert float(m.max()) <= 100.0


def test_mfi_pure_uptrend_is_100():
    closes = pd.Series([100.0 * 1.01**i for i in range(50)])
    f = _frame_from_closes(closes)
    m = mfi(f["high"], f["low"], f["close"], f["volume"], window=14).dropna()
    assert len(m) > 0
    # Every bar is an up-bar -> only positive money flow -> MFI ~ 100.
    assert float(m.iloc[-1]) == 100.0


def test_mfi_pure_downtrend_is_0():
    closes = pd.Series([100.0 * 0.99**i for i in range(50)])
    f = _frame_from_closes(closes)
    m = mfi(f["high"], f["low"], f["close"], f["volume"], window=14).dropna()
    assert len(m) > 0
    # Every bar is a down-bar -> only negative money flow -> MFI ~ 0.
    assert float(m.iloc[-1]) == 0.0


def test_mfi_rises_with_accumulation_direction():
    """MFI should be higher when up-volume dominates than when it is balanced,
    monotone in the balance of positive vs negative money flow."""
    up = pd.Series([100.0 * 1.01**i for i in range(40)])
    # Uptrend with heavy volume + sparse small down-moves keeps MFI high.
    down = pd.Series([100.0 * 1.01**i for i in range(40)][::-1])
    m_up = mfi(up * 1.01, up * 0.99, up, pd.Series(1e6, index=up.index), 14).dropna()
    m_dn = mfi(
        down * 1.01, down * 0.99, down, pd.Series(1e6, index=down.index), 14
    ).dropna()
    assert float(m_up.iloc[-1]) > float(m_dn.iloc[-1])


def test_mfi_handles_flat_series_no_nan_blowup():
    """A constant close (all flat bars) has zero flows; MFI should be nan, not
    explode, and the dropped output must be empty (no garbage)."""
    closes = pd.Series([100.0] * 40)
    f = _frame_from_closes(closes)
    m = mfi(f["high"], f["low"], f["close"], f["volume"], window=14)
    assert m.isna().all() or m.dropna().empty


def test_rsi_bounded_0_100():
    closes = pd.Series([100.0 * (1 + 0.02 * np.sin(i / 3.0)) for i in range(60)])
    r = rsi(closes, 14).dropna()
    assert float(r.min()) >= 0.0
    assert float(r.max()) <= 100.0


# ---- OBV / OBV z-score (direct cumulative-flow channel) ----


def test_obv_accumulates_on_up_days_distributes_on_down_days():
    close = pd.Series([10.0, 11.0, 10.5, 12.0])
    volume = pd.Series([100.0, 200.0, 300.0, 400.0])
    # day0 diff NaN -> 0; day1 +1 up (+200); day2 -0.5 down (-300); day3 +1.5 up (+400)
    out = obv(close, volume)
    assert list(out) == [0.0, 200.0, -100.0, 300.0]


def test_obv_z_positive_when_flow_accumulates():
    closes = pd.Series([float(x) for x in range(10, 60)])  # strictly rising
    z = obv_z(closes, pd.Series(1.0, index=closes.index), window=40).dropna()
    assert len(z) > 0
    assert float(z.iloc[-1]) > 0


def test_obv_z_negative_when_flow_distributes():
    closes = pd.Series([float(x) for x in range(60, 10, -1)])  # strictly falling
    z = obv_z(closes, pd.Series(1.0, index=closes.index), window=40).dropna()
    assert len(z) > 0
    assert float(z.iloc[-1]) < 0


def test_obv_z_nan_when_flow_flat():
    # flat price -> OBV never moves -> zero rolling std -> fail closed
    closes = pd.Series([10.0] * 60)
    z = obv_z(closes, pd.Series(1.0, index=closes.index), window=40)
    assert z.isna().all()


def test_obv_z_window_off_is_all_nan():
    closes = pd.Series([float(x) for x in range(10, 60)])
    z = obv_z(closes, pd.Series(1.0, index=closes.index), window=0)
    assert z.isna().all()


def test_obv_z_first_window_rows_are_nan():
    closes = pd.Series([float(x) for x in range(10, 60)])
    z = obv_z(closes, pd.Series(1.0, index=closes.index), window=40)
    assert z.iloc[:39].isna().all()
    assert np.isfinite(z.iloc[-1])
