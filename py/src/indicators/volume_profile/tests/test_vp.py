"""Tests for the Volume Profile indicator (``src.indicators.volume_profile``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.volume_profile import VolumeProfile, volume_profile


def _frame(
    highs: list[float], lows: list[float], volumes: list[float]
) -> tuple[pd.Series, pd.Series, pd.Series]:
    return (
        pd.Series(highs, dtype=float),
        pd.Series(lows, dtype=float),
        pd.Series(volumes, dtype=float),
    )


def test_total_volume_is_conserved():
    """The sum of per-bin volume equals the raw total volume."""
    high, low, vol = _frame(
        [10.0, 12.0, 11.0, 13.0], [8.0, 10.0, 9.0, 11.0], [100.0, 50.0, 75.0, 25.0]
    )
    vp = volume_profile(high, low, vol, num_bins=20)
    assert float(vp.volume.sum()) == pytest.approx(float(vol.sum()))
    assert vp.total_volume == pytest.approx(float(vol.sum()))


def test_single_price_level_raises():
    high, low, vol = _frame([10.0, 10.0], [10.0, 10.0], [5.0, 5.0])
    with pytest.raises(ValueError):
        volume_profile(high, low, vol, num_bins=10)


def test_empty_input_raises():
    high = pd.Series([], dtype=float)
    low = pd.Series([], dtype=float)
    vol = pd.Series([], dtype=float)
    with pytest.raises(ValueError):
        volume_profile(high, low, vol, num_bins=10)


def test_nan_and_invalid_rows_are_skipped():
    """Rows with NaN or high < low must not poison the profile or total.

    The valid subset should behave like a smaller clean frame whose total
    volume is reproduced exactly."""
    clean_h, clean_l, clean_v = _frame(
        [10.0, 12.0, 11.0, 13.0], [8.0, 10.0, 9.0, 11.0], [100.0, 50.0, 75.0, 25.0]
    )
    high = pd.concat([clean_h, pd.Series([float("nan"), 5.0, float("nan")])])
    low = pd.concat([clean_l, pd.Series([3.0, 10.0, float("nan")])])
    vol = pd.concat([clean_v, pd.Series([99.0, 1.0, 42.0])])

    vp = volume_profile(high, low, vol, num_bins=20)
    assert vp.total_volume == pytest.approx(250.0)  # only the 4 valid rows
    assert float(vp.volume.sum()) == pytest.approx(250.0)


def test_poc_points_to_dominant_level():
    """A candle that towers in volume over a narrow level should set the POC."""
    # Most volume sits right around price 10.0-10.5.
    high, low, vol = _frame(
        [10.5, 10.5, 10.5, 20.0, 5.0],
        [10.0, 10.0, 10.0, 15.0, 3.0],
        [1000.0, 1000.0, 1000.0, 1.0, 1.0],
    )
    vp = volume_profile(high, low, vol, num_bins=50)
    assert 10.0 <= vp.poc <= 10.5
    # The 3000-pt heavy cluster dwarfs the two 1-pt outliers, so the POC bin
    # must capture the bulk of it (overlap-weighted, not all of it).
    assert vp.poc_volume == pytest.approx(3000.0, rel=0.35)


def test_volume_spread_across_bins_narrows_range():
    """A wide low-volume candle contributes little per bin; total is conserved."""
    high, low, vol = _frame([20.0, 11.0], [1.0, 10.0], [2000.0, 1000.0])
    vp = volume_profile(high, low, vol, num_bins=50)
    assert vp.total_volume == pytest.approx(3000.0)
    # The wide candle's 2000 over a 19-pt range is ~105/bin vs 1000 on the
    # tight basket -> POC should sit at the tight 10-11 level.
    assert 10.0 <= vp.poc <= 11.0


def test_value_area_coverage_and_bounds():
    """Value Area should cover ~value_area_pct and be inside the price range."""
    rng = np.random.default_rng(7)
    n = 300
    mid = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    high = mid + rng.uniform(0.2, 1.0, n)
    low = mid - rng.uniform(0.2, 1.0, n)
    vol = rng.uniform(100, 1000, n)

    vp = volume_profile(
        pd.Series(high),
        pd.Series(low),
        pd.Series(vol),
        num_bins=50,
        value_area_pct=0.70,
    )
    ratio = vp.value_volume / vp.total_volume
    # Coverage reaches at least the target (may slightly overshoot per-bin).
    assert ratio >= 0.70 and ratio <= 1.0
    assert vp.val <= vp.poc <= vp.vah
    # VAH/VAL sit within the overall low/high range.
    assert vp.val >= float(min(low)) - 1e-9
    assert vp.vah <= float(max(high)) + 1e-9


def test_determinism():
    high, low, vol = _frame(
        [10.0, 12.0, 11.0, 13.0, 9.5],
        [8.0, 10.0, 9.0, 11.0, 7.0],
        [100.0, 50.0, 75.0, 25.0, 60.0],
    )
    a = volume_profile(high, low, vol, num_bins=40)
    b = volume_profile(high, low, vol, num_bins=40)
    assert a.poc == b.poc
    assert a.vah == b.vah
    assert a.val == b.val
    np.testing.assert_array_equal(a.bin_price, b.bin_price)
    np.testing.assert_array_equal(a.volume, b.volume)


def test_single_bin_squashes_whole_range():
    high, low, vol = _frame(
        [10.0, 12.0, 11.0, 13.0], [8.0, 10.0, 9.0, 11.0], [100.0, 50.0, 75.0, 25.0]
    )
    vp = volume_profile(high, low, vol, num_bins=1)
    assert len(vp.volume) == 1
    assert vp.total_volume == pytest.approx(250.0)
    assert vp.poc == pytest.approx((vp.val + vp.vah) / 2.0)


def test_invalid_parameters_raise():
    high, low, vol = _frame([10.0, 12.0], [8.0, 10.0], [100.0, 50.0])
    with pytest.raises(ValueError):
        volume_profile(high, low, vol, num_bins=0)
    with pytest.raises(ValueError):
        volume_profile(high, low, vol, value_area_pct=0.0)
    with pytest.raises(ValueError):
        volume_profile(high, low, vol, value_area_pct=1.5)
    with pytest.raises(ValueError):
        volume_profile(high, low, vol, num_bins=-3)


def test_returns_frozen_dataclass_alias():
    high, low, vol = _frame([10.0, 12.0], [8.0, 10.0], [100.0, 50.0])
    vp = volume_profile(high, low, vol, num_bins=20)
    assert isinstance(vp, VolumeProfile)
