"""Tests for the melt-up "ripper" up-day-magnitude predictor.

The predictor estimates a per-name "melt-up intensity" as the trailing mean size
of up-close days, used to veto shorting persistent rippers (large, ratcheting
up-moves) while leaving genuine rollover names admissible. The distinguishing
property central to the tests: rollover names that reward exhaustion-top shorts
(NVO/SNPS/...) print modest up-day magnitudes even at their own tops, whereas
rippers print large ones, and the two cohorts do NOT overlap on this axis.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bt.strategies.mfi_exhaustion_dsl import _meltup_upday_mag


def _daily_close_series(
    seed: int, n: int, up_size: float, dn_size: float
) -> np.ndarray:
    """Deterministic OHLC series whose up-days average ~``up_size`` % and
    down-days ~``dn_size`` % (of the prior close)."""
    rng = np.random.default_rng(seed)
    ups = up_size / 100.0
    dns = dn_size / 100.0
    signs = np.where(rng.uniform(size=n - 1) < 0.5, ups, -dns)
    close = 100.0 * np.cumprod(1.0 + signs)
    return np.concatenate([[100.0], close[: n - 1]])


def _avg_upday(close: np.ndarray) -> float:
    rets = np.diff(close) / close[:-1]
    ups = rets[rets > 0]
    return float(np.mean(ups) * 100.0) if ups.size else 0.0


def test_single_upday_matches_manual_mean():
    # current bar is excluded; the preceding 4 closed bars are 100->105->100->106
    arr = np.array([100.0, 105.0, 100.0, 106.0, 106.0], dtype=float)
    # up-days: +5%, +6% => mean 5.5 ; the -5% down-day is ignored
    assert _meltup_upday_mag(arr, 4) == pytest.approx(5.5)


def test_matches_full_series_average():
    for seed in (0, 1, 2, 7):
        closes = _daily_close_series(seed, n=500, up_size=1.5, dn_size=1.2)
        expected = _avg_upday(closes)
        # tail window of 400 closed bars should approximate the series avg
        assert _meltup_upday_mag(closes, 400) == pytest.approx(expected, abs=0.35)


def test_ripper_separates_from_rollover():
    """A ripper cohort (large up-day ~2.8%) must read >= a 2.2% veto, a rollover
    cohort (~1.6%) below it -- the core generalization the gate depends on."""
    ripper = _daily_close_series(seed=3, n=500, up_size=2.8, dn_size=2.3)
    rollover = _daily_close_series(seed=4, n=500, up_size=1.6, dn_size=1.6)
    print(
        "ripper",
        _meltup_upday_mag(ripper, 200),
        "rollover",
        _meltup_upday_mag(rollover, 200),
    )
    assert _meltup_upday_mag(ripper, 200) >= 2.2
    assert _meltup_upday_mag(rollover, 200) < 2.2


def test_zero_when_insufficient_history():
    arr = np.array([100.0, 101.0], dtype=float)
    assert _meltup_upday_mag(arr, 200) == 0.0
    assert _meltup_upday_mag(np.array([100.0], dtype=float), 5) == 0.0


def test_zero_when_no_up_days():
    arr = np.array([100.0, 99.0, 98.0, 97.5], dtype=float)
    assert _meltup_upday_mag(arr, 3) == 0.0
