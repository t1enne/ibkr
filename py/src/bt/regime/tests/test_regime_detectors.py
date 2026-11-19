"""Tests for regime detectors — critical detectors + mapping."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.bt.regime.detectors import (
    create_hmm_vol_detector,
    create_sma_detector,
    create_volatility_detector,
    current_trend_label,
)
from src.bt.regime.types import TREND_INT_TO_LABEL, TREND_LABEL_TO_INT


def _make_price_series(
    n_bars: int, drift: float = 0.0, vol: float = 0.01, seed: int = 42
) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = drift / n_bars + vol * rng.standard_normal(n_bars)
    return pd.Series(100.0 * np.exp(np.cumsum(returns)))


def _make_bull(n: int = 500) -> pd.DataFrame:
    prices = _make_price_series(n, drift=0.25, vol=0.10, seed=3)
    return pd.DataFrame(
        {
            "close": prices,
            "open": prices * 0.999,
            "high": prices * 1.002,
            "low": prices * 0.998,
            "volume": 1_000_000,
        }
    )


def _make_bear(n: int = 500) -> pd.DataFrame:
    prices = _make_price_series(n, drift=-0.30, vol=0.22, seed=2)
    return pd.DataFrame(
        {
            "close": prices,
            "open": prices * 1.001,
            "high": prices * 1.003,
            "low": prices * 0.997,
            "volume": 1_000_000,
        }
    )


def _make_range(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    theta, mu, sigma = 0.05, 100.0, 0.4
    prices = np.zeros(n)
    prices[0] = mu
    for t in range(1, n):
        prices[t] = (
            prices[t - 1] + theta * (mu - prices[t - 1]) + sigma * rng.standard_normal()
        )
    s = pd.Series(prices)
    return pd.DataFrame(
        {
            "close": s,
            "open": s * 0.999,
            "high": s * 1.002,
            "low": s * 0.998,
            "volume": 500_000,
        }
    )


# ── SMA detector ────────────────────────────────────────────────


def test_sma_bull_dominant():
    detect = create_sma_detector(fast_window=20, slow_window=50)
    regimes = detect(_make_bull())
    valid = regimes[regimes >= 0]
    assert (valid == 1).mean() > 0.3  # BULL dominant


def test_sma_bear_dominant():
    detect = create_sma_detector(fast_window=20, slow_window=50)
    regimes = detect(_make_bear())
    valid = regimes[regimes >= 0]
    assert (valid == 2).mean() > 0.55  # BEAR dominant


def test_sma_range_detected():
    detect = create_sma_detector(
        fast_window=10, slow_window=30, range_threshold_pct=0.003
    )
    regimes = detect(_make_range())
    valid = regimes[regimes >= 0]
    assert (valid == 0).mean() > 0.3  # RANGE detected


def test_sma_insufficient_warmup():
    short = pd.DataFrame({"close": [100.0] * 10})
    regimes = create_sma_detector(slow_window=50)(short)
    assert (regimes == -1).all()


# ── Volatility detector ─────────────────────────────────────────


def test_vol_bull_dominant():
    regimes = create_volatility_detector(vol_window=20, direction_window=50)(
        _make_bull()
    )
    valid = regimes[regimes >= 0]
    assert (valid == 1).mean() > 0.4


def test_vol_bear_dominant():
    detect = create_volatility_detector(
        vol_window=20, direction_window=50, low_vol_pctile=0.25, high_vol_pctile=0.75
    )
    regimes = detect(_make_bear())
    valid = regimes[regimes >= 0]
    assert (valid == 2).mean() > 0.18


# ── HMM detector ────────────────────────────────────────────────


def test_hmm_detects_multiple_regimes():
    bull = _make_bull(300)
    bear = _make_bear(300)
    rng_data = _make_range(300)
    base = pd.Timestamp("2024-01-01")
    bull.index = pd.date_range(base, periods=300, freq="D")
    bear.index = pd.date_range(base + pd.Timedelta(days=300), periods=300, freq="D")
    rng_data.index = pd.date_range(base + pd.Timedelta(days=600), periods=300, freq="D")
    multi = pd.concat([bull, bear, rng_data])
    regimes = create_hmm_vol_detector(min_train_size=250)(multi)
    valid = regimes[regimes >= 0]
    assert len(set(valid.unique())) >= 2


def test_hmm_too_short():
    short = pd.DataFrame({"close": [100.0] * 50})
    regimes = create_hmm_vol_detector(min_train_size=252)(short)
    assert (regimes == -1).all()


def test_hmm_warmup_boundary_accounts_for_feature_warmup():
    """The guard must cover the feature warmup, not just ``min_train_size``.

    ``_create_features`` drops ``max(vol_window, momentum_window)`` rows, so the
    band ``[min_train_size, min_train_size + feature_warmup)`` used to pass the
    guard and then raise ValueError inside ``fit``. One bar below the boundary
    must return -1; at and above it the fit must run.
    """
    feature_warmup = 20  # max(vol_window=20, momentum_window=10)
    boundary = 252 + feature_warmup

    below = _make_bull(boundary - 1)
    assert (create_hmm_vol_detector(min_train_size=252)(below) == -1).all()

    # At the boundary the model fits; every bar is still inside the
    # min_train_size label mask, so the output is unlabelled but not an error.
    at = _make_bull(boundary)
    assert (create_hmm_vol_detector(min_train_size=252)(at) == -1).all()

    # One bar past the mask, labels start appearing.
    past = _make_bull(boundary + 1)
    assert (create_hmm_vol_detector(min_train_size=252)(past) >= 0).any()


# ── Mapping consistency ─────────────────────────────────────────


def test_label_mapping():
    assert set(TREND_INT_TO_LABEL.keys()) == {0, 1, 2}
    assert set(TREND_LABEL_TO_INT.keys()) == {"BULL", "BEAR", "RANGE"}
    for i, label in TREND_INT_TO_LABEL.items():
        assert TREND_LABEL_TO_INT[label] == i


def test_current_trend_label_none_on_empty():
    assert current_trend_label(pd.Series([], dtype=float)) is None
