"""Warmup boundary of ``MarketRegimeHMM.fit``.

Goal: pin the exact price length the model needs. ``_create_features`` drops
``max(vol_window, momentum_window)`` rows to the rolling warmup, so ``fit``
requires ``min_train_size`` *features* — i.e. ``min_train_size +
max(vol_window, momentum_window)`` prices, not ``min_train_size``. The regime
detector's guard (``src/bt/regime/detectors.py``) must respect that band; this
test is the contract it is written against.

Method: fit at the boundary (must succeed) and one bar below it (must raise
ValueError), on a deterministic synthetic walk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.hmm import MarketRegimeHMM

VOL_WINDOW = 20
MOMENTUM_WINDOW = 10
MIN_TRAIN_SIZE = 60


def _prices(n: int) -> pd.Series:
    rng = np.random.default_rng(4)
    walk = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    return pd.Series(walk, index=pd.date_range("2024-01-01", periods=n, freq="D"))


def test_fit_needs_min_train_size_plus_feature_warmup():
    boundary = MIN_TRAIN_SIZE + max(VOL_WINDOW, MOMENTUM_WINDOW)
    prices = _prices(boundary)

    model = MarketRegimeHMM(
        n_regimes=3,
        vol_window=VOL_WINDOW,
        momentum_window=MOMENTUM_WINDOW,
        min_train_size=MIN_TRAIN_SIZE,
        random_state=42,
    ).fit(prices)
    assert model.fitted

    with pytest.raises(ValueError):
        MarketRegimeHMM(
            n_regimes=3,
            vol_window=VOL_WINDOW,
            momentum_window=MOMENTUM_WINDOW,
            min_train_size=MIN_TRAIN_SIZE,
            random_state=42,
        ).fit(prices.iloc[:-1])
