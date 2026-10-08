"""Tests for pairs Kalman filter — critical paths only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.kalman.pure import run_pairs_kalman, _warm_start_ols
from src.indicators.kalman.online import PairsKalmanOnline
# ── _warm_start_ols ─────────────────────────────────────────────


def test_ols_recovers_parameters():
    log_p2 = np.linspace(3.0, 5.0, 100)
    log_p1 = 0.5 + 0.8 * log_p2 + np.random.default_rng(99).normal(0, 0.01, 100)
    alpha, beta, P = _warm_start_ols(log_p1, log_p2, window=50)
    assert abs(alpha - 0.5) < 0.1
    assert abs(beta - 0.8) < 0.1
    assert P.shape == (2, 2)


# ── Batch ───────────────────────────────────────────────────────


def test_t_stat_spread_sign_agreement():
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    p2 = pd.Series(100 + rng.standard_normal(200).cumsum() * 0.5, index=idx)
    p1 = 0.8 * p2 + rng.normal(0, 0.5, 200)
    result = run_pairs_kalman(p1, p2)
    tail_spread = result.spread.iloc[30:]
    tail_t = result.t_stat.iloc[30:]
    assert (tail_spread * tail_t > 0).mean() > 0.99


def test_deterministic():
    rng = np.random.default_rng(99)
    idx = pd.date_range("2024-01-01", periods=50, freq="D")
    p1 = pd.Series(100 + rng.standard_normal(50).cumsum(), index=idx)
    p2 = pd.Series(100 + rng.standard_normal(50).cumsum(), index=idx)
    a = run_pairs_kalman(p1, p2)
    b = run_pairs_kalman(p1, p2)
    assert a.t_stat.equals(b.t_stat)
    assert a.beta.equals(b.beta)


# ── Online ──────────────────────────────────────────────────────


def test_online_no_nan():
    rng = np.random.default_rng(7)
    kf = PairsKalmanOnline()
    p2 = 100 + rng.standard_normal(100).cumsum() * 0.3
    p1 = 0.8 * p2 + rng.standard_normal(100) * 1.0
    kf.init(np.log(p1[0]), np.log(p2[0]))
    for i in range(1, len(p1)):
        t = kf.update(np.log(p1[i]), np.log(p2[i]))
        assert not np.isnan(t)
        assert not np.isnan(kf.t_stat)


def test_online_properties():
    kf = PairsKalmanOnline()
    kf.init(5.0, 4.5)
    for i in range(10):
        kf.update(5.0 + i * 0.01, 4.5 + i * 0.005)
    assert 0.5 < kf.beta < 1.5
    assert isinstance(kf.alpha, float)
    assert isinstance(kf.t_stat, float)
    assert kf.n_steps == 11
