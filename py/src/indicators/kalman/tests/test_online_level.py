"""Tests for the online single-symbol constant-velocity Kalman (OnlineLevel).

``OnlineLevel`` steps a [log price, drift] Kalman one observation at a time and
exposes ``z_stat`` = standardized one-step-ahead residual — a fully **online**
mean-reversion signal (state-only, no rolling window). Three properties matter:

1. it must match the batch ``run_filter`` numerically (one series, two paths);
2. ``z_stat`` must be **scale-free** (B3): |z| of order one on any price level
   and any volatility, so ``z_entry`` is a real sigma threshold. The old absolute
   price² ``measurement_noise`` gave sd 75-245 on real 1d bars, and ~18 on a
   synthetic random walk — this test fails against that code;
3. it must still fire on genuine overextension and stay quiet otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.kalman.strategy import OnlineLevel
from src.indicators.kalman.pure import run_filter
from src.indicators.kalman.types import KalmanConfig

# The shipped strategy settings (kalman_mr_regime_dsl's strategy_params).
Q, R = 1e-4, 1e-3
WARMUP = 150


def _replay(prices: list[float], *, warmup: int = WARMUP, **kw) -> list[float]:
    """Feed prices bar-by-bar to an OnlineLevel; return per-bar z_stat (0 pre-warm)."""
    f = OnlineLevel(warmup_bars=warmup, **kw)
    out: list[float] = []
    for p in prices:
        r = f.observe(p)
        out.append(r.z_stat if r.z_stat is not None else 0.0)
    return out


def _random_walk(n: int = 1500, vol: float = 0.01, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.exp(np.cumsum(rng.standard_normal(n) * vol))


def test_matches_batch_run_filter():
    """The online and batch paths must produce the same residual series.

    Batch's one-step-ahead residual is ``log P - predicted log P``; online
    reports exactly that at every bar, so the two must agree to float noise
    (they only differ in the covariance update form: Joseph vs (I-KH)P).
    """
    price = _random_walk(500, vol=0.015)
    s = pd.Series(price)

    bres = run_filter(
        s, KalmanConfig(process_noise=Q, measurement_noise=R, adaptive=False)
    )

    f = OnlineLevel(process_noise=Q, measurement_noise=R, warmup_bars=WARMUP)
    online_resid = []
    for p in price:
        r = f.observe(float(p))
        assert r.residual is not None
        online_resid.append(r.residual)

    np.testing.assert_allclose(
        np.array(online_resid), bres.residuals.values, rtol=0, atol=1e-9
    )


def test_z_stat_is_scale_free_on_a_random_walk():
    """|z_stat| must be O(1) — and price-level invariant (B3).

    Defect: ``measurement_noise`` was consumed as an absolute price² variance,
    so the innovation sigma was ~$0.05 on a $150 asset while the residuals were
    dollars. The standardized residual then had sd ~18 on this synthetic walk
    (75-245 on real 1d bars) and any ``z_entry=2`` gate passed on ~90% of bars.
    The filter is now log-domain and self-standardized, so:
      * ``|z|`` has sd well inside [0.5, 5] (a few, not tens);
      * the *same* log path at a different price level gives the *same* z.
    """
    log_path = np.cumsum(np.random.default_rng(11).standard_normal(1500) * 0.01)

    sds = {}
    for level in (5.0, 150.0):  # ~$148 vs ~$3.3e65 — absurd gap on purpose
        z = np.abs(np.array(_replay(list(np.exp(log_path + level)))[WARMUP:]))
        sds[level] = float(z.std())
        assert 0.5 < sds[level] < 5.0, f"|z| sd at level {level}: {sds[level]}"

    # Scale-free: the price level must not move the signal's scale at all.
    assert abs(sds[5.0] - sds[150.0]) < 0.05


def test_fires_on_sustained_overshoot():
    """A sustained jump above the fitted level produces a strong +z_stat."""
    n = 300
    rng = np.random.default_rng(3)
    base = 100 + rng.standard_normal(n).cumsum() * 0.08
    price = list(base)
    # Last 30 bars jump +1.5% and hold — a genuine overextension the level
    # filter (low q) cannot immediately absorb.
    for i in range(max(0, n - 30), n):
        price[i] = price[i - 1] * 1.015
    z = _replay(price, process_noise=1e-5, measurement_noise=1e-2)
    # The ramp is the last 30 bars; z spikes at its first bar then decays as the
    # self-standardising scale catches up.
    assert max(z[-35:]) > 3.0, f"no positive overshoot z: {z[-35:]}"


def test_quiet_on_a_pure_random_walk():
    """A pure random walk is unpredictable: z must stay centred and bounded.

    ``z_stat`` is self-standardized, so ~5% of bars sit beyond ±2 by
    construction; what must not happen is a drifting, runaway scale.
    """
    z = _replay(list(_random_walk(300, vol=0.006, seed=5)))
    tail = np.array(z[-100:])
    assert abs(float(tail.mean())) < 1.0, f"biased z: {tail.mean()}"
    assert float(np.abs(tail).max()) < 5.0, f"runaway z: {tail}"


def test_deterministic():
    price = list(_random_walk(300, vol=0.01, seed=99))
    assert _replay(price, process_noise=Q) == _replay(price, process_noise=Q)
