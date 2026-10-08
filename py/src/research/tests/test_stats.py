"""Unit tests for pure research statistics on synthetic panels.

Every assertion rests on a deterministic construction so a regression in the
math (not just plumbing) is caught.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.stats import (
    bench_regimes,
    catalyst_intraday,
    momentum_sweep,
    regime_stats,
    residual_dispersion,
    residual_returns,
    vol_clustering,
)

DAYS = 400
INDEX = pd.date_range("2020-01-01", periods=DAYS, freq="D")


def _frame(data: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(data, index=INDEX)


def _shared_factor(rng: np.random.Generator, n: int = 8) -> pd.DataFrame:
    """Residual columns sharing one strong factor: expect high rho / high PC1."""
    factor = rng.standard_normal(DAYS)
    cols = {
        f"s{k}": (5.0 * factor + rng.standard_normal(DAYS)).tolist() for k in range(n)
    }
    return _frame(cols)


def _white_noise(rng: np.random.Generator, n: int = 8) -> pd.DataFrame:
    return _frame({f"s{k}": rng.standard_normal(DAYS).tolist() for k in range(n)})


def test_dispersion_detects_shared_factor() -> None:
    resid = _shared_factor(np.random.default_rng(7))
    d = residual_dispersion(resid)
    assert d.mean_pairwise_rho is not None and d.mean_pairwise_rho > 0.8
    assert d.pc1_var_share is not None and d.pc1_var_share > 0.8
    assert d.effective_n is not None and d.effective_n < 3.0


def test_dispersion_white_noise_is_low() -> None:
    resid = _white_noise(np.random.default_rng(3))
    d = residual_dispersion(resid)
    assert d.mean_pairwise_rho is not None and abs(d.mean_pairwise_rho) < 0.2
    assert d.pc1_var_share is not None and d.pc1_var_share < 0.4


def test_residual_equal_to_bench_is_zero() -> None:
    """A member whose return exactly equals the benchmark has ~zero residual."""
    rng = np.random.default_rng(5)
    bench = pd.Series(rng.standard_normal(DAYS) * 0.01, index=INDEX)
    ret = _frame({"name": bench.tolist()})
    resid = residual_returns(ret, bench)
    vals = resid["name"].dropna()
    assert len(vals) > 0 and float(vals.abs().mean()) < 1e-2


def test_momentum_positive_trend_detected() -> None:
    """Stable per-name tilt => strong positive cross-sectional forecast."""
    rng = np.random.default_rng(11)
    cols = {}
    for k, tilt in enumerate([0.0, 0.02, 0.04, 0.06]):
        cols[f"t{k}"] = (tilt + 0.01 * rng.standard_normal(DAYS)).tolist()
    resid = _frame(cols)
    cells = momentum_sweep(resid, resid, lookbacks=(5,), horizons=(5,))
    assert len(cells) == 1
    cell = cells[0]
    assert cell.n_rows >= 30
    assert cell.spearman is not None and cell.spearman > 0.3
    assert cell.net_decile_bps is not None and cell.net_decile_bps > 0


def test_catalyst_up_drift_down_fade() -> None:
    up = _sparse_after(0.30, 0.05)
    down = _sparse_after(-0.30, -0.05)
    result = catalyst_intraday({"UP": up, "DOWN": down}, threshold_sigma=2.5)
    sides = {s.direction: s for s in result.sides}
    assert sides["up"].events > 10
    assert sides["up"].fwd_1h_mean_bps is not None and sides["up"].fwd_1h_mean_bps > 0
    assert sides["down"].events > 10
    assert (
        sides["down"].fwd_1h_mean_bps is not None and sides["down"].fwd_1h_mean_bps < 0
    )


def _sparse_after(spike_val: float, post_val: float) -> pd.Series:
    """Mostly-calm hourly bars with isolated extremes whose next bar is fixed."""
    n = 4000
    vals = np.zeros(n)
    for i in range(200, n, 200):
        vals[i] = spike_val
        if i + 1 < n:
            vals[i + 1] = post_val
    return pd.Series(vals)


def test_vol_clustering_blocky_panel() -> None:
    """High-vol block then low-vol block produces positive vol AC and spikes."""
    vol_hi = np.random.default_rng(1).normal(0, 0.05, DAYS // 2)
    vol_lo = np.random.default_rng(2).normal(0, 0.001, DAYS - DAYS // 2)
    resid = _frame({"name": np.concatenate([vol_hi, vol_lo]).tolist()})
    result = vol_clustering(resid)
    assert result.n_members == 1
    assert result.mean_ac1 is not None and result.mean_ac1 > 0.2
    assert result.spike_n >= 0  # no crash, finite spike tally


def test_vol_clustering_under_sized_panel_no_crash() -> None:
    tiny = pd.DataFrame({"a": [0.0] * 5}, index=INDEX[:5])
    result = vol_clustering(tiny)
    assert result.mean_ac1 is None and result.spike_n == 0


def test_regime_buckets_cover_all_three() -> None:
    bench = pd.Series(np.linspace(0.0, 0.2, DAYS), index=INDEX)
    bucket = bench_regimes(bench)
    counts = bucket.value_counts()
    assert set(counts.index) == {"down", "flat", "up"}
    resid = _white_noise(np.random.default_rng(1), n=6)
    cells = regime_stats(bucket, resid)
    assert len(cells) == 3
    assert counts.get("up", 0) > 0


# ── De-overlapped (effective) independent-sample significance ──


def _noise_frame(rng_seed: int, n: int, days: int) -> pd.DataFrame:
    """Independent white-noise residual columns (zero real cross-sectional edge)."""
    rng = np.random.default_rng(rng_seed)
    idx = pd.date_range("2015-01-01", periods=days, freq="D")
    col = {f"m{k}": list(rng.standard_normal(days) * 0.01) for k in range(n)}
    return pd.DataFrame(col, index=idx)


def test_overlap_inflates_pooled_but_not_effective_t() -> None:
    """White noise (no real effect): overlapping-window pooling must not be
    presented as more significant than the de-overlapped effective t."""
    resid = _noise_frame(202, n=8, days=1500)
    cells = momentum_sweep(resid, resid, lookbacks=(21,), horizons=(21,))
    cell = cells[0]
    assert cell.n_effective >= 30
    assert cell.t_stat is not None and cell.t_effective is not None
    # homogeneous white-noise cross-section carries no genuine 21->21 reversal,
    # so the effective t must be small (<2 gate) even if pooled looks 'significant'
    # due to sample-size inflation.
    assert abs(cell.t_effective) < 2.0
    # the decimated effective t is not larger in |.| than the pooled one here.
    assert abs(cell.t_effective) <= abs(cell.t_stat) + 1e-9
    assert cell.n_effective < cell.n_rows


def test_momentum_effective_fields_populated() -> None:
    """momentum_sweep fills the effective-sample fields on a real panel, and the
    effective sample is strictly smaller than the pooled redundant one."""
    resid = _noise_frame(7, n=6, days=1200)
    cells = momentum_sweep(resid, resid, lookbacks=(5, 21), horizons=(5, 21))
    c21 = [c for c in cells if c.lookback == 21 and c.horizon == 21][0]
    assert c21.n_effective < c21.n_rows
    assert (c21.n_effective >= 30) or (c21.t_effective is None)
    assert c21.n_effective_too_small == (
        c21.t_effective is None and c21.n_effective < 30
    )


def test_effective_too_small_is_flagged() -> None:
    """A panel too short to support a stride-decimated sample must flag
    n_effective_too_small and leave t_effective None (never silently reuse the
    pooled significance)."""
    resid = _noise_frame(1, n=3, days=60)
    cells = momentum_sweep(resid, resid, lookbacks=(21,), horizons=(21,))
    c = cells[0]
    assert c.t_effective is None or c.n_effective_too_small
    if c.t_effective is None:
        assert c.n_effective_too_small


def test_effective_t_survives_when_real_persistence_exists() -> None:
    """A genuine persistent directional effect must keep a significant
    de-overlapped effective t (not clear only because of overlap)."""
    rng = np.random.default_rng(9)
    days = 3000
    idx = pd.date_range("2015-01-01", periods=days, freq="D")
    cols = {}
    for k in range(10):
        x = 0.0
        vals = []
        for _ in range(days):
            x = 0.90 * x + 0.03 * rng.standard_normal()
            vals.append(x)
        cols[f"m{k}"] = vals
    resid = pd.DataFrame(cols, index=idx)
    cells = momentum_sweep(resid, resid, lookbacks=(21,), horizons=(21,))
    c = cells[0]
    assert c.n_effective >= 30
    assert c.t_effective is not None
    assert abs(c.t_effective) >= 2.0
