"""Tests for the Chan (2013) Example 3.3 pairs-filter reproduction.

Every test here is about the RECURSION — its convergence, its standardisation, its
first-bar special case and its causality.  None of them is about P&L.

Generator convention: a synthetic pair is drawn EXACTLY from the reference's model
(items 3-4 of ``reference.md`` §2) — both states random-walk with ``Vw = delta/(1-delta)``
and the observation carries ``N(0, ve)`` noise — so the filter's assumed model and the
data-generating process are the same object.  Regressors are iid uniform rather than a
random walk: a random walk drifts out of price scale over the sample and gives the
intercept-regressor pair far worse conditioning, which muddies every measurement below.

Measured numbers quoted in docstrings are the values asserted on this exact design.
"""

from __future__ import annotations

import math
import random

import pytest

from src.indicators.kalman.chan_pairs import (
    DELTA,
    VE,
    ChanPairsConfig,
    ChanPairsFilter,
    ChanPairsState,
    step,
)


def _ols_slope(xs: list[float], ys: list[float]) -> float:
    """Full-sample OLS slope of y on x, computed from the normal equations."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def _model_series(
    n: int,
    delta: float,
    seed: int,
    beta: float = 1.7,
    alpha: float = 3.0,
    ve: float = VE,
    lo: float = 80.0,
    hi: float = 120.0,
    drift: bool = True,
) -> tuple[list[float], list[float], list[float]]:
    """Draw ``(x, y, beta_path)`` exactly from the reference's model (items 3-4).

    ``drift=False`` freezes the states on purpose: that is the ``Vw = 0`` instance of
    the same model, the one place where a *single* generating beta exists to converge to.
    """
    rng = random.Random(seed)
    sd = math.sqrt(delta / (1.0 - delta)) if drift else 0.0
    beta_t, alpha_t = beta, alpha
    xs, ys, beta_path = [], [], []
    for _ in range(n):
        x = rng.uniform(lo, hi)
        beta_t += rng.gauss(0.0, sd)
        alpha_t += rng.gauss(0.0, sd)
        xs.append(x)
        beta_path.append(beta_t)
        ys.append(beta_t * x + alpha_t + rng.gauss(0.0, math.sqrt(ve)))
    return xs, ys, beta_path


# ── 1. convergence to the generating beta ───────────────────────


def test_beta_converges_to_generating_beta() -> None:
    """The filter's beta must track the beta that generated the series.

    Model and filter share ``delta = 1e-4`` (items 3-4), so the only error left is
    estimation error.  Two targets: the *path* (the states drift, so the last true beta
    is the honest target) and the frozen ``Vw = 0`` instance, where beta is one number.

    Measured over 10 seeds x 5000 bars: mean relative tracking error 1.31e-3
    (max 2.04e-3) against the path, and 2.23e-4 (max 4.24e-4) against a frozen
    beta of 1.7.
    """
    path_err = []
    frozen_err = []
    for seed in range(10):
        xs, ys, beta_path = _model_series(5000, DELTA, seed)
        f = ChanPairsFilter()
        for x, y in zip(xs, ys):
            out = f.update(x, y)
        path_err.append(abs(out.beta - beta_path[-1]) / 1.7)

        xs, ys, _ = _model_series(5000, DELTA, seed, drift=False)
        f = ChanPairsFilter()
        for x, y in zip(xs, ys):
            out = f.update(x, y)
        frozen_err.append(abs(out.beta - 1.7) / 1.7)

    assert sum(path_err) / len(path_err) < 5e-3
    assert max(path_err) < 1e-2
    assert sum(frozen_err) / len(frozen_err) < 1e-3


# ── 2. the Vw -> 0 limit is an OLS limit ────────────────────────


def _ols_limit_rel_err(n: int, ve: float, delta: float = 1e-12, seed: int = 7) -> float:
    rng = random.Random(seed)
    xs = [rng.uniform(80.0, 120.0) for _ in range(n)]
    ys = [1.7 * x + 3.0 for x in xs]  # noise-free: OLS slope is exactly 1.7
    f = ChanPairsFilter(ChanPairsConfig(delta=delta, ve=ve))
    for x, y in zip(xs, ys):
        out = f.update(x, y)
    return abs(out.beta - _ols_slope(xs, ys)) / abs(_ols_slope(xs, ys))


def test_vw_to_zero_reduces_to_ols() -> None:
    """``delta = 1e-12`` must reproduce the OLS slope, i.e. recursive least squares.

    This is the assertion that fails on a reversed predict/update order or a
    transposed F: either one destroys the least-squares correspondence completely.

    The bar count is not free.  Item 5's ``P0 = 0`` is an infinitely strong prior at
    ``theta = 0``, so the filter only becomes OLS once the data have out-voted it —
    which takes ``~ t^2 * Vw * var(x) / Ve >> 1`` bars.  Measured at the reference's
    ``Ve = 1e-3``: |beta - OLS|/OLS = 1.4e-2 at both 2,000 and 200,000 bars, 2.4e-5
    only at 2,000,000.  A precise observation (``Ve = 1e-9``) releases the prior at
    once, so the limit is reached inside a testable sample: 5.9e-15 at 20,000 bars.
    """
    assert _ols_limit_rel_err(20_000, ve=1e-9) < 1e-6
    # Same recursion at the reference's Ve after 2,000 bars is NOT OLS yet: the
    # zero prior still holds beta ~1.5% high.  Pinned so neither direction can drift.
    assert _ols_limit_rel_err(2_000, ve=VE) == pytest.approx(1.375e-2, rel=0.05)


# ── 3. e and S are correctly standardised ───────────────────────


def test_z_is_standard_normal_on_white_noise() -> None:
    """``z = e/S`` must have the normal tail rates when the model is right.

    Data are drawn from the model with the filter's own ``delta``, so the standardised
    innovations are exactly standard normal by construction — any error in ``S``
    (wrong F R F' term, missing Ve, R instead of R+Vw) breaks the rates.

    Measured over 10 seeds x 5000 bars (first 1000 bars of each discarded: the
    ``P0 = 0`` start is not yet consistent): |z| > 1 for 0.3219 of bars and
    |z| > 2 for 0.0456, with sd(z) = 1.007.
    """
    zs: list[float] = []
    for seed in range(10):
        xs, ys, _ = _model_series(5000, DELTA, seed)
        f = ChanPairsFilter()
        for i, (x, y) in enumerate(zip(xs, ys)):
            out = f.update(x, y)
            if i >= 1000:
                zs.append(out.z)

    n = len(zs)
    frac1 = sum(abs(z) > 1.0 for z in zs) / n
    frac2 = sum(abs(z) > 2.0 for z in zs) / n
    mean = sum(zs) / n
    sd = math.sqrt(sum((z - mean) ** 2 for z in zs) / n)

    assert abs(frac1 - 0.317) < 0.03, f"|z|>1 fraction {frac1:.4f}"
    assert abs(frac2 - 0.046) < 0.02, f"|z|>2 fraction {frac2:.4f}"
    assert abs(sd - 1.0) < 0.1, f"sd(z) {sd:.4f}"


# ── 4. the first-bar R = 0 special case ─────────────────────────


def test_first_bar_zero_prior_and_second_bar_R() -> None:
    """Item 5: bar 1 has ``R = 0`` (not ``Vw``), and every later bar has ``R = P + Vw``.

    Bar 1: ``R = 0`` makes ``K = 0``, so theta stays exactly ``(0, 0)`` and ``P`` stays
    exactly zero; the innovation is therefore ``y1 - 0`` and ``S1 = sqrt(Ve)``
    (measured 0.031623).  A filter that used ``Vw`` on bar 1 instead would move beta
    off zero on the very first bar (beta1 = 1.0007, S1 = 1.00505 on this input, a 32x
    wider band): the exact zeros below are what rules that out.

    Bar 3 pins the *recurring* rule, ``R = P + Vw``.  ``P`` after bar 2 is non-zero
    (beta 9.08e-8, alpha 1.0000e-4), so ``S3 = sqrt(F3 P2 F3' + F3 Vw F3' + Ve)`` =
    0.105792, against 0.105361 for a filter that re-added ``Vw`` alone: a small but
    exact-to-2e-11 discriminator (P has already collapsed in the beta direction).
    """
    vw = DELTA / (1.0 - DELTA)
    x1, y1 = 100.0, 101.0
    x2, y2 = 110.0, 112.0
    x3, y3 = 10.0, 12.0

    s0 = ChanPairsState()
    s1, o1 = step(s0, ChanPairsConfig(), x1, y1)
    assert o1.innovation == y1  # theta_0 = (0, 0), so e_1 = y_1 - 0 exactly
    assert o1.beta == 0.0
    assert o1.alpha == 0.0
    assert o1.s == math.sqrt(VE)
    assert s1.p == ((0.0, 0.0), (0.0, 0.0))  # R = 0 => K = 0 => P stays zero
    assert o1.ready is False  # item 6: "trade if days > 1"

    s2, o2 = step(s1, ChanPairsConfig(), x2, y2)
    q2 = vw * (x2 * x2 + 1.0) + VE  # R_2 = P_1 + Vw = Vw, because P_1 = 0
    assert o2.s == pytest.approx(math.sqrt(q2), rel=1e-12)
    assert o2.innovation == y2  # theta_1 is still (0, 0)
    assert o2.ready is True
    assert s2.p_beta_beta > 0.0

    s3, o3 = step(s2, ChanPairsConfig(), x3, y3)
    q3 = (
        (s2.p_beta_beta + vw) * x3 * x3
        + 2.0 * s2.p_beta_alpha * x3
        + (s2.p_alpha_alpha + vw)
        + VE
    )
    q3_vw_only = vw * (x3 * x3 + 1.0) + VE
    assert o3.s == pytest.approx(math.sqrt(q3), rel=1e-12)
    assert abs(o3.s / math.sqrt(q3_vw_only) - 1.0) > 1e-3


# ── 5. causality ────────────────────────────────────────────────


def test_prefix_filtering_matches_full_series() -> None:
    """Filtering a prefix must give the same ``theta_t`` as filtering the full series.

    No future bar may reach an earlier state: the recursion is one-way, so a filter
    fed bars 1..t sees exactly the state the full run had at t.
    """
    xs, ys, _ = _model_series(200, DELTA, seed=3)
    full = ChanPairsFilter()
    states: list[ChanPairsState] = []
    for x, y in zip(xs, ys):
        full.update(x, y)
        states.append(full.state)

    for t in (1, 2, 7, 50, 137, 200):
        prefix = ChanPairsFilter()
        for x, y in zip(xs[:t], ys[:t]):
            prefix.update(x, y)
        assert prefix.state == states[t - 1]
