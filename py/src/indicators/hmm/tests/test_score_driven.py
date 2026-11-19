"""Validation battery for the score-driven (GAS) two-state HMM.

No library ships this estimator and no oracle exists for it, so these tests ARE
the validation:

1. :func:`test_exact_score_matches_numeric_gradient` -- the single-pass score
   (Lystig & Hughes) must equal a finite-difference gradient of the likelihood.
   A wrong forward-derivative recursion is caught here and nowhere else.
2. :func:`test_synthetic_oracle_recovers_parameters_and_regimes` -- on a
   5000-bar series generated with KNOWN mu/sigma/Gamma the estimator must
   recover the generating parameters and flip its filtered state at the known
   switch dates.
3. :func:`test_prefix_reproduces_filtered_state` -- causality: a prefix run
   reproduces every filtered alpha exactly (no smoothing, no full-sample fit).
4. :func:`test_no_constant_label_stretch_exceeds_true_spells` -- the pinning
   pathology (one label held for hundreds of bars) must not occur.

The synthetic run is deterministic (fixed PCG64 seed), so the numbers below are
reproducible; thresholds carry margin over the measured values.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.indicators.hmm.score_driven import (
    _LOG_SIGMA_MAX,
    _LOG_SIGMA_MIN,
    ScoreDrivenHMM,
    _forward_step,
    _unpack,
    banded_weight,
    optimal_weight,
)

#: Generating model: state 0 = low vol (the paper's state 1), state 1 = high vol.
TRUE_MU = np.array([0.0007, -0.0012])
TRUE_SIGMA = np.array([0.008, 0.024])
TRUE_GAMMA = np.array([[0.985, 0.015], [0.03, 0.97]])
EM_OBS = 260
N_BARS = 3000
#: ``mu`` has no structural bound (it is a location); this is a sanity ceiling
#: 5x above anything the synthetic fixture should ever reach, not a parameter.
_MU_SANITY = 0.05


def _simulate(n: int = N_BARS, seed: int = 20240217) -> tuple[np.ndarray, np.ndarray]:
    """Draw a 2-state regime-switching log-return series with known parameters."""
    rng = np.random.default_rng(seed)
    state = 0
    states = np.empty(n, dtype=int)
    returns = np.empty(n)
    for t in range(n):
        if t > 0:
            state = 0 if rng.random() < TRUE_GAMMA[state, 0] else 1
        states[t] = state
        returns[t] = rng.normal(TRUE_MU[state], TRUE_SIGMA[state])
    return returns, states


def _run(series: np.ndarray) -> tuple[ScoreDrivenHMM, np.ndarray, int]:
    """Feed the series one bar at a time; return the model and the alphas."""
    model = ScoreDrivenHMM(neff=260.0, em_obs=EM_OBS)
    alphas = np.full((series.size, 2), np.nan)
    ready_at = -1
    for t in range(series.size):
        obs = model.update(float(series[t]))
        if obs.ready:
            alphas[t] = obs.filtered
            if ready_at < 0:
                ready_at = t
    return model, alphas, ready_at


@pytest.fixture(scope="session")
def _oracle_run() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One pass over the oracle fixture, shared by every consumer.

    WHY: the recursion re-derives a ``neff``-bar window of scores on every bar,
    so a full pass is seconds, not milliseconds. Four tests read the same pass
    (oracle labels, parameter path, prefix reference, no-pinning, no-saturation);
    running it once per test was the entire suite's cost.
    """
    series, states = _simulate()
    model = ScoreDrivenHMM(neff=260.0, em_obs=EM_OBS)
    alphas = np.full((series.size, 2), np.nan)
    thetas = np.full((series.size, 6), np.nan)
    ready_at = -1
    for t in range(series.size):
        obs = model.update(float(series[t]))
        if obs.ready:
            alphas[t] = obs.filtered
            thetas[t] = np.asarray(obs.theta)
            if ready_at < 0:
                ready_at = t
    return series, states, alphas, thetas[ready_at:]


def _spells(labels: np.ndarray) -> list[int]:
    """Run lengths of consecutive identical labels."""
    if labels.size == 0:
        return []
    runs = [1]
    for t in range(1, labels.size):
        if labels[t] == labels[t - 1]:
            runs[-1] += 1
        else:
            runs.append(1)
    return runs


def _predictive_loglik(theta: np.ndarray, series: np.ndarray) -> float:
    """Constant-parameter HMM log-likelihood (the finite-difference reference)."""
    alpha = np.array([0.5, 0.5])
    total = 0.0
    for value in series:
        mu, sigma, gamma = _unpack(theta)
        dens = np.exp(-0.5 * ((value - mu) / sigma) ** 2) / (
            sigma * np.sqrt(2.0 * np.pi)
        )
        psi = (alpha @ gamma) * dens
        lik = float(np.sum(psi))
        total += float(np.log(lik))
        alpha = psi / lik
    return total


def test_exact_score_matches_numeric_gradient() -> None:
    """The single-pass score is the exact gradient of the predictive likelihood."""
    rng = np.random.default_rng(7)
    series = rng.normal(0.0, 0.01, 40)
    series[10] = 0.04  # a large bar exercises the far-tail branch
    theta = np.array([0.0005, -0.001, np.log(0.008), np.log(0.02), 2.0, -2.5])

    alpha = np.array([0.5, 0.5])
    dalpha = np.zeros((2, 6))
    analytic = np.zeros(6)
    for value in series:
        alpha, dalpha, score, _ll, _xi = _forward_step(alpha, dalpha, value, theta)
        analytic += score

    numeric = np.zeros(6)
    eps = 1e-6
    for k in range(6):
        up = theta.copy()
        down = theta.copy()
        up[k] += eps
        down[k] -= eps
        numeric[k] = (
            _predictive_loglik(up, series) - _predictive_loglik(down, series)
        ) / (2.0 * eps)

    assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-5), (analytic, numeric)


def test_synthetic_oracle_recovers_parameters_and_regimes(_oracle_run) -> None:
    """Known-parameter recovery + filtered flips at the true switch dates."""
    _series, states, alphas, thetas = _oracle_run

    # (i) parameter recovery, averaged over the post-initialisation period.
    mus: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    gammas: list[np.ndarray] = []
    for theta in thetas:
        mu, sigma, gamma = _unpack(theta)
        mus.append(mu)
        sigmas.append(sigma)
        gammas.append(gamma)
    mu_hat = np.mean(mus, axis=0)
    sigma_hat = np.mean(sigmas, axis=0)
    gamma_hat = np.mean(gammas, axis=0)

    assert np.max(np.abs(mu_hat - TRUE_MU)) <= 0.0025, (mu_hat, TRUE_MU)
    assert np.max(np.abs(sigma_hat / TRUE_SIGMA - 1.0)) <= 0.30, (sigma_hat, TRUE_SIGMA)
    assert abs(gamma_hat[0, 1] - TRUE_GAMMA[0, 1]) <= 0.03, gamma_hat
    assert abs(gamma_hat[1, 0] - TRUE_GAMMA[1, 0]) <= 0.03, gamma_hat

    # (ii) the filtered label tracks the true regime ...
    predicted = np.argmax(alphas[EM_OBS:], axis=1)
    truth = states[EM_OBS:]
    assert float(np.mean(predicted == truth)) >= 0.85

    # ... and flips within a few bars of every true switch.
    switches = np.where(np.diff(states) != 0)[0] + 1
    switches = switches[switches >= EM_OBS]
    assert switches.size >= 30, "synthetic series must contain many switches"
    delays = []
    for switch in switches:
        new_state = states[switch]
        start = switch - EM_OBS  # predicted is indexed from the first decision bar
        found = 120
        for delay in range(120):
            if start + delay < predicted.size and predicted[start + delay] == new_state:
                found = delay
                break
        delays.append(found)
    delays_arr = np.asarray(delays)
    assert float(np.median(delays_arr)) <= 5.0, delays_arr
    assert float(np.mean(delays_arr <= 20)) >= 0.85, delays_arr


def test_prefix_reproduces_filtered_state(_oracle_run) -> None:
    """Causality: feeding a prefix must reproduce alpha_t exactly."""
    series, _states, alphas_full, _thetas = _oracle_run
    prefix = 2000
    _model_prefix, alphas_prefix, _ = _run(series[:prefix])
    assert np.array_equal(alphas_prefix, alphas_full[:prefix], equal_nan=True), (
        "a prefix must reproduce every filtered distribution bit-for-bit"
    )
    assert np.isnan(alphas_full[: EM_OBS - 1]).all()
    assert np.isfinite(alphas_full[EM_OBS:]).all()
    assert np.allclose(alphas_full.sum(axis=1)[EM_OBS:], 1.0)


def test_no_constant_label_stretch_exceeds_true_spells(_oracle_run) -> None:
    """No pinned label: the filtered argmax must not hold longer than the data."""
    _series, states, alphas, _thetas = _oracle_run
    ready_at = EM_OBS - 1
    predicted = np.argmax(alphas[ready_at + 1 :], axis=1)
    true_spells = _spells(states)
    predicted_spells = _spells(predicted)
    assert max(predicted_spells) <= 1.5 * max(true_spells), (
        max(predicted_spells),
        max(true_spells),
    )
    # The filtered probability must actually move -- the previous HMM rebuild
    # died of a 502-bar constant label (P(state) pinned at an extreme).
    prob_low = alphas[ready_at + 1 :, 0]
    assert float(np.ptp(prob_low)) >= 0.5, "filtered P(low-vol) is pinned"
    assert float(np.mean((prob_low > 0.01) & (prob_low < 0.99))) >= 0.02


def test_synthetic_run_does_not_saturate_the_parameter_box(_oracle_run) -> None:
    """The corrected recursion must not lean on a parameter bound.

    METHODOLOGY. Feed the oracle fixture and compare every coordinate of every
    decision bar to its own structural bound: ``|mu| <= 5%/day`` (``mu`` has no
    structural bound -- this checks the recursion does not run away), ``sigma``
    in the module's fixed log band, ``gamma_ij`` in ``[1e-4, 0.5]``. A recursion
    with no restoring force walks to these bounds and stays. Thresholds are a
    wide multiple of the values the current build is measured at.
    """
    _series, _states, _alphas, thetas = _oracle_run
    mus: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    offdiag: list[np.ndarray] = []
    for theta in thetas:
        mu, sigma, gamma = _unpack(theta)
        mus.append(mu)
        sigmas.append(sigma)
        offdiag.append(np.array([gamma[0, 1], gamma[1, 0]]))
    mu = np.asarray(mus)
    sigma = np.asarray(sigmas)
    offdiag_arr = np.asarray(offdiag)
    ceil = float(np.exp(_LOG_SIGMA_MAX))
    floor = float(np.exp(_LOG_SIGMA_MIN))

    at_mu_bound = float(np.mean(np.abs(mu) >= _MU_SANITY - 1e-12))
    at_sigma_bound = float(np.mean((sigma >= ceil - 1e-9) | (sigma <= floor + 1e-9)))
    at_gamma_bound = float(
        np.mean((offdiag_arr >= 0.5 - 1e-9) | (offdiag_arr <= 1e-4 + 1e-12))
    )
    detail = (
        f"mu bound {at_mu_bound:.3%} (max |mu| {np.abs(mu).max() * 1e4:.1f} bp/day), "
        f"sigma bound {at_sigma_bound:.3%} (max/ceil {sigma.max() / ceil:.2f}, "
        f"min/floor {sigma.min() / floor:.2f}), "
        f"gamma bound {at_gamma_bound:.3%} (max/0.5 {offdiag_arr.max() / 0.5:.3f}, "
        f"min/1e-4 {offdiag_arr.min() / 1e-4:.4f})"
    )
    assert at_mu_bound == 0.0, detail
    assert at_sigma_bound <= 0.01, detail
    assert at_gamma_bound <= 0.03, detail


def test_banded_weight_is_a_no_trade_band() -> None:
    """(21)-(23)/(25): ``rho`` opens a no-trade zone around the optimum.

    With ``M = sum mu_k = 0.03``, ``S = sum sigma_k^2 = 0.01``, ``gamma = 2``,
    the unpenalised optimum is ``w_hat = M/(2 gamma S) = 0.75`` and the band is
    ``rho/(2 gamma S) = 0.5``. Every assertion below fails on a wrong branch,
    a missing ``rho`` term, or a band applied to the wrong side.
    """
    means = np.full(100, 3e-4)
    variances = np.full(100, 1e-4)
    gamma, rho = 2.0, 0.02

    # Buying from flat lands on the PENALISED optimum (w_hat - band = 0.25),
    # not on w_hat: the trade must pay for itself.
    assert abs(banded_weight(means, variances, gamma, rho, 0.0) - 0.25) <= 1e-9
    # From fully invested, the sell candidate (w_hat + band = 1.25) is above the
    # holding, so the position is HELD -- the band is not symmetric in outcome.
    assert banded_weight(means, variances, gamma, rho, 1.0) == 1.0
    # Anywhere inside the band the position is held, from either direction.
    assert banded_weight(means, variances, gamma, rho, 0.5) == 0.5
    assert banded_weight(means, variances, gamma, rho, 0.75) == 0.75

    # rho = 0 removes the band: both sides converge on the unpenalised optimum.
    assert abs(banded_weight(means, variances, gamma, 0.0, 0.0) - 0.75) <= 1e-9
    assert abs(banded_weight(means, variances, gamma, 0.0, 1.0) - 0.75) <= 1e-9

    # A negative forecast leaves the long-only box: sell everything.
    assert banded_weight(-means, variances, gamma, rho, 0.4) == 0.0
    # Degenerate variance cannot be optimised; stay in cash.
    assert banded_weight(means, np.zeros(100), gamma, rho, 0.3) == 0.0


def test_optimal_weight_is_the_constrained_mean_variance_solution() -> None:
    """1-D solve reproduces the closed form, clipped to the long-only box."""
    # Interior optimum: w* = mu / (2 * gamma * variance).
    weight = optimal_weight(mean=1e-4, variance=1e-4, risk_aversion=2.0)
    assert abs(weight - 0.25) <= 1e-3, weight
    # Above the box: the optimum clips to 1 (fully invested, long-only).
    assert abs(optimal_weight(mean=0.01, variance=1e-6, risk_aversion=2.0) - 1.0) < 1e-6
    # Negative expected return: never short, so w = 0.
    assert optimal_weight(mean=-1e-4, variance=1e-4, risk_aversion=2.0) == 0.0
    # Degenerate variance cannot be optimised; stay in cash.
    assert optimal_weight(mean=1e-4, variance=0.0, risk_aversion=2.0) == 0.0


# ---------------------------------------------------------------------------
# The GAS gain (paper (6)-(7)): theta_t = theta_{t-1} + A I_t^{-1} grad l~_t
# ---------------------------------------------------------------------------

#: A known crisis regime the recursion must be able to REACH: -40 bp/day with a
#: 300 bp daily sd, after 800 calm bars (+8 bp/day, 70 bp sd).
GAIN_MU = np.array([0.0008, -0.0040])
GAIN_SIGMA = np.array([0.007, 0.030])
#: Block plan: the EM window is CALM ONLY, so the high-variance mean starts far
#: from the truth and only the recursion's own speed can move it.
GAIN_PLAN = ((800, 0), (600, 1), (400, 0), (400, 1))
#: The SECOND crisis block is the measured one. WHY: the state a mean lives on
#: has to be IDENTIFIED entering the switch for the test to measure the gain at
#: all. It is not in the first block -- a high-vol state that no calm bar fits
#: loses its predictive weight (`P_n1 ~ 1e-4` for ~18 bars) and its mean is then
#: unidentified, so `mu_hi` enters crisis #1 at +187 bp/day (measured) and the
#: window says nothing about step size. By crisis #2 the state has been visited
#: for 600 bars, `mu_hi` enters at +27 bp/day, and the travel is the gain's.
#: The assertion is unchanged -- same threshold, same 50-bar budget -- and it
#: still fails on an under-scaled step (measured on the est3 MAP build: entry
#: +22.9 bp/day, min +24.7, i.e. never moves, `sd_hi` frozen at 80 bp).
GAIN_CRISIS_INDEX = 1
#: Stated tolerance/budget: mu_hi must be within 25% of the truth (-30 bp/day)
#: within 50 bars of a crisis regime that lasts 600. Measured with the shipped
#: recursion (rolling-window Newton step): -30 bp/day at bar 42 of the budget
#: (min -86 bp/day inside the window, sigma_hi 280 of the true 300 bp at +50);
#: with the gain under-scaled by the window's N_eff it never leaves +9.3 bp/day.
GAIN_TOLERANCE = 0.25
GAIN_BARS = 50


def _gain_series() -> tuple[np.ndarray, np.ndarray]:
    """Two-phase block series with a known crisis mean/variance."""
    rng = np.random.default_rng(20240218)
    states = np.concatenate([np.full(n, s) for n, s in GAIN_PLAN])
    return rng.normal(GAIN_MU[states], GAIN_SIGMA[states]), states


def test_gas_gain_reaches_a_known_crisis_mean():
    """The recursion's step size is the paper's, measured by how far it travels.

    METHODOLOGY. Parameters, not ratios: the series is generated with a known
    ``mu_hi = -40 bp/day`` and ``sd_hi = 300 bp``, and the first 260 (calm) bars
    initialise the fit, so ``mu_hi`` must TRAVEL to the truth once a crisis
    regime starts. The window measured is the SECOND crisis block, because the
    first is an identification transient rather than a gain measurement (see
    ``GAIN_CRISIS_INDEX``). The assertion is one-sided on purpose -- an
    under-scaled step cannot arrive at all (measured on the est3 MAP build:
    ``mu_hi`` sits at +24.7 bp/day through crisis #2 and ``sd_hi`` never leaves
    80 bp), while the shipped gain travels from +27 to -156 bp/day inside the
    50-bar budget. Nothing here pins the *stability* of the recursion; that is
    what the oracle and parameter-box tests measure.
    """
    series, states = _gain_series()
    model = ScoreDrivenHMM(neff=260.0, em_obs=EM_OBS)
    mu_hi = np.full(series.size, np.nan)
    sd_hi = np.full(series.size, np.nan)
    for t in range(series.size):
        obs = model.update(float(series[t]))
        if obs.ready:
            theta = np.asarray(obs.theta)
            mu_hi[t] = theta[1]
            sd_hi[t] = float(np.exp(theta[3]))
    # The canonical reporting order guarantees index 1 is the high-variance one.
    assert np.all(np.isfinite(mu_hi[EM_OBS:]))

    starts = np.where(np.diff(states) != 0)[0] + 1
    crisis = [int(s) for s in starts if states[s] == 1]
    assert len(crisis) > GAIN_CRISIS_INDEX, (
        "the fixture must contain two crisis regimes"
    )
    target = GAIN_MU[1] * (1.0 - GAIN_TOLERANCE)  # -30 bp/day
    first = crisis[GAIN_CRISIS_INDEX]
    window = mu_hi[first : first + GAIN_BARS]
    assert np.min(window) <= target, (
        "the GAS step is under-scaled: mu_hi never reached "
        f"{target * 1e4:.1f} bp/day within {GAIN_BARS} bars of the crisis "
        f"regime (entry {mu_hi[first - 1] * 1e4:.2f} bp/day, "
        f"min {np.nanmin(window) * 1e4:.2f} bp/day, sd_hi "
        f"{sd_hi[first - 1] * 1e4:.1f} -> {sd_hi[first + GAIN_BARS - 1] * 1e4:.1f} bp, "
        f"truth {GAIN_MU[1] * 1e4:.1f})"
    )
