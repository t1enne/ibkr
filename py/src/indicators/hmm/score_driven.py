"""Score-driven (GAS) two-state Gaussian HMM on daily log-returns.

Reference: Nystrup, Madsen & Lindström (2017/2018), "Dynamic portfolio
optimization across hidden market regimes", Quantitative Finance 18(1) 83-95
(accepted manuscript). This module implements that paper's recursion -- with
two DECLARED DEVIATIONS, stated under (5)-(7) below and in the caveats -- and
its forecast moments only; the allocation rule lives with the strategy.

Why this shape
--------------
The paper's persistence does not come from a sticky prior, a jump penalty or a
periodic refit: the parameters are re-estimated **every day** by a recursive
score (GAS) update on an exponentially weighted likelihood, and the decision
uses the **filtered** state distribution (forward pass only). A frozen fit is
the pathology this construction is meant to avoid, so nothing here stores a
full-sample estimate, runs a smoother, or decodes a Viterbi path.

The recursions, by paper equation
---------------------------------
(3)-(4) Weighted conditional log-likelihood with ``w_n = f**(t-n)`` and
        ``N_eff = 1/(1-f)``; the paper uses ``N_eff = 260`` (``f = 259/260``).
(5)-(7) Newton-type update with the Hessian replaced by the Fisher information,
        ``theta_t = theta_{t-1} + A * I_t^{-1} * grad l~_t``. Two aspects are
        DEVIATIONS from a literal reading, not the paper's own choices:

        * ``I_t`` is the **expected** information ``E[score score' | y_<t]``,
          not the realised outer product ``s s'``. (Neither is stated in the
          source spec, and its "matrix inversion lemma" -- a rank-one update of
          ``I^{-1}`` -- implies the realised form.) The expectation BUYS a
          bounded, well-scaled matrix: the realised form lets a single 13-sigma
          bar own the mu block while the sigma score cancels, so sigma stops
          moving exactly in a crisis (``dev_est3.md`` §4). It COSTS a curvature
          that is a functional of ``theta`` only, so it cannot reward one
          unusually informative bar, and its emission diagonals use a
          ``gamma^2 -> gamma`` plug-in that is 1.2-3x too large -- i.e. the
          variance response is if anything a lower bound.
        * ``A`` is applied per block. The transition block takes the paper's
          ``A = 1/N_eff``; the emission blocks take ``A = 1``, the exact Newton
          step of the truncated window objective. The paper's literal
          ``A = 1/N_eff`` everywhere is ``w_sum/N_eff = 0.633`` of that step
          (``w_sum = 164.5`` at ``W = 260``, ``N_eff = 260``), so the emission
          block steps 1.58x harder than the paper's recursion.
        Constrained parameters are reparameterised (``log sigma``, ``logit
        gamma``) so the recursion cannot leave the parameter space.
      The score is the **exact** score of the one-step predictive likelihood,
        computed in a single forward pass (Lystig & Hughes 2002) by carrying
        ``d alpha_t / d theta`` alongside ``alpha_t``. The paper is explicit
        that the HMM score cannot be approximated by the newest observation's
        own score.
(8)   Filtered distribution ``alpha_{t|t}`` -- forward pass, no backward pass.
(9)   ``k``-step forecast ``alpha_{t+k|t} = alpha_{t|t} Gamma**k``; this module
        exposes the one-step forecast used by the decision.
(10)-(11) Mixture mean/variance of the forecast.
(12)-(13) Within-state log-normal -> arithmetic conversion before mixing.

Initialisation is a plain ML/EM fit (Baum-Welch) on the first
``em_obs = 260`` returns, exactly as §2.1 describes; the recursion starts with
the next observation.

Caveats stated where they bite
------------------------------
* ``alpha_t`` is filtered with the parameters used for observation ``t`` (the
  paper's Eq. 8 uses the current parameters); the *forecast* is formed with the
  day's **updated** parameters, matching Algorithm 1's (i) then (ii) order. The
  tracked ``d alpha`` is the pathwise derivative, so the score is exact up to
  the (tiny) parameter step taken each day -- inherent to an online estimator.
* The recursion ascends a **posterior**, not a likelihood. A rolling weighted ML
  fit of a Gaussian HMM is unbounded: parking one component on the largest
  outlier in the window with ``sigma -> 0`` always lowers the loss, so the
  "spike state" is the *correct* maximum of the unconstrained problem (measured:
  removing the old parameter box made it worse, see ``dev_est2.md``). The cure
  is a bounded objective, not a fenced parameter space: an inverse-gamma prior
  on each ``sigma_j`` and a Dirichlet prior on each transition row enter the
  objective as ``grad += d log prior`` / ``info += -d2 log prior`` (see
  :func:`_prior_grad_info`). ``mu`` is a location parameter: nothing bounds it
  structurally and no prior is placed on it.
* One day's **step** is capped per coordinate (``mu`` <= 2%/day, ``log sigma``
  <= 2.0, logits <= 4.0) -- the step, never the parameter. ``model.capped_steps``
  counts every bar the guard fires, so a recursion leaning on it is visible.
  The only parameter bounds left are structural: ``log sigma`` inside this
  module's fixed numerical band and the off-diagonal logit inside
  ``[log 1e-4, log 0.5]`` (a regime lasts more than one bar).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

_LOG_2PI = float(np.log(2.0 * np.pi))
_N_STATES = 2
#: theta index permutation that swaps the two states: mu_0<->mu_1,
#: log sigma_0<->log sigma_1, logit g_01<->logit g_10 (Gamma -> P Gamma P').
_STATE_SWAP = np.array([1, 0, 3, 2, 5, 4])
#: theta = [mu_0, mu_1, log sigma_0, log sigma_1, logit g_01, logit g_10].
_N_PARAMS = 6
#: Structural parameter bounds and nothing else. ``log sigma`` covers
#: ``sigma in [1.2e-4, 4.5]`` -- eight orders of magnitude around SPY's 110 bp,
#: spanning "a one-bar spike" to "flat over every admissible return".
_LOG_SIGMA_MIN = -9.0
_LOG_SIGMA_MAX = 1.5
_LOGIT_EPS = 1e-4
#: Off-diagonal transition cap: a regime lasts more than one bar. This is an
#: identification constraint on the MODEL CLASS, not on the recursion.
_GAMMA_OFF_MAX = 0.5
#: Per-bar STEP caps (not parameter bounds): two orders past any plausible daily
#: move of the estimate, so they bound one Newton step, never the parameter.
_MU_STEP_MAX = 0.02
_LOG_SIGMA_STEP_MAX = 2.0
_LOGIT_STEP_MAX = 4.0
_EM_ITERS = 300
_EM_TOL = 1e-10
_SIGMA_FLOOR = 1e-6
#: Identification constraints on the ML/EM initialisation (see ``_em_fit``).
_EM_SIGMA_FRACTION = 0.1
_EM_SIGMA_CEIL_FRACTION = 4.0
_EM_MU_FRACTION = 0.5
_GAMMA_DIAG_FLOOR = 0.9
# -- MAP prior (identification of the objective) --------------------------------
#: The prior is worth 26 pseudo-observations, 10% of the paper's ``N_eff = 260``.
#: Far too weak to set a coordinate on its own, strong enough that the
#: unbounded ML direction (``sigma -> 0`` parked on an outlier) is not an
#: optimum: the inverse-gamma information is ``b/sigma``, i.e. it grows exactly
#: where the degeneracy lives (65 at an 80 bp state vs the window's ~330, but
#: 430 at a 1.2 bp spike state) and vanishes where sigma is large.
_PRIOR_OBS = 26.0
#: Inverse-gamma prior on each ``sigma_j``, in ``log sigma`` coordinates. Shape
#: 1.0 gives an infinite-mean ("weakly informative") inverse-gamma: zero
#: pull at the mode, strong pull below it. The mode is the calm-regime equity
#: vol, 80 bp/day, so the prior is neutral where SPY's low-vol state lives and
#: only reasserts itself in the spike-state corner (at 1.2 bp its information is
#: ~430, against the window's ~330 -- it wins exactly where the likelihood is
#: degenerate, and is worth 65/330 = 20% of the curvature at 80 bp, a ~19%
#: downward MAP shrinkage of a 240 bp crisis state).
_PRIOR_SIGMA_MODE = 0.008
_PRIOR_IG_SHAPE = 1.0
#: Inverse-gamma rate, chosen so the prior mode is ``_PRIOR_SIGMA_MODE``.
_PRIOR_IG_RATE = (_PRIOR_IG_SHAPE + 1.0) * _PRIOR_SIGMA_MODE
#: Dirichlet prior on each transition row, by its pseudo-counts, chosen for
#: BOUNDEDNESS ONLY. Both entries must exceed 1: a count <= 1 leaves the
#: posterior flat-to-singular at the simplex edge, i.e. it PINS the probability
#: at the bound instead of leaving the likelihood free to place it. Strength
#: ``a_diag + a_off - 2 = 0.15`` pseudo-observations, so it never sets the
#: persistence (an earlier version carried (24.7, 1.3) -- worth 26 -- and dragged
#: SPY's g01 to 0.013, starving the high-variance component: sd_hi 103 bp through
#: the GFC).
#: The MODE is placed ON the admissible ceiling, not outside it. The off-diagonal
#: logit is capped at ``log 0.5``, i.e. ``g <= 1/3``, and these counts put the
#: mode at ``(a_off-1)/(a_diag+a_off-2) = 1/3`` exactly. Equal counts (mode 1/2,
#: outside the box) leave a constant ``(a_off-a_diag)/2 = +0.05`` gradient that
#: points INTO the bound at the bound: harmless under the old complete-data
#: transition curvature, which was stiff enough to ignore it, but with the (much
#: smaller) marginal curvature it walked the block onto ``g = 1/3`` for a 38-bar
#: spell in 2005 (``dev_est6.md`` §4). A prior whose only job is boundedness
#: must contribute no force at the bound; at the mode the gradient is zero there.
_PRIOR_GAMMA_DIAG = 1.1
_PRIOR_GAMMA_OFF = 1.05
# -- Expected information -------------------------------------------------------
#: ``E[(z^2 - 1)^2]`` for a standard normal ``z`` (``z^2 ~ chi^2_1``, whose
#: variance is 2): the ``log sigma`` score's mean square, independent of the
#: parameter and therefore bounded -- the whole point of using the expectation.
_E_SIGMA_SCORE_SQ = 2.0


# ---------------------------------------------------------------------------
# Parameterisation
# ---------------------------------------------------------------------------


def _sigmoid(z: float) -> float:
    assert np.isfinite(z)
    return float(1.0 / (1.0 + np.exp(-z)))


def _logit(p: float) -> float:
    assert 0.0 < p < 1.0
    return float(np.log(p / (1.0 - p)))


def _clip_p(p: float) -> float:
    return float(min(1.0 - _LOGIT_EPS, max(_LOGIT_EPS, p)))


def _unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """theta -> (mu, sigma, Gamma). Gamma[i, j] = P(S_t = j | S_{t-1} = i)."""
    assert theta.shape == (_N_PARAMS,)
    mu = np.asarray(theta[0:2], dtype=float)
    sigma = np.exp(np.asarray(theta[2:4], dtype=float))
    g01 = _sigmoid(float(theta[4]))
    g10 = _sigmoid(float(theta[5]))
    gamma = np.array([[1.0 - g01, g01], [g10, 1.0 - g10]])
    return mu, sigma, gamma


def _dgamma(theta: np.ndarray) -> np.ndarray:
    """d Gamma[i, j] / d theta_k, shape (2, 2, 6)."""
    g01 = _sigmoid(float(theta[4]))
    g10 = _sigmoid(float(theta[5]))
    d = np.zeros((_N_STATES, _N_STATES, _N_PARAMS))
    d[0, 0, 4] = -g01 * (1.0 - g01)
    d[0, 1, 4] = g01 * (1.0 - g01)
    d[1, 0, 5] = g10 * (1.0 - g10)
    d[1, 1, 5] = -g10 * (1.0 - g10)
    return d


def _pack(mu: np.ndarray, sigma: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Constrained parameters -> unconstrained theta (structural bounds only).

    ``mu`` is NOT clipped: it is a location, and nothing structurally bounds it.
    """
    assert mu.shape == (_N_STATES,) and sigma.shape == (_N_STATES,)
    log_sigma = np.clip(
        np.log(np.asarray(sigma, dtype=float)), _LOG_SIGMA_MIN, _LOG_SIGMA_MAX
    )
    g01 = _clip_p(float(gamma[0, 1]))
    g10 = _clip_p(float(gamma[1, 0]))
    return np.array(
        [
            float(mu[0]),
            float(mu[1]),
            float(log_sigma[0]),
            float(log_sigma[1]),
            _logit(g01),
            _logit(g10),
        ]
    )


def _prior_grad_info(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradient and observed information of the MAP prior on ``theta``.

    WHY this exists: the weighted ML objective is unbounded above -- a component
    parked on the window's largest outlier with ``sigma -> 0`` has unbounded
    density, so a rolling 260-bar ML fit walks there whenever the window holds
    such a bar (measured on real SPY: 216 consecutive bars on a sigma floor).
    Forbidding the region only moves where the spike sits. A prior on the
    SCALE (and on the transition rows) makes the penalised objective bounded,
    so the spike state stops being an optimum instead of being clipped away.

    Both priors are in ``theta`` coordinates, so the terms drop into the
    existing recursion unchanged:

    * inverse-gamma on ``sigma_j``, ``l = -(a+1) u - b e^{-u}`` with
      ``u = log sigma``: ``dl/du = b/sigma - (a+1)``, ``-d2l/du2 = b/sigma``;
    * Dirichlet on each transition row, pseudo-counts ``(a_diag, a_off)``:
      ``l = (a_diag-1) log(1-g) + (a_off-1) log g``, so
      ``dl/du = (a_off-1)(1-g) - (a_diag-1) g`` and
      ``-d2l/du2 = (a_diag + a_off - 2) g (1-g)``.

    ``mu`` gets nothing: a location parameter has no structural bound and the
    moving weighted likelihood identifies it (it is the only coordinate whose
    score vanishes at the weighted mean of its own component).

    Returns ``(grad, info)`` already in the recursion's accumulation units:
    the inverse-gamma log-density is weightless by construction, so it carries
    ``_PRIOR_OBS`` here; the Dirichlet's strength IS its pseudo-counts, which
    sum to ``_PRIOR_OBS``, so scaling them again would count the prior twice.
    """
    assert theta.shape == (_N_PARAMS,)
    grad = np.zeros(_N_PARAMS)
    info = np.zeros((_N_PARAMS, _N_PARAMS))
    for j in range(_N_STATES):
        sigma = math.exp(float(theta[2 + j]))
        assert sigma > 0.0
        grad[2 + j] = _PRIOR_OBS * (_PRIOR_IG_RATE / sigma - (_PRIOR_IG_SHAPE + 1.0))
        info[2 + j, 2 + j] = _PRIOR_OBS * _PRIOR_IG_RATE / sigma
    a_diag = _PRIOR_GAMMA_DIAG
    a_off = _PRIOR_GAMMA_OFF
    assert a_diag > 1.0, "a Dirichlet count <= 1 unbounds the posterior"
    assert a_off > 1.0, "a Dirichlet count <= 1 unbounds the posterior"
    for k in (4, 5):  # logit g_01 (row 0) and logit g_10 (row 1), same form
        g = _sigmoid(float(theta[k]))
        grad[k] = (a_off - 1.0) * (1.0 - g) - (a_diag - 1.0) * g
        info[k, k] = (a_diag + a_off - 2.0) * g * (1.0 - g)
    assert np.all(np.isfinite(grad)) and np.all(np.isfinite(info))
    return grad, info


def _newton_step(info: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """Rank-cut Fisher solve: no step along the information's null space.

    ``pinv`` inverts every direction, including the ones the window never
    visits: an unvisited state's ``mu``/``log sigma`` direction has information
    ~1e-9 next to the visited state's O(1), and dividing a noise gradient by
    that eigenvalue is a random walk (measured: ``mu = +17%/day``,
    ``sigma = 448%/day`` on a 2021-2023 slice). The relative cut keeps the
    Newton step on the identified subspace and takes zero along the rest.

    ``info`` is the WHOLE assembled matrix (emission + transition + prior), so
    ``top`` is the mu block's scale and the cut is GLOBAL, not per block: a
    small but live direction in another block falls under the same absolute
    threshold and is zeroed. Zeroed, not inverted, so the step cannot be
    amplified; it is simply not taken.
    """
    assert info.shape == (_N_PARAMS, _N_PARAMS) and grad.shape == (_N_PARAMS,)
    assert np.allclose(info, info.T), "information must be symmetric"
    values, vectors = np.linalg.eigh(info)
    top = float(values[-1])
    assert top >= 0.0
    if top <= 0.0:
        return np.zeros(_N_PARAMS)
    keep = values > 1e-8 * top
    inverse = np.where(keep, 1.0 / np.where(keep, values, 1.0), 0.0)
    step = vectors @ (inverse * (vectors.T @ grad))
    assert np.all(np.isfinite(step))
    return step


# ---------------------------------------------------------------------------
# Transition block: the marginal information of the exact score
# ---------------------------------------------------------------------------

#: Trapezoid step of the mixture-integral grid, in units of the NARROWER
#: component's ``sigma``. The integrand's sharpest feature is at that scale, and
#: the trapezoid rule on an analytic, Gaussian-decaying integrand is already
#: converged there: measured relative error against a 25x finer reference over
#: seven (sigma, mean, weight) configurations was <= 1.1e-4 at this step
#: (1.5e-6 at 0.35, 9e-9 at 0.25), spanning near-identical components (K =
#: 0.022) to a 10x sigma ratio (K = 1169).
_INFO_GRID_STEP = 0.5
#: Grid half-span, in units of the WIDER ``sigma``: beyond 9 sigma of the wider
#: component the integrand is below 1e-17 of its maximum.
_INFO_GRID_SPAN = 9.0


def _mixture_integrals(
    pred0: np.ndarray, pred1: np.ndarray, mu: np.ndarray, sigma: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(A, B, C) = int N0^2/p, int N0 N1/p, int N1^2/p``, one per bar.

    ``p = P_n0 N0 + P_n1 N1`` is the bar's one-step predictive density and
    ``N_j`` its state-``j`` emission. These three integrals are all the mixture
    geometry the logit block needs, because the exact score's two logit
    coordinates have the closed form ``s_i = (a_i N0 + b_i N1) / p`` (asserted
    against the kernel on every bar in ``_recursive_step``), so

        ``E[s_i s_j | y_<n] = q_i' M q_j``,  ``M = [[A, B], [B, C]]``.

    Composite trapezoid over ``[min(mu) - _INFO_GRID_SPAN * sigma_max,
    max(mu) + _INFO_GRID_SPAN * sigma_max]`` with step ``_INFO_GRID_STEP *
    sigma_min``. The grid is in ``y``, NOT in a component's standardised
    coordinate: a Gauss-Hermite rule per component was tried and rejected --
    integrating in the wider component's own coordinate leaves the narrower
    component's peak under-resolved (measured: K drifted 10% and had not
    converged at 32 nodes when the sigma ratio is 3).
    """
    assert pred0.shape == pred1.shape
    assert mu.shape == sigma.shape == (_N_STATES,)
    assert np.all(sigma > 0.0) and np.all(np.isfinite(mu))
    h = _INFO_GRID_STEP * float(np.min(sigma))
    lo = float(np.min(mu)) - _INFO_GRID_SPAN * float(np.max(sigma))
    hi = float(np.max(mu)) + _INFO_GRID_SPAN * float(np.max(sigma))
    assert hi > lo and h > 0.0
    grid = lo + h * np.arange(int(np.ceil((hi - lo) / h)) + 1)
    dens = np.empty((_N_STATES, grid.size))
    for j in range(_N_STATES):
        z = (grid - mu[j]) / sigma[j]
        dens[j] = np.exp(-0.5 * z * z) / (sigma[j] * math.sqrt(2.0 * np.pi))
    # p varies per bar, the two densities do not: broadcast (n, 1) x (1, k).
    mix = pred0[:, None] * dens[0] + pred1[:, None] * dens[1]
    assert np.all(mix > 0.0)
    int_00 = (dens[0] * dens[0] / mix).sum(axis=1) * h
    int_01 = (dens[0] * dens[1] / mix).sum(axis=1) * h
    int_11 = (dens[1] * dens[1] / mix).sum(axis=1) * h
    assert np.all(int_00 >= 0.0) and np.all(int_11 >= 0.0)
    # The mixture integrals are a Gram-like system: `int (u N0 + v N1)^2 / p`
    # must be non-negative for every (u, v), i.e. the 2x2 form is PSD.
    assert np.all(int_00 * int_11 - int_01 * int_01 >= -1e-12)
    return int_00, int_01, int_11


def _transition_block(
    coef: np.ndarray,
    pred: np.ndarray,
    weights: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> tuple[float, float, float]:
    """Marginal information of the transition block: ``(i44, i45, i55)``.

    ``coef`` carries the two score coefficients per bar for each logit
    (``(a_4, b_4, a_5, b_5)``), ``pred`` the bar's predictive weights and
    ``weights`` the window's exponential weights. The block is a quadratic form
    in the three mixture integrals, and the CROSS TERM is carried, not dropped:
    the block is very nearly rank one (both logits are driven by the same
    ``N_1 - N_0``), so a diagonal here invents a direction the data does not
    constrain -- measured ``rho(s_4, s_5)`` of -0.40..-0.95 on real SPY against
    the 0 a diagonal asserts (``dev_est6.md`` §2).
    """
    n_bars = weights.size
    assert coef.shape == (n_bars, 4)
    assert pred.shape == (n_bars, 2)
    int_00, int_01, int_11 = _mixture_integrals(pred[:, 0], pred[:, 1], mu, sigma)
    m00, m01, m11 = weights * int_00, weights * int_01, weights * int_11
    t4a, t4b, t5a, t5b = coef[:, 0], coef[:, 1], coef[:, 2], coef[:, 3]
    i44 = float(np.sum(t4a * t4a * m00 + 2.0 * t4a * t4b * m01 + t4b * t4b * m11))
    i55 = float(np.sum(t5a * t5a * m00 + 2.0 * t5a * t5b * m01 + t5b * t5b * m11))
    i45 = float(
        np.sum(t4a * t5a * m00 + (t4a * t5b + t4b * t5a) * m01 + t4b * t5b * m11)
    )
    # Positive AND negative space: a 2x2 covariance sits inside its own
    # Cauchy-Schwarz cone, and this one is very nearly singular (the null
    # direction is ``(occ_n1 c_10, occ_n0 c_01)``).
    assert i44 >= 0.0 and i55 >= 0.0
    assert abs(i45) <= math.sqrt(i44 * i55) + 1e-12
    return i44, i45, i55


# ---------------------------------------------------------------------------
# Exact score: one forward step of alpha and d alpha / d theta
# ---------------------------------------------------------------------------


def _step_scalars(
    a0: float,
    a1: float,
    da0: list[float],
    da1: list[float],
    y: float,
    theta: np.ndarray,
) -> tuple[float, float, list[float], list[float], list[float], float, float, float]:
    """One filter step in plain floats: the whole estimator's arithmetic.

    Both :func:`_forward_step` (one call per bar) and
    :meth:`ScoreDrivenHMM._recursive_step` (one call per window bar) route
    through this kernel, so the
    score the recursion ascends and the score the filter reports cannot drift
    apart, and the 260-step inner loop does not pay numpy's per-op overhead on
    six-element arrays (measured 2.7x: 37 -> 14 us per step).

    Returns ``(alpha_0, alpha_1, dalpha_0, dalpha_1, score, loglik, p_0, p_1)``:
    the new filtered distribution, its exact derivative, the exact score of
    ``log p(y | y_<t)``, that log-likelihood and the two unnormalised emission
    densities (the caller needs them for the joint ``xi``).
    """
    mu, sigma, gamma = _unpack(theta)
    g01 = _sigmoid(float(theta[4]))
    g10 = _sigmoid(float(theta[5]))
    c01 = g01 * (1.0 - g01)
    c10 = g10 * (1.0 - g10)
    g00, g01_, g10_, g11 = gamma[0, 0], gamma[0, 1], gamma[1, 0], gamma[1, 1]

    # Predictive state distribution.
    pred0 = a0 * g00 + a1 * g10_
    pred1 = a0 * g01_ + a1 * g11

    # Emission densities and their derivatives w.r.t. (mu_j, log sigma_j).
    z0 = (y - mu[0]) / sigma[0]
    z1 = (y - mu[1]) / sigma[1]
    logp0 = -0.5 * _LOG_2PI - np.log(sigma[0]) - 0.5 * z0 * z0
    logp1 = -0.5 * _LOG_2PI - np.log(sigma[1]) - 0.5 * z1 * z1
    d0 = [z0 / sigma[0], 0.0, z0 * z0 - 1.0, 0.0, 0.0, 0.0]
    d1 = [0.0, z1 / sigma[1], 0.0, z1 * z1 - 1.0, 0.0, 0.0]

    # Shift by the largest log density so a far-tail bar cannot underflow both
    # states. The shift is a function of theta; its derivative dm is added back
    # to the score and cancels exactly in the alpha ratio.
    if logp0 >= logp1:
        m, dm = logp0, d0
    else:
        m, dm = logp1, d1
    p0 = math.exp(logp0 - m)
    p1 = math.exp(logp1 - m)

    psi0 = pred0 * p0
    psi1 = pred1 * p1
    lik = psi0 + psi1
    assert lik > 0.0
    alpha0 = psi0 / lik
    alpha1 = psi1 / lik
    assert abs(alpha0 + alpha1 - 1.0) < 1e-9

    # dpred[j, k] = sum_i gamma[i, j] dalpha_prev[i, k]
    #              + sum_i alpha_prev[i] dgamma[i, j, k],
    # with dgamma non-zero only on the two logit columns (k = 4, 5).
    dpred0 = [g00 * da0[k] + g10_ * da1[k] for k in range(_N_PARAMS)]
    dpred1 = [g01_ * da0[k] + g11 * da1[k] for k in range(_N_PARAMS)]
    dpred0[4] -= a0 * c01
    dpred1[4] += a0 * c01
    dpred0[5] += a1 * c10
    dpred1[5] -= a1 * c10

    score = [0.0] * _N_PARAMS
    total = [0.0] * _N_PARAMS
    for k in range(_N_PARAMS):
        t = (dpred0[k] + pred0 * (d0[k] - dm[k])) * p0 + (
            dpred1[k] + pred1 * (d1[k] - dm[k])
        ) * p1
        total[k] = t
        score[k] = t / lik + dm[k]
    nda0 = [
        ((dpred0[k] + pred0 * (d0[k] - dm[k])) * p0 - alpha0 * total[k]) / lik
        for k in range(_N_PARAMS)
    ]
    nda1 = [
        ((dpred1[k] + pred1 * (d1[k] - dm[k])) * p1 - alpha1 * total[k]) / lik
        for k in range(_N_PARAMS)
    ]
    return alpha0, alpha1, nda0, nda1, score, m + math.log(lik), p0, p1


def _forward_step(
    alpha_prev: np.ndarray,
    dalpha_prev: np.ndarray,
    y: float,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Advance the filter one observation and return its exact score.

    ``alpha_prev`` is ``alpha_{t-1|t-1}``, ``dalpha_prev`` its derivative
    ``(2, 6)`` under the same parameter vector ``theta``. Returns the filtered
    ``alpha_t``, its derivative, the score of ``log p(y_t | y_{<t})`` at
    ``theta``, that predictive log-likelihood, and the two-state filtered joint
    ``xi[i, j] = P(S_{t-1}=i, S_t=j | y_1..y_t)`` (whose off-diagonal mass is the
    probability that a regime change occurred at ``t``).
    """
    assert alpha_prev.shape == (_N_STATES,)
    assert dalpha_prev.shape == (_N_STATES, _N_PARAMS)
    assert np.isfinite(y)
    a0, a1, da0, da1, score, loglik, p0, p1 = _step_scalars(
        float(alpha_prev[0]),
        float(alpha_prev[1]),
        [float(v) for v in dalpha_prev[0]],
        [float(v) for v in dalpha_prev[1]],
        y,
        theta,
    )
    mu, sigma, gamma = _unpack(theta)
    alpha = np.array([a0, a1])
    lik = p0 * (alpha_prev @ gamma)[0] + p1 * (alpha_prev @ gamma)[1]
    xi = np.array(
        [
            [
                float(alpha_prev[0]) * gamma[0, 0] * p0 / lik,
                float(alpha_prev[0]) * gamma[0, 1] * p1 / lik,
            ],
            [
                float(alpha_prev[1]) * gamma[1, 0] * p0 / lik,
                float(alpha_prev[1]) * gamma[1, 1] * p1 / lik,
            ],
        ]
    )
    return alpha, np.array([da0, da1]), np.array(score), loglik, xi


# ---------------------------------------------------------------------------
# ML/EM initialisation of the first ``em_obs`` observations
# ---------------------------------------------------------------------------


def _em_fit(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Baum-Welch ML fit of the 2-state Gaussian HMM (returns only).

    A plain unconstrained ML fit is **not** usable here: the Gaussian HMM
    likelihood is unbounded. On the paper's own 260-observation initialisation
    window (SPY 2004-02-02..2005-02-11) an unconstrained EM -- mine and
    ``hmmlearn`` both -- lands on a degenerate solution (a one-bar SPIKE state
    with ``sigma -> 0``, or a positive-day/negative-day split with ``gamma_ii ~
    0.5``), which then drives the recursion into a permanent one-label pin. The
    four identification constraints below keep the fit inside the paper's own
    state semantics (two persistent low/high-variance regimes, small means):
    ``sigma`` in ``[0.1, 4] * sd(y)``, ``|mu - mean(y)| <= 0.5 * sd(y)`` and
    ``gamma_ii >= 0.9`` (a state is a regime, not a coin flip). They are
    deviations from a bare ML fit, recorded in the module docstring; without
    them the estimator cannot pass its synthetic-oracle and no-pinning tests.

    State 0 is canonicalised to the LOW-variance component (the paper's
    "state 1", Sec. 4.2).
    """
    assert y.ndim == 1 and y.size >= 4
    sample_sd = float(np.std(y))
    assert sample_sd > 0.0
    var_floor = max(_SIGMA_FLOOR, _EM_SIGMA_FRACTION * sample_sd)
    var_ceil = _EM_SIGMA_CEIL_FRACTION * sample_sd
    mu_centre = float(np.mean(y))
    mu_bound = _EM_MU_FRACTION * sample_sd
    # Deterministic start on the variance split the paper's states describe.
    mu = np.array([mu_centre, mu_centre])
    sigma = np.array([0.6 * sample_sd, 1.5 * sample_sd])
    gamma = np.array([[0.95, 0.05], [0.1, 0.9]])
    loglik = -np.inf

    for _ in range(_EM_ITERS):
        # Forward pass with scaling, storing the scaled alphas.
        alpha_f = np.zeros((y.size, _N_STATES))
        scale = np.zeros(y.size)
        a = np.array([0.5, 0.5])
        ll = 0.0
        for t in range(y.size):
            pred = a @ gamma
            dens = np.exp(-0.5 * ((y[t] - mu) / sigma) ** 2) / (
                sigma * np.sqrt(2.0 * np.pi)
            )
            psi = pred * dens
            c = float(np.sum(psi))
            if c <= 0.0:  # numerically dead branch: stop, keep the last fit
                break
            a = psi / c
            alpha_f[t] = a
            scale[t] = c
            ll += float(np.log(c))
        else:
            # Backward pass on the scaled alphas.
            beta = np.ones((y.size, _N_STATES))
            for t in range(y.size - 2, -1, -1):
                dens = np.exp(-0.5 * ((y[t + 1] - mu) / sigma) ** 2) / (
                    sigma * np.sqrt(2.0 * np.pi)
                )
                beta[t] = (gamma @ (dens * beta[t + 1])) / scale[t + 1]

            post = alpha_f * beta
            post /= np.sum(post, axis=1, keepdims=True)

            # Pairwise posterior, then the M-step.
            xi_sum = np.zeros((_N_STATES, _N_STATES))
            for t in range(y.size - 1):
                dens = np.exp(-0.5 * ((y[t + 1] - mu) / sigma) ** 2) / (
                    sigma * np.sqrt(2.0 * np.pi)
                )
                num = alpha_f[t][:, None] * gamma * (dens * beta[t + 1])[None, :]
                den = float(np.sum(num))
                if den > 0.0:
                    xi_sum += num / den

            count = post.sum(axis=0)
            if not np.any(count < 3.0):
                mu = (post * y[:, None]).sum(axis=0) / count
                mu = np.clip(mu, mu_centre - mu_bound, mu_centre + mu_bound)
                var = (post * (y[:, None] - mu) ** 2).sum(axis=0) / count
                sigma = np.clip(np.sqrt(var), var_floor, var_ceil)
                gamma = xi_sum / count[:, None]
                gamma = np.maximum(gamma, _LOGIT_EPS)
                gamma /= gamma.sum(axis=1, keepdims=True)
                # Persistence floor: a state is a regime, not a coin flip.
                for k in range(_N_STATES):
                    if gamma[k, k] < _GAMMA_DIAG_FLOOR:
                        gamma[k, k] = _GAMMA_DIAG_FLOOR
                        gamma[k, 1 - k] = 1.0 - _GAMMA_DIAG_FLOOR

            if abs(ll - loglik) < _EM_TOL:
                loglik = ll
                break
            loglik = ll

    # Canonicalise: component 0 = low variance (the paper's state 1).
    if sigma[0] > sigma[1]:
        mu = mu[::-1].copy()
        sigma = sigma[::-1].copy()
        gamma = gamma[::-1, ::-1].copy()
    theta = _pack(mu, sigma, gamma)
    mu, sigma, gamma = _unpack(theta)
    alpha_last = np.array([0.5, 0.5])
    dalpha = np.zeros((_N_STATES, _N_PARAMS))
    for t in range(y.size):
        alpha_last, dalpha, _score, _ll, _xi = _forward_step(
            alpha_last, dalpha, y[t], theta
        )
    return theta, alpha_last, dalpha


# ---------------------------------------------------------------------------
# Online estimator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GasObservation:
    """One day's filtered state, forecast moments and transition probabilities.

    ``filtered`` and ``forecast_prob`` are ordered ``(low-vol, high-vol)``;
    ``transition[i][j] = P(S_t = j | S_{t-1} = i)`` in the same order.
    ``change_prob`` is the filtered probability that a regime change occurred
    on this bar, ``P(S_t != S_{t-1} | y_1..y_t)``.
    """

    ready: bool
    filtered: tuple[float, float] = (0.5, 0.5)
    forecast_prob: tuple[float, float] = (0.5, 0.5)
    mean: float = 0.0
    variance: float = 0.0
    transition: tuple[tuple[float, float], tuple[float, float]] = (
        (0.0, 1.0),
        (1.0, 0.0),
    )
    change_prob: float = 0.0
    loglik: float = 0.0
    theta: tuple[float, ...] = ()


class ScoreDrivenHMM:
    """Recursive score-driven 2-state Gaussian HMM on log-returns.

    ``neff`` is the paper's effective memory length ``N_eff`` (260 for the MPC
    arm, 504 for the RBAA control); ``em_obs`` is the ML/EM initialisation
    window (260). Feed exactly one log-return per daily bar with :meth:`update`;
    nothing outside the update reads future data.
    """

    def __init__(self, neff: float = 260.0, em_obs: int = 260) -> None:
        assert neff > 1.0, "N_eff must exceed one observation"
        assert em_obs >= 20, "EM needs a usable sample"
        self.neff = float(neff)
        self.f = 1.0 - 1.0 / self.neff
        self.em_obs = int(em_obs)
        #: Rolling window the recursion differentiates: the finite-memory form
        #: of the paper's exponential weights (see ``_recursive_step``).
        self.window = int(round(self.neff))
        assert self.window >= self.em_obs, "the window must cover the EM fit"
        #: Gain ``A`` on the TRANSITION block: the paper's own ``1/N_eff`` (see
        #: ``_recursive_step``). The emission blocks use ``A = 1`` -- the exact
        #: Newton step of the truncated window objective -- which is a DECLARED
        #: deviation from the paper's uniform ``A = 1/N_eff``: it steps 1.58x
        #: harder on every emission coordinate (``w_sum = 164.5`` at ``W = 260``
        #: against the nominal ``N_eff = 260``). See the module docstring.
        self._a_logit = 1.0 / self.neff
        assert 0.0 < self._a_logit <= 1.0
        #: Gain ``A`` on the EMISSION block. ``1.0`` is the exact one-observation
        #: Newton step of the truncated weighted objective; the paper's literal
        #: normalisation is ``w_sum / N_eff`` (see the module docstring, which
        #: states the measured cost of both). A plain attribute so a measurement
        #: can run the paper-literal variant without editing the module.
        self._a_emit = 1.0
        assert 0.0 < self._a_emit <= 1.0
        self.n_seen = 0
        self.n_updates = 0
        self._buffer: list[float] = []
        #: Per-coordinate count of bars on which the STEP cap fired. A recursion
        #: that leans on the guard shows up here instead of silently clipping.
        self.capped_steps = np.zeros(_N_PARAMS, dtype=int)
        #: Structural sigma band (fixed constants, not the init window's sd: that
        #: band was an accident of where the series started and it bound 10.6% of
        #: real-SPY bars -- see ``dev_est2.md``).
        self._log_sigma_floor = _LOG_SIGMA_MIN
        self._log_sigma_ceil = _LOG_SIGMA_MAX
        self._theta: np.ndarray | None = None
        self._alpha = np.array([0.5, 0.5])
        self._dalpha = np.zeros((_N_STATES, _N_PARAMS))
        #: Rolling window: ``_win_y[k]`` with ``_win_alpha[k]`` the filtered
        #: distribution *before* it. Bounded by ``window`` entries.
        self._win_y: list[float] = []
        self._win_alpha: list[np.ndarray] = []

    @property
    def ready(self) -> bool:
        return self._theta is not None

    # -- internals ---------------------------------------------------------

    def _initialise(self) -> None:
        y = np.asarray(self._buffer, dtype=float)
        assert y.size == self.em_obs
        theta, alpha, dalpha = _em_fit(y)
        # Seed the rolling window with the initialisation observations and the
        # filtered distribution preceding each of them. The recursion re-derives
        # every score in this window at the current theta (see
        # ``_recursive_step``), so the window must be carried from here on.
        self._theta = theta
        self._alpha = alpha
        self._dalpha = dalpha
        self._win_y = []
        self._win_alpha = []
        a = np.array([0.5, 0.5])
        da = np.zeros((_N_STATES, _N_PARAMS))
        for obs in y:
            self._win_y.append(float(obs))
            self._win_alpha.append(a.copy())
            a, da, _s, _ll, _xi = _forward_step(a, da, float(obs), theta)
        assert len(self._win_y) == self.em_obs
        assert abs(float(np.sum(a)) - 1.0) < 1e-9

    def _recursive_step(self) -> np.ndarray:
        """One GAS update: the weighted score and Fisher at the CURRENT theta.

        Paper (6)-(7): ``theta_t = theta_{t-1} + A I_t^{-1} grad l~_t`` with
        ``A = 1/N_w``, where

            l~_t(theta) = sum_n w_n log p(y_n | y_<n, theta),  w_n = f^(t-n)
            grad l~_t = sum_n w_n s_n(theta)                 (weights sum N_w)
            I_t        = (1/N_w) sum_n w_n E[s_n s_n']       (per observation,
                                                              eq. 6)

        so ``A I_t^{-1} grad`` is exactly ONE Newton step on the moving weighted
        likelihood -- the paper's sentence "it is common to choose A ~ N_eff"
        read as the normalisation it is. Written without the two cancelling
        ``1/N_w`` factors the step is simply ``S^{-1} G`` with
        ``S = sum_n w_n E[s_n s_n']`` -- so ``S`` must be the UN-normalised sum
        and must not be scaled again, and the cancellation is exact only for the
        untruncated geometric sum: the window carries ``w_sum = 164.5`` against
        the nominal ``N_eff = 260``, so the paper's literal ``A = 1/N_eff`` is
        ``0.633`` of this step. ``A`` is therefore applied PER BLOCK and
        declared as a deviation (module docstring): ``A = 1`` on the emission
        blocks (1.58x the paper's recursion) and ``A = 1/N_eff`` on the
        transition block (the paper's value).

        **The information is the EXPECTED one, ``E[s s' | y_<n]``, not the
        realised outer product.** WHY (measured on real SPY, ``dev_est3.md``
        §4): the realised ``sum_n w_n s_n s_n'`` is ill-conditioned exactly when
        it matters most. At the foot of a crisis one bar with ``z = (y - mu) /
        sigma ~ 13`` contributes ``s_mu s_mu' ~ (z/sigma)^2 ~ 1e8`` to the mu
        block while the sigma score is a near-cancellation (a handful of
        ``z^2 - 1 ~ +30..+80`` against ~250 calm bars at -1) and its information
        stays ~1e4. The solve then satisfies the giant mu equation by moving mu
        alone: over 2008-09-10..2008-11-05, which contains -502, -1011 and
        +1285 bp bars, ``sigma_hi`` moved 144 -> 133 bp, DOWN. The expectation
        removes the single-bar dominance because it is taken over the model's
        own predictive distribution -- a density whose tails are the current
        ``sigma``, not a realised outlier -- so the matrix is bounded and
        well-scaled by construction and no clip is needed to steer it.

        Derivation (``theta = (mu_j, log sigma_j, logit g_01, logit g_10)``).
        With predictive weights ``P_nj = P(S_n = j | y_<n)``, a one-step
        Gaussian emission ``N_j`` and ``gamma_nj`` the filtered weight, the
        conditional expectation is taken component by component (Jensen plug-in
        ``alpha_j(y)^2 ~ gamma_j(y)``):

            mu_j         E[(z_j/sigma_j)^2]        = 1 / sigma_j^2
            log sigma_j  E[(z_j^2 - 1)^2]          = 2      (z^2 ~ chi^2_1)
            mu_j x log sigma_j  E[z_j (z_j^2 - 1)] = 0
            cross-state / cross-block             ~ 0
            logit block  E[s_i s_j] -- RANK ONE, not two independent diagonals

        Emission blocks: ``diag(P_n0/sigma_0^2, P_n1/sigma_1^2, 2 P_n0,
        2 P_n1)``, the mu/sigma diagonals of each state taken at the larger of
        ``P_nj`` and the filtered ``gamma_nj`` (rare-state guard, see the loop)
        -- inert whenever the two agree.

        Transition block: the score's logit coordinates are EXACTLY
        ``s_4 = (a_4 N_0 + b_4 N_1)/p``, ``s_5 = (a_5 N_0 + b_5 N_1)/p`` with
        ``p = P_n0 N_0 + P_n1 N_1`` and ``(a_i, b_i)`` the filter-feedback term
        (the pathwise derivative ``da``) plus ``+-(occ_nj c_0j)``, the
        likelihood-ratio term. Hence ``E[s_i s_j | y_<n] = q_i' M q_j`` with
        ``M = [[A, B], [B, C]]``, the three mixture integrals computed by
        :func:`_mixture_integrals`. Two consequences. It is the MARGINAL
        information, not the state-conditional one, and it is very nearly RANK
        ONE: the likelihood-ratio part is exactly rank one (both logits are
        driven by the same ``N_1 - N_0``) and only the feedback term lifts the
        block off that, so ``|rho(s_4, s_5)|`` is measured at 0.40-0.95 on real
        SPY. The diagonal of complete-data ``occ_nj g_j (1-g_j)`` used through
        ``dev_est5`` is the wrong object twice over: 1.1-15x too large against
        the marginal block, 8-317x too large against its likelihood-ratio part
        alone, and it invents the second direction the block does not have.

        The cut in :func:`_newton_step` is GLOBAL (``1e-8 * top``, ``top`` being
        the mu block's eigenvalue), not per block: a transition-block direction
        that falls under it is truncated to a zero step (measured at one probe
        window: its minor eigenvalue at ``0.10x`` the cut). That direction is
        then un-stepped, never amplified, so the step stays bounded -- but the
        cut does not distinguish "an unvisited state's direction" from a live
        one, and the prior's curvature enters the assembled matrix before the
        cut is taken.

        **Every score is re-derived at ``theta_{t-1}``.** This is what makes the
        step a Newton step on a real objective rather than a random walk: an
        EWMA of previously-recorded scores is ``sum_n w_n s_n(theta_n)``, i.e.
        the gradient of a *different* objective, and it therefore exerts no
        restoring force. Measured on the oracle fixture (5000 bars, truth
        mu = [+7, -12] bp/day, sigma = [80, 240] bp): the EWMA form's direction
        against the true weighted gradient had cos = -0.95..-0.99 at bars 5-500,
        and every tested step *lowered* the predictive log-likelihood
        (-13 .. -209 nats); it walked to the parameter box (mu on the +-1% clip
        49% of bars, sigma on its ceiling 22%) and finished 2900 nats below the
        true-parameter likelihood. See ``dev_gas3.md``.

        The window is the finite-memory form of the exponential weights: ``W``
        = ``round(N_eff)`` observations, weights ``f^(W-1-k)``. With
        ``f = 259/260`` the weights decay only 2.7x across the window, so the
        objective is close to a rolling-``W``-observation ML fit, which is the
        reference the estimator is judged against.
        """
        assert self._theta is not None
        assert len(self._win_y) == len(self._win_alpha)
        assert self.em_obs <= len(self._win_y) <= self.window
        theta = self._theta
        a0 = float(self._win_alpha[0][0])
        a1 = float(self._win_alpha[0][1])
        da0 = [0.0] * _N_PARAMS
        da1 = [0.0] * _N_PARAMS
        grad = np.zeros(_N_PARAMS)
        # Expected information only: a function of theta, so its per-state
        # constants are hoisted out of the window loop.
        info = np.zeros((_N_PARAMS, _N_PARAMS))
        # ``w_n = f**(t-n)``: the NEWEST bar of the window carries weight 1.0 and
        # the oldest ``f**(W-1)``. The window is walked oldest-first, so the
        # running weight must RISE along the loop (divide by ``f``), starting at
        # the oldest bar's ``f**(W-1)``. A previous revision carried it the other
        # way -- seeded 1.0 and multiplied by ``f`` -- which weighted the oldest
        # bar 2.7x the newest, an anti-recency objective the docstring does not
        # describe (measured to reverse the step's direction at crisis bars).
        weight = self.f ** (len(self._win_y) - 1)
        w_sum = 0.0
        _mu_t, sigma_t, gamma_t = _unpack(theta)
        inv_var = 1.0 / (sigma_t * sigma_t)
        g_off = np.array([_sigmoid(float(theta[4])), _sigmoid(float(theta[5]))])
        bern = g_off * (1.0 - g_off)
        # Warm-started at the filtered distribution preceding the window; the
        # filter's own memory (0.985^260 = 2%) makes that starting point
        # immaterial, and it is the same convention a rolling ML fit uses.
        #
        # The transition block's information is NOT accumulated here: its
        # per-bar value depends on the bar's predictive mixture through three
        # integrals over ``y`` (``_mixture_integrals``), so only the two score
        # coefficients per coordinate are carried out of the loop.
        tr_coef: list[tuple[float, float, float, float]] = []
        tr_pred: list[tuple[float, float]] = []
        for value in self._win_y:
            # Predictive state weights, evaluated BEFORE this bar's emission:
            # the expectation is over p(S_n | y_<n), which cannot see y_n.
            prev0, prev1 = a0, a1  # = P(S_{n-1} = i | y_<n)
            pred0 = prev0 * gamma_t[0, 0] + prev1 * gamma_t[1, 0]
            pred1 = prev0 * gamma_t[0, 1] + prev1 * gamma_t[1, 1]
            # This bar's exact logit scores, in closed form
            # ``s_i = (a_i N_0 + b_i N_1) / p``: a filter-feedback term (the
            # pathwise derivative ``da`` of the filtered distribution, a function
            # of ``y_<n`` only) plus the likelihood-ratio term that the state
            # attribution carries. Read from the derivatives BEFORE the step, so
            # they match the kernel's own use of them.
            occ_c01 = prev0 * bern[0]  # occ_n0 g_01 (1 - g_01)
            occ_c10 = prev1 * bern[1]  # occ_n1 g_10 (1 - g_10)
            tr_a4 = gamma_t[0, 0] * da0[4] + gamma_t[1, 0] * da1[4] - occ_c01
            tr_b4 = gamma_t[0, 1] * da0[4] + gamma_t[1, 1] * da1[4] + occ_c01
            tr_a5 = gamma_t[0, 0] * da0[5] + gamma_t[1, 0] * da1[5] + occ_c10
            tr_b5 = gamma_t[0, 1] * da0[5] + gamma_t[1, 1] * da1[5] - occ_c10
            a0, a1, da0, da1, score, _ll, p0, p1 = _step_scalars(
                a0, a1, da0, da1, value, theta
            )
            grad += weight * np.asarray(score)
            # Pair assertion: the closed form and the kernel's own score are the
            # same object (the kernel's ``p0``/``p1`` already carry the common
            # shift, so their mixture is the normaliser).
            assert abs(
                (tr_a4 * p0 + tr_b4 * p1) / (pred0 * p0 + pred1 * p1) - score[4]
            ) <= 1e-9 * max(1.0, abs(score[4]))
            assert abs(
                (tr_a5 * p0 + tr_b5 * p1) / (pred0 * p0 + pred1 * p1) - score[5]
            ) <= 1e-9 * max(1.0, abs(score[5]))
            tr_coef.append((tr_a4, tr_b4, tr_a5, tr_b5))
            tr_pred.append((pred0, pred1))
            # E[s s'] for this bar (derivation in the docstring): diagonal on
            # the emission blocks.
            # -- Rare-state guard, EMISSION coordinates only. The declared
            # information weights bar n by the PREDICTIVE weight ``P_nj``, but
            # the score that bar contributes is weighted by the FILTERED one
            # ``gamma_nj`` (``s_j = gamma_nj * d_j log N_j``). The two agree
            # while a bar is unattributed, and diverge exactly when a state is
            # rare: at ``P_n1 ~ 1e-4`` the mu/sigma information for that state
            # collapses with ``P_n1`` while a single bar that lands on it gets
            # ``gamma_n1 ~ 1`` and a full-size score -- measured on the
            # synthetic fixture as ``mu_hi = +4%/day`` in three bars. Taking the
            # larger of the two weights declares the information the bar
            # actually carries, so a lone bar cannot move a state the window
            # calls unvisited. It is inert when ``gamma <= P`` (the normal
            # case, and every well-populated state), and it is deliberately NOT
            # a floor on the realised score's outer product ``G_k^2``: that
            # variant fires on the sigma block in the crisis, where ``G_sigma^2``
            # exceeds the sigma information by construction, and it halved the
            # crisis ``sd_hi`` response (224.3 -> 209.2 bp, ``dev_est4`` §5).
            w_emit0 = pred0 if pred0 >= a0 else a0
            w_emit1 = pred1 if pred1 >= a1 else a1
            assert 0.0 <= pred0 <= w_emit0 <= 1.0
            assert 0.0 <= pred1 <= w_emit1 <= 1.0
            info[0, 0] += weight * w_emit0 * inv_var[0]
            info[1, 1] += weight * w_emit1 * inv_var[1]
            info[2, 2] += weight * w_emit0 * _E_SIGMA_SCORE_SQ
            info[3, 3] += weight * w_emit1 * _E_SIGMA_SCORE_SQ
            w_sum += weight
            weight /= self.f
        assert w_sum > 0.0
        # -- Transition block: the MARGINAL information of the exact score.
        # The old ``occ_nj g_j (1 - g_j)`` is the COMPLETE-DATA (state-
        # conditional) information: the variance of the logit score when the
        # state path is known. The recursion needs the marginal one, which is
        # smaller and, decisively, RANK ONE -- both logits are driven by the same
        # per-bar quantity ``N_1 - N_0``, so a diagonal block invents a second,
        # unconstrained direction that the data does not identify. Measured on
        # real SPY windows (``dev_est6.md`` §2): the shipped diagonal is 1.1-15x
        # the marginal block (median 2.5x), and the block's own correlation
        # ``E[s_4 s_5]/sqrt(E[s_4^2] E[s_5^2])`` is -0.40..-0.95 where the
        # diagonal structure says 0.
        coef = np.asarray(tr_coef, dtype=float)
        predn = np.asarray(tr_pred, dtype=float)
        # Same weight order as the loop above: ``w_n = f**(t-n)``, newest heaviest.
        wts = self.f ** np.arange(coef.shape[0] - 1, -1, -1, dtype=float)
        info[4, 4], info[4, 5], info[5, 5] = _transition_block(
            coef, predn, wts, _mu_t, sigma_t
        )
        info[5, 4] = info[4, 5]
        # -- MAP: the objective is the weighted posterior, not the likelihood.
        # Without the prior terms the recursion descends an unbounded objective
        # whose maximum is a one-bar spike state (see ``_prior_grad_info``).
        # ``info`` is already S = sum_n w_n E[s_n s_n'] (the un-normalised eq. 6),
        # which is the matrix the step ``S^{-1} G`` needs; scaling it again would
        # damp every coordinate by the window's weight sum (a previous revision
        # multiplied it by w_sum, i.e. steepest descent at 1/164 of the step).
        prior_grad, prior_info = _prior_grad_info(theta)
        step = _newton_step(info + prior_info, grad + prior_grad)
        # -- Per-block gain (see the module docstring for why this is a declared
        # deviation, and for the 1.58x factor the emission block carries).
        step[0:4] *= self._a_emit
        step[4:6] *= self._a_logit
        # Step caps bound ONE BAR'S MOVE, not the parameter: a single 10-sigma
        # bar cannot send sigma to zero in one update, while the recursion is
        # free to walk anywhere the (now bounded) objective leads it.
        caps = np.concatenate(
            [
                np.full(2, _MU_STEP_MAX),
                np.full(2, _LOG_SIGMA_STEP_MAX),
                np.full(2, _LOGIT_STEP_MAX),
            ]
        )
        step = np.clip(step, -caps, caps)
        self.capped_steps += (np.abs(step) >= caps).astype(int)
        theta = theta.copy()
        theta[0:2] = theta[0:2] + step[0:2]  # mu: never capped, never clipped
        theta[2:4] = np.clip(theta[2:4] + step[2:4], _LOG_SIGMA_MIN, _LOG_SIGMA_MAX)
        # Off-diagonal logits: log(1e-4) <= logit(gamma_ij) <= log(0.5).
        low = float(np.log(_LOGIT_EPS))
        high = float(np.log(_GAMMA_OFF_MAX))
        theta[4] = float(np.clip(theta[4] + step[4], low, high))
        theta[5] = float(np.clip(theta[5] + step[5], low, high))
        assert np.all(np.isfinite(theta)), "the recursion left the real line"
        self._theta = theta
        self.n_updates += 1
        return theta

    def _observation(
        self, alpha: np.ndarray, xi: np.ndarray, loglik: float
    ) -> GasObservation:
        theta = self._theta
        assert theta is not None
        mu, sigma, gamma = _unpack(theta)
        # State order is canonicalised on EVERY observation, not once at EM.
        # The recursion re-estimates the parameters daily and nothing in the
        # update keeps the variance ordering: at a fast gain the two components
        # cross (measured: 87.7% of SPY bars at the paper's (6)-(7) gain), and
        # ``hmm_gas_allocation`` reads ``filtered[0] >= filtered[1]`` as risk-on,
        # so a stale label silently flips every decision. Reporting is the only
        # place the labels are consumed, so the swap is confined here: the
        # mixture mean/variance (10)-(13) and ``forecast_moments`` are invariant
        # under a relabelling as long as theta, alpha and Gamma move together,
        # which they do below.
        if sigma[1] < sigma[0]:
            mu = mu[::-1].copy()
            sigma = sigma[::-1].copy()
            gamma = gamma[::-1, ::-1].copy()
            alpha = alpha[::-1].copy()
            theta = theta[_STATE_SWAP].copy()
        assert sigma[1] >= sigma[0]
        forecast = alpha @ gamma
        forecast = forecast / float(np.sum(forecast))
        # Eq. 12-13: per-state arithmetic moments from the log-normal.
        state_mean = np.exp(mu + 0.5 * sigma * sigma) - 1.0
        state_var = (np.exp(sigma * sigma) - 1.0) * np.exp(2.0 * mu + sigma * sigma)
        # Eq. 10-11: mixture mean/variance of the one-step forecast.
        mean = float(np.sum(forecast * state_mean))
        variance = float(
            np.sum(forecast * (state_mean * state_mean + state_var)) - mean * mean
        )
        change = float(xi[0, 1] + xi[1, 0])
        return GasObservation(
            ready=True,
            filtered=(float(alpha[0]), float(alpha[1])),
            forecast_prob=(float(forecast[0]), float(forecast[1])),
            mean=mean,
            variance=variance,
            transition=(
                (float(gamma[0, 0]), float(gamma[0, 1])),
                (float(gamma[1, 0]), float(gamma[1, 1])),
            ),
            change_prob=change,
            loglik=loglik,
            theta=tuple(float(v) for v in theta),
        )

    # -- public API --------------------------------------------------------

    def update(self, log_return: float) -> GasObservation:
        """Feed one daily log-return; return that bar's state/forecast."""
        y = float(log_return)
        assert np.isfinite(y), "log-return must be finite"
        assert abs(y) < 1.0, "daily log-return must be a return, not a level"
        self.n_seen += 1

        if self._theta is None:
            self._buffer.append(y)
            if len(self._buffer) < self.em_obs:
                return GasObservation(ready=False)
            self._initialise()
            alpha = self._alpha
            # The initialisation window is over; the joint is not needed for a
            # decision that will be taken by the next call, so a neutral xi is
            # supplied (change_prob is not used on the initialisation bar).
            zeros = np.zeros((_N_STATES, _N_STATES))
            return self._observation(alpha, zeros, 0.0)

        alpha, dalpha, score, loglik, xi = _forward_step(
            self._alpha, self._dalpha, y, self._theta
        )
        # Window bookkeeping: pair this observation with the filtered
        # distribution that preceded it, then bound the window.
        self._win_y.append(y)
        self._win_alpha.append(self._alpha.copy())
        if len(self._win_y) > self.window:
            self._win_y.pop(0)
            self._win_alpha.pop(0)
        self._alpha = alpha
        self._dalpha = dalpha
        # The paper starts the recursion with the first observation after the
        # ML fit (``n_seen == em_obs`` is the initialisation bar itself).
        if self.n_seen > self.em_obs:
            self._recursive_step()
        return self._observation(alpha, xi, loglik)


# ---------------------------------------------------------------------------
# Mean-variance allocation (paper §3), one risky asset
# ---------------------------------------------------------------------------


def forecast_moments(
    obs: GasObservation, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast mean/variance for ``k = 0 .. horizon-1`` ahead -- paper (9)-(13).

    ``alpha_{t+k|t} = alpha_t|t Gamma^k`` (Eq. 9) mixes the state-level
    arithmetic moments (Eqs. 12-13) into the mixture mean/variance of Eqs.
    (10)-(11), with the parameters held at their current (today's) values --
    exactly the paper's planning assumption (parameters are re-estimated daily,
    the plan is not). ``k = 0`` reproduces ``obs.mean`` / ``obs.variance``.
    """
    assert obs.ready, "a forecast needs an initialised filter"
    assert horizon >= 1
    mu, sigma, gamma = _unpack(np.asarray(obs.theta))
    state_mean = np.exp(mu + 0.5 * sigma * sigma) - 1.0
    state_var = (np.exp(sigma * sigma) - 1.0) * np.exp(2.0 * mu + sigma * sigma)
    alpha = np.asarray(obs.filtered, dtype=float)
    means = np.empty(horizon)
    variances = np.empty(horizon)
    for k in range(horizon):
        means[k] = float(alpha @ state_mean)
        variances[k] = (
            float(alpha @ (state_mean * state_mean + state_var)) - means[k] ** 2
        )
        alpha = alpha @ gamma
    assert np.all(np.isfinite(means)) and np.all(np.isfinite(variances))
    return means, variances


def banded_weight(
    means: np.ndarray,
    variances: np.ndarray,
    risk_aversion: float,
    rho: float,
    w_prev: float,
) -> float:
    """Paper (21)-(23)/(25): mean-variance weight minus a trading penalty band.

    The objective is the truncated-MPC one (Eq. 25) for ONE risky asset over a
    ``K``-step horizon, with ``psi = gamma h'Sigma h / V`` (Eq. 22) and the
    trading penalty ``psi(u) = rho'|u|`` (Eq. 23):

        max_{0<=w<=1}  W * sum_k mu_k - gamma * w^2 * sum_k sigma_k^2
                       - rho * |w - w_prev|

    where ``W = 1`` (a constant planned weight held across the horizon). A
    constant plan is optimal when the forecast does not anticipate a reversal:
    the L1 penalty is additive over steps, so splitting one move across two bars
    costs the same as doing it at once while earning less -- the plan therefore
    front-loads the trade. The solution is the paper's **no-trade band**:

        w_hat  = M / (2 gamma S),        M = sum mu_k,  S = sum sigma_k^2
        buy    to w_hat - rho/(2 gamma S)   iff that is above ``w_prev``
        sell   to w_hat + rho/(2 gamma S)   iff that is below ``w_prev``
        hold   otherwise

    The band ``rho / (2 gamma S)`` widens with the trading penalty and narrows
    as the horizon's risk ``S`` grows -- the paper's Figure 4 behaviour (the
    thresholds move with the parameters), and the reason ``rho = 0.02`` on a
    ONE-step problem would freeze the position entirely while the same ``rho``
    on the paper's ``K = 100`` plan leaves a usable band.
    """
    assert risk_aversion > 0.0, "risk aversion must be positive"
    assert rho >= 0.0, "trading aversion cannot be negative"
    assert 0.0 <= w_prev <= 1.0
    assert means.shape == variances.shape and means.size >= 1
    total_mean = float(np.sum(means))
    total_var = float(np.sum(variances))
    if total_var <= 0.0:
        return 0.0
    slope = 2.0 * risk_aversion * total_var
    up = (total_mean - rho) / slope
    down = (total_mean + rho) / slope
    if up > w_prev:
        weight = min(1.0, up)
    elif down < w_prev:
        weight = max(0.0, down)
    else:
        weight = w_prev
    assert 0.0 <= weight <= 1.0
    return weight


def optimal_weight(mean: float, variance: float, risk_aversion: float) -> float:
    """Long-only mean-variance weight in [0, 1] for one risky asset + cash.

    The paper's stage cost is ``psi = gamma * h' Sigma h / V`` (§3.1), so with
    one risky asset and a 0% cash leg the deterministic programme is
    ``max_w w*mu - gamma*w**2*variance`` over ``0 <= w <= 1``. Solved as a 1-D
    bounded scalar search (scipy only -- no new dependency).

    Transaction costs are deliberately NOT in this solve: the paper charges
    ``kappa * |u_t|`` inside the budget constraint, which this engine cannot
    express (costs are a fill-time price haircut). See the strategy module for
    the consequence -- no no-trade band, and hence higher turnover.
    """
    assert risk_aversion > 0.0, "risk aversion must be positive"
    assert np.isfinite(mean) and np.isfinite(variance)
    if variance <= 0.0:
        return 0.0

    def objective(w: float) -> float:
        return -(w * mean - risk_aversion * w * w * variance)

    result = minimize_scalar(
        objective, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-10}
    )
    weight = float(result.x)
    # The bounded search stops a hair inside a bound; snap so a corner solution
    # is exactly the paper's no-position / fully-invested weight.
    if weight <= 1e-8:
        weight = 0.0
    elif weight >= 1.0 - 1e-8:
        weight = 1.0
    weight = float(min(1.0, max(0.0, weight)))
    assert 0.0 <= weight <= 1.0
    return weight


__all__ = [
    "GasObservation",
    "ScoreDrivenHMM",
    "banded_weight",
    "forecast_moments",
    "optimal_weight",
]
