"""
Functional Kalman filter for price smoothing and trend estimation.

Uses filterpy.kalman.KalmanFilter internally with a constant-velocity
state model on **log price**: state = [log price, drift per bar].

All public helpers are plain functions — no custom classes.

Forecast horizon note
---------------------
The Kalman filter naturally produces one-step-ahead predictions (the
a-priori state estimate before the current observation is folded in).
Multi-step forecasts are straightforward: propagate x = F @ x with no
measurement update for N steps, and let the covariance grow as
P = F @ P @ F.T + Q each step.  A future version could expose this via
a `forecast(steps: int) -> pd.DataFrame` helper when there's a concrete
use-case for it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter as _KF

from src.indicators.kalman.types import (
    KalmanConfig,
    FilterResult,
    KalmanStats,
    PairsKalmanConfig,
    PairsKalmanResult,
)


# ---------------------------------------------------------------------------
# core filter
# ---------------------------------------------------------------------------


def run_filter(
    prices: pd.Series,
    config: KalmanConfig,
) -> FilterResult:
    """Run the Kalman filter over *prices* and return structured outputs.

    **The filter runs in the log-price domain** (B3): ``measurement_noise`` and
    ``process_noise`` are log-price variances, and the residual is a relative
    mispricing. This is the same convention the pairs filter uses, and it is
    what makes the signal scale-free — an absolute price² R gave a $586 asset
    an innovation sigma of ~$0.047 against dollar residuals, so the
    standardised residual ran to tens/hundreds instead of O(1). ``filtered`` /
    ``predicted`` / the CI are exponentiated back to price units; ``residuals``
    and ``velocity`` stay in log units.

    Parameters
    ----------
    prices : pd.Series
        Close prices indexed by timestamp.
    config : KalmanConfig
        Filter hyper-parameters (process noise, measurement noise, …).

    Returns
    -------
    FilterResult with filtered, predicted, CI, residuals, gains, velocity.
    """
    # Build filter matrices, in the log-price domain.
    dt = 1.0  # each row is one time-step
    F = np.array([[1, dt], [0, 1]], dtype=float)  # state transition
    H = np.array([[1, 0]], dtype=float)  # observation matrix
    Q_base = _build_Q(config.process_noise, dt)
    R_base = np.array([[config.measurement_noise]], dtype=float)
    log_prices = np.log(prices.to_numpy(dtype=float))

    kf = _KF(dim_x=2, dim_z=1)
    kf.F = F
    kf.H = H
    kf.Q = Q_base
    kf.R = R_base

    # Initialise state to first observation, zero velocity
    kf.x = np.array([[log_prices[0]], [0.0]])
    # Wide initial uncertainty — let the filter converge from the data
    # rather than assuming we know the starting point precisely.
    kf.P = np.diag([1.0, 1.0])

    # Pre-compute returns for adaptive R (log returns = log-price units, so the
    # scale is now consistent with R)
    returns = np.log(prices / prices.shift(1))
    baseline_vol = 0.0
    if config.adaptive:
        baseline_vol = returns.rolling(config.vol_window).std().dropna().mean()
        baseline_vol = max(baseline_vol, 1e-12)

    n = len(prices)
    filtered = np.empty(n)
    predicted = np.empty(n)  # price units
    predicted_log = np.empty(n)  # filter units (log price)
    cov_diag = np.empty(n)  # post-update P[0,0]
    pre_cov = np.empty(n)  # pre-update P[0,0] for prediction CI
    residuals = np.empty(n)
    kalman_gains = np.empty(n)
    velocity = np.empty(n)

    for k, z in enumerate(log_prices):
        # --- adaptive measurement noise ---
        if config.adaptive and k >= config.vol_window:
            recent_vol = returns.iloc[k - config.vol_window + 1 : k + 1].std()
            if not np.isnan(recent_vol) and recent_vol > 0:
                scale = (recent_vol / baseline_vol) ** 2
                kf.R = R_base * scale
            else:
                kf.R = R_base

        # predict (a-priori)
        kf.predict()
        predicted_log[k] = kf.x[0, 0]
        predicted[k] = np.exp(kf.x[0, 0])

        # store pre-update covariance for prediction CI (log-variance)
        p00 = kf.P[0, 0]
        pre_cov[k] = p00

        # update (a-posteriori)
        kf.update(np.array([[z]]))
        filtered[k] = np.exp(kf.x[0, 0])
        velocity[k] = kf.x[1, 0]
        cov_diag[k] = kf.P[0, 0]  # post-update, tighter
        kalman_gains[k] = kf.K[0, 0]
        # Innovation residual: log price minus its one-step-ahead prediction.
        # This is the standard definition — not (z − filtered), which is
        # the a-posteriori error and is always tiny.
        residuals[k] = z - predicted_log[k]

    # 95 % CI around the *prediction* using pre-update covariance, mapped back
    # to price units. This answers "where will price be?" rather than "where is
    # it now?" and is the right band for breakout / mean-reversion signals.
    upper_ci = np.exp(predicted_log + 2.0 * np.sqrt(pre_cov))
    lower_ci = np.exp(predicted_log - 2.0 * np.sqrt(pre_cov))

    idx = prices.index
    return FilterResult(
        filtered=pd.Series(filtered, index=idx, name="filtered"),
        predicted=pd.Series(predicted, index=idx, name="predicted"),
        upper_ci=pd.Series(upper_ci, index=idx, name="upper_ci"),
        lower_ci=pd.Series(lower_ci, index=idx, name="lower_ci"),
        residuals=pd.Series(residuals, index=idx, name="residual"),
        kalman_gains=pd.Series(kalman_gains, index=idx, name="kalman_gain"),
        velocity=pd.Series(velocity, index=idx, name="velocity"),
    )


def compute_stats(prices: pd.Series, result: FilterResult) -> KalmanStats:
    """Compute summary statistics for a filter run.

    ``rmse``/``mae`` are expressed in the residual's units — log price, i.e. a
    relative error — and ``coverage_95`` is the share of closes inside the
    price-unit CI, so it stays a plain count.
    """
    resid = result.residuals
    n = len(resid)
    rmse = float(np.sqrt((resid**2).mean()))
    mae = float(resid.abs().mean())

    inside = ((prices >= result.lower_ci) & (prices <= result.upper_ci)).sum()
    coverage_95 = float(inside / n) if n else 0.0

    avg_gain = float(result.kalman_gains.mean())

    return KalmanStats(
        rmse=rmse,
        mae=mae,
        coverage_95=coverage_95,
        avg_kalman_gain=avg_gain,
        n_observations=n,
    )


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _build_Q(q: float, dt: float) -> np.ndarray:
    """Process-noise covariance for the constant-velocity model.

    Q = q * [[dt³/3, dt²/2],
             [dt²/2, dt   ]]

    See Bar-Shalom & Li, "Estimation with Applications to Tracking".
    Currently called with dt=1.0 (each candle is one time-step), so the
    expression collapses to:
        Q = process_noise * [[1/3, 0.5],
                             [0.5, 1.0]]
    The full discretization is kept for correctness when irregular or
    intradaily intervals are passed in a future version.
    """
    dt2 = dt * dt
    dt3 = dt2 * dt
    return q * np.array(
        [[dt3 / 3, dt2 / 2], [dt2 / 2, dt]],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# pairs-trading Kalman filter
# ---------------------------------------------------------------------------


def _warm_start_ols(
    log_p1: np.ndarray, log_p2: np.ndarray, window: int = 50
) -> tuple[float, float, np.ndarray]:
    """Estimate [α, β] and P₀ from an initial OLS window of log-prices.

    Fits:   log(P1) = α + β·log(P2) + ε

    The state covariance P₀ is derived from the OLS parameter covariance
    σ²_resid · (X'X)^(-1), so the Kalman starts with uncertainty that
    matches the regression quality rather than a generic wide prior.

    Falls back to α₀=0, β₀=1.0, P₀=I when the window is too short for a
    reliable regression (w < 3).

    Parameters
    ----------
    log_p1 : np.ndarray
        Log-prices of symbol 1.
    log_p2 : np.ndarray
        Log-prices of symbol 2.
    window : int
        Number of initial observations to use for the OLS fit.

    Returns
    -------
    tuple[float, float, np.ndarray]
        (α₀, β₀, P₀) — initial state and 2×2 covariance.
    """
    n = len(log_p1)
    w = min(window, n)

    if w < 3:
        # Too few points for a meaningful regression — fall back to defaults
        return 0.0, 1.0, np.eye(2, dtype=float)

    X = np.column_stack([np.ones(w, dtype=float), log_p2[:w]])
    y = log_p1[:w]

    try:
        theta = np.linalg.lstsq(X, y, rcond=None)[
            0
        ]  # [α, β] — column order: [ones, log_p2]
    except np.linalg.LinAlgError:
        return 0.0, 1.0, np.eye(2, dtype=float)

    alpha0 = float(theta[0])
    beta0 = float(theta[1])

    # Parameter covariance = σ²_resid · (X'X)^(-1)
    resid = y - X @ theta
    sigma2 = float(resid.var(ddof=2)) if w > 2 else 1.0
    sigma2 = max(sigma2, 1e-12)

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        # Collinear X'X — fall back to diagonal P
        return alpha0, beta0, np.diag([sigma2, sigma2])

    P_diag = np.diag(XtX_inv) * sigma2
    P = np.diag(np.maximum(P_diag, 1e-6))
    return alpha0, beta0, P


def run_pairs_kalman(
    prices1: pd.Series,
    prices2: pd.Series,
    config: PairsKalmanConfig | None = None,
) -> PairsKalmanResult:
    """Run a pairs-trading Kalman filter over two price series.

    Uses the canonical two-state [α, β] model from practitioner literature
    (Robot Wealth, Quantopian, Montana et al. 2009).

    State model (random walk on both components):
      [α_{t+1}] = [1 0] [α_t] + w_t     w_t ~ N(0, Q)
      [β_{t+1}]   [0 1] [β_t]
    Observation:
      log(P1_t) = α_t + β_t * log(P2_t) + v_t    v_t ~ N(0, R)

    The innovation (spread) is mean-zero by construction when α is
    included in the state.  The standardized innovation t_stat =
    spread / √S is the complete trading signal — no EWMA needed.

    Warm-start: OLS over the initial ``mean_halflife`` observations
    provides the initial state and a data-driven P₀.  This eliminates
    burn-in so the filter is signalling-usable from the first step.

    Parameters
    ----------
    prices1 : pd.Series
        Close prices for symbol 1, timestamp-indexed.
    prices2 : pd.Series
        Close prices for symbol 2, timestamp-indexed.
    config : PairsKalmanConfig, optional
        Filter hyper-parameters.  Defaults to PairsKalmanConfig().

    Returns
    -------
    PairsKalmanResult with alpha, beta, spread, t_stat, innovation_S.
    """
    if config is None:
        cfg = PairsKalmanConfig()
    else:
        cfg = config

    # Align on inner join
    aligned = pd.concat([prices1.rename("p1"), prices2.rename("p2")], axis=1).dropna()
    if aligned.empty:
        empty_idx = pd.DatetimeIndex([])
        return _empty_pairs_result(empty_idx)

    p1 = aligned["p1"]
    p2 = aligned["p2"]
    n = len(p1)
    idx = p1.index

    log_p2 = np.log(p2.values.astype(float))
    log_p1 = np.log(p1.values.astype(float))

    # Build 2-state Kalman filter: dim_x=2 (α, β), dim_z=1
    kf = _KF(dim_x=2, dim_z=1)
    kf.F = np.eye(2, dtype=float)  # random walk on both states
    kf.Q = np.eye(2, dtype=float) * cfg.process_noise

    R_base = np.array([[cfg.measurement_noise]], dtype=float)
    kf.R = R_base

    # Warm-start from OLS over initial window
    alpha0, beta0, P0 = _warm_start_ols(log_p1, log_p2, cfg.mean_halflife)
    kf.x = np.array([[alpha0], [beta0]], dtype=float)
    kf.P = P0

    # Pre-compute adaptive R baseline if needed
    baseline_vol = 0.0
    if cfg.adaptive:
        returns = np.log(p2.values[1:] / p2.values[:-1])
        if len(returns) >= cfg.vol_window:
            bv = pd.Series(returns).rolling(cfg.vol_window).std().dropna().mean()
            baseline_vol = max(float(bv), 1e-12)

    alpha_arr = np.empty(n)
    beta_arr = np.empty(n)
    spread_arr = np.empty(n)
    innov_S_arr = np.empty(n)

    for t in range(n):
        logp2 = log_p2[t]
        logp1 = log_p1[t]

        # Observation matrix: H = [[1, log(P2_t)]] (1×2)
        kf.H = np.array([[1.0, logp2]], dtype=float)

        # --- adaptive measurement noise ---
        if cfg.adaptive and t >= cfg.vol_window:
            recent_ret = np.log(
                p2.values[t - cfg.vol_window + 1 : t + 1]
                / p2.values[t - cfg.vol_window : t]
            )
            recent_vol = float(np.std(recent_ret))
            if not np.isnan(recent_vol) and recent_vol > 0 and baseline_vol > 0:
                scale = (recent_vol / baseline_vol) ** 2
                kf.R = R_base * scale
            else:
                kf.R = R_base

        # Predict
        kf.predict()
        beta_pred = kf.x[1, 0]  # β prior (state element 1)
        alpha_pred = kf.x[0, 0]  # α prior (state element 0)

        # Innovation covariance S = H·P·H' + R  (scalar)
        S = float((kf.H @ kf.P @ kf.H.T + kf.R)[0, 0])

        # Innovation = spread = log(P1) - (α_pred + β_pred·log(P2))
        innovation = float(logp1 - (alpha_pred + beta_pred * logp2))

        # Update
        kf.update(np.array([[logp1]]))

        alpha_arr[t] = kf.x[0, 0]
        beta_arr[t] = kf.x[1, 0]
        spread_arr[t] = innovation
        innov_S_arr[t] = S

    # t-statistic = spread / √S  (no EWMA post-processing needed)
    sqrt_S = np.sqrt(np.maximum(innov_S_arr, 1e-24))
    t_stat_arr = spread_arr / sqrt_S

    return PairsKalmanResult(
        alpha=pd.Series(alpha_arr, index=idx, name="alpha"),
        beta=pd.Series(beta_arr, index=idx, name="beta"),
        spread=pd.Series(spread_arr, index=idx, name="spread"),
        t_stat=pd.Series(t_stat_arr, index=idx, name="t_stat"),
        innovation_S=pd.Series(innov_S_arr, index=idx, name="innovation_S"),
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _empty_pairs_result(idx: pd.DatetimeIndex) -> PairsKalmanResult:
    """Return an empty PairsKalmanResult with the given index."""
    empty_s = pd.Series([], dtype=float, index=idx)
    return PairsKalmanResult(
        alpha=empty_s.rename("alpha"),
        beta=empty_s.rename("beta"),
        spread=empty_s.rename("spread"),
        t_stat=empty_s.rename("t_stat"),
        innovation_S=empty_s.rename("innovation_S"),
    )
