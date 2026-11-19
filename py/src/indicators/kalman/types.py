from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class KalmanConfig:
    """Configuration for the Kalman filter.

    The filter runs in the **log-price domain**, so every noise term below is a
    log-price variance (a relative, price-level-free quantity), never a
    price² variance.

    State vector: [log price, drift] — the classic constant-velocity model,
    with the level observed in log price.
    Transition:   x_{k+1} = F * x_k + w,  w ~ N(0, Q)
    Observation:  z_k     = H * x_k + v,  v ~ N(0, R),  z_k = log(close_k)

    process_noise (q)  log-price variance controlling how fast the state is
                       allowed to change
    measurement_noise (r) log-price variance controlling how much the filter
                       trusts each log-price observation.

    When adaptive=True, R is rescaled at each step by
    (recent_vol / baseline_vol)^2 so the filter automatically becomes
    less responsive in calm markets and more responsive in volatile ones.
    """

    process_noise: float = 1e-5
    measurement_noise: float = 1e-3
    state_dim: int = 2
    adaptive: bool = False
    vol_window: int = 20


@dataclass(frozen=True)
class FilterResult:
    """Outputs of a single Kalman-filter pass over a price series.

    The filter itself runs in log price; the level series are exponentiated
    back to price units, while the residual/drift series stay in log units.

    filtered    — a-posteriori smoothed price (best estimate given all
                  observations up to and including the current one)
    predicted   — a-priori one-step prediction (before the current
                  observation is incorporated)
    upper_ci / lower_ci — 2-sigma confidence band around the *prediction*,
                  derived from the a-priori covariance, in price units
    residuals   — log innovation: log(actual) − its one-step-ahead log
                  prediction, i.e. a relative mispricing (dimensionless),
                  useful as a mean-reversion signal
    kalman_gains — scalar Kalman gain K[0,0] at each step; spikes
                   indicate the filter is reacting to new information
    velocity    — estimated d(log price)/dt (relative drift) from the second
                  state component, in log units
    """

    filtered: pd.Series
    predicted: pd.Series
    upper_ci: pd.Series
    lower_ci: pd.Series
    residuals: pd.Series
    kalman_gains: pd.Series
    velocity: pd.Series


@dataclass(frozen=True)
class KalmanStats:
    """Summary statistics for a Kalman filter run."""

    rmse: float
    mae: float
    coverage_95: float  # fraction of actual prices within the 95 % CI
    avg_kalman_gain: float
    n_observations: int


# ---------------------------------------------------------------------------
# pairs-trading types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairsKalmanConfig:
    """Configuration for the pairs-trading Kalman filter.

    State model (random-walk [α, β]):
      α_{t+1} = α_t + w_α_t          w_α_t ~ N(0, Q)  (intercept drift)
      β_{t+1} = β_t + w_β_t          w_β_t ~ N(0, Q)  (hedge-ratio drift)
    Observation:
      log(P1_t) = α_t + β_t * log(P2_t) + v_t   v_t ~ N(0, R)

    Including an intercept α in the state is canonical in practitioner
    literature (Robot Wealth, Quantopian, Montana et al. 2009).  It
    eliminates the persistent level bias that a β-only spread carries
    when α ≠ 0, and removes the need for EWMA post-processing: the
    Kalman innovation is genuinely mean-zero by construction.

    process_noise (q)  — how fast the state [α, β] is allowed to drift
    measurement_noise (r) — observation noise variance
    mean_halflife — OLS warm-start window size (repurposed from former
                    EWMA halflife; warm-start uses this many initial
                    observations to estimate [α₀, β₀] and P₀ via OLS)
    adaptive — when True, rescale R by (recent_vol / baseline_vol)^2
               so the filter auto-adjusts to volatility regimes
    vol_window — rolling window length for adaptive volatility calc
    """

    process_noise: float = 1e-4
    measurement_noise: float = 1e-3
    mean_halflife: int = 50
    adaptive: bool = False
    vol_window: int = 20


@dataclass(frozen=True)
class PairsKalmanResult:
    """Output of a pairs-trading Kalman filter over a full series.

    Two-state [α, β] model with the Kalman's own standardized innovation
    as the sole trading signal.

    alpha        — time-varying intercept (dimensionless, log-space)
    beta         — time-varying hedge ratio (dimensionless, log-space
                   elasticity: % move ratio)
    spread       — innovation: log(P1) - α_pred - β_pred·log(P2);
                   this is the raw mispricing, mean-zero by construction
                   when α is included in the state
    t_stat       — THE trading signal: spread / √S, the Kalman's
                   standardized innovation.  Naturally downweights
                   observations when the filter is uncertain.
    innovation_S — Kalman innovation covariance S = H·P·H' + R at each
                   step; large values mean the filter is uncertain about
                   the current observation.
    """

    alpha: pd.Series
    beta: pd.Series
    spread: pd.Series
    t_stat: pd.Series
    innovation_S: pd.Series
