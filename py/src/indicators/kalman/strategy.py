"""Strategy-facing online pairs Kalman owner.

``OnlinePairs`` is the strategy-level owner of the pairs Kalman filter — the
successor to the removed engine ``model_updater`` channel. Construct one in the
strategy's ``reset_global()`` (or lazily in the DSL's ``ctx.shared``), hold it
in state, and call ``observe()`` once per candle. It reads both legs' closes
from ``CandleStore`` (cursor-safe, no look-ahead), OLS-warm-starts the filter,
updates it, and tracks the rolling z-score of the Kalman spread (the tradable
signal, ~±2).

Usage (inside a strategy's ``on_candle``)::

    GLOBAL = {"kf": OnlinePairs(process_noise=1e-4, measurement_noise=1e-3)}

    def reset_global() -> None:
        global GLOBAL
        GLOBAL = {"kf": OnlinePairs(process_noise=1e-4, measurement_noise=1e-3)}

    def on_candle(state, candle, params):
        result = GLOBAL["kf"].observe(state, "SPY", "QQQ", candle.interval or "1h")
        if not result.ready:
            return []  # warmup
        if abs(result.z_score) > params.z_entry:
            ...  # enter
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from src.indicators.kalman.online import PairsKalmanOnline
from src.indicators.kalman.types import PairsKalmanConfig
from src.bt.state import BacktestState


@dataclass(frozen=True)
class OnlinePairsResult:
    """One observation's Kalman outputs."""

    z_score: float | None  # rolling z of Kalman spread (tradable ~±2)
    beta: float | None
    alpha: float | None
    n_steps: int
    ready: bool  # False during warmup (insufficient aligned history)


@dataclass(frozen=True)
class KalmanOffsetResult:
    """One symbol's Kalman mispricing vs a benchmark (single-leg fade signal).

    ``t_stat`` is the tradable signal: the Kalman's **standardized innovation**
    ``t = spread / sqrt(S)`` where ``spread = log(sym) - (alpha + beta*log(bench))``
    is the log-space innovation (mean-zero by construction, intercept in state)
    and ``sqrt(S)`` is the innovation standard deviation the filter maintains
    **online** (S = H·P·H' + R). Because both the residual and its scale are
    produced in-line at each step from past + current observations only, this is
    a true online signal — no rolling window over a fixed history.

    Interpretation: a large positive ``t_stat`` means the symbol is stretched
    *above* its Kalman relationship to the benchmark (overextended — fade-short
    when ``t_stat > z_entry``); a large negative one stretched below
    (under-extended — fade-long)."""

    t_stat: float | None
    beta: float | None
    alpha: float | None
    spread: float | None
    n_steps: int
    ready: bool  # False during warmup (insufficient aligned history)


def _rolling_zscore(deq: deque[float], current: float, window: int) -> float:
    """Append ``current`` and return its rolling z-score within ``deq``."""
    deq.append(current)
    if len(deq) < max(window, 3):
        return 0.0
    vals = list(deq)[-window:]
    mean = np.mean(vals)
    std = np.std(vals, ddof=1)
    if std < 1e-12:
        return 0.0
    return float((current - mean) / std)


class OnlinePairs:
    """Online pairs Kalman + rolling z-score, owned by a strategy.

    Parameters mirror the removed engine ``model_updater`` factory. Re-constructing
    in ``reset_global()`` (or fresh in ``ctx.shared``) rebuilds filter and history
    for a clean IS/OOS split.
    """

    def __init__(
        self,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-3,
        ols_warmup: int = 50,
        adaptive: bool = True,
        vol_window: int = 20,
        z_window: int = 20,
        warmup_bars: int = 150,
    ) -> None:
        self._ols_warmup = ols_warmup
        self._z_window = z_window
        self._warmup_bars = warmup_bars

        kf_cfg = PairsKalmanConfig(
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            mean_halflife=ols_warmup,
            adaptive=adaptive,
            vol_window=vol_window,
        )
        self._kf = PairsKalmanOnline(config=kf_cfg)
        self._spread_history: deque[float] = deque(maxlen=max(z_window, warmup_bars))

    @property
    def kf(self) -> PairsKalmanOnline:
        """Underlying filter (for diagnostics / custom warm-start)."""
        return self._kf

    def _ols_warmstart(
        self, closes1: pd.Series, closes2: pd.Series, window: int
    ) -> None:
        if self._kf.n_steps >= window:
            return
        n = min(len(closes1), len(closes2), window)
        if n < 3:
            return
        lp1 = np.log(closes1.iloc[-n:].values)
        lp2 = np.log(closes2.iloc[-n:].values)
        self._kf.init_from_ols(lp1, lp2)

    def observe(
        self,
        state: BacktestState,
        s1: str,
        s2: str,
        interval: str,
    ) -> OnlinePairsResult:
        """Feed the pair's latest closes to the Kalman filter.

        Returns a result with ``ready=False`` until sufficient aligned history
        exists (warmup) or either leg is missing. ``ready`` becomes True after
        warmup, even if the rolling z-score is still 0.
        """
        result = OnlinePairsResult(None, None, None, self._kf.n_steps, False)

        df1 = state.candles.get((s1, interval))
        df2 = state.candles.get((s2, interval))
        if df1 is None or df2 is None or len(df1) < 2 or len(df2) < 2:
            return result

        c1 = cast(pd.Series, df1["close"])
        c2 = cast(pd.Series, df2["close"])
        aligned = pd.concat([c1.rename("a"), c2.rename("b")], axis=1).dropna()
        if len(aligned) < self._warmup_bars:
            return result

        # OLS warm-start the filter on the initial window if not yet fitted.
        self._ols_warmstart(
            cast(pd.Series, aligned["a"]),
            cast(pd.Series, aligned["b"]),
            self._ols_warmup,
        )
        if self._kf.n_steps < 3:
            return result

        log_p1 = float(np.log(aligned["a"].iloc[-1]))
        log_p2 = float(np.log(aligned["b"].iloc[-1]))
        self._kf.update(log_p1, log_p2)

        spread = self._kf.spread
        z = _rolling_zscore(self._spread_history, spread, self._z_window)

        return OnlinePairsResult(
            z_score=z,
            beta=float(self._kf.beta),
            alpha=float(self._kf.alpha),
            n_steps=self._kf.n_steps,
            ready=True,
        )


@dataclass(frozen=True)
class OnlineLevelResult:
    """One observation's constant-velocity Kalman outputs.

    ``residual``/``sigma``/``velocity`` are log-price quantities (the filter's
    domain); ``predicted`` is the equivalent price level."""

    residual: float | None  # log(price) - predicted log level (innovation)
    sigma: float | None  # sqrt(V): realised scale of that residual
    z_stat: float | None  # residual / sigma — the online fade signal
    predicted: float | None  # a-priori price prediction, exp(stated log level)
    velocity: float | None  # estimated d(log price)/dt (state[1])
    n_steps: int
    ready: bool


# EWMA weight for the recursive squared-residual mean that standardises
# ``z_stat``. A hard-coded structural constant (~60-bar memory), not a knob:
# the scale estimate only has to track volatility, not be tuned per symbol.
_Z_VAR_ALPHA = 1.0 / 60.0
# Floor for that scale, in log units (~a 1e-6 % move per bar). Keeps the
# division defined on a dead-flat series.
_Z_VAR_FLOOR = 1e-12


class OnlineLevel:
    """Online constant-velocity Kalman for one symbol — scale-free (B3).

    Matches ``src.indicators.kalman.pure.run_filter`` numerically but steps one
    observation per :meth:`observe` call instead of sweeping a full Series.

    **The filter runs in the log-price domain**, the same convention the pairs
    filter already uses (``run_pairs_kalman`` / ``OnlinePairs``), so the
    residual is a *relative* mispricing: invariant to the price level, to a
    split, and comparable across a $5 and a $586 asset. Consequences:

    * ``measurement_noise`` (r) is a **log-price variance**, not a dollar
      variance: 1e-3 ≈ 3.2% per-bar observation noise. Reading it as price²
      was the original defect — on a $586 asset it made sigma ≈ $0.047 while
      residuals were dollars, so ``z_stat`` had sd 75-245 and any ``z_entry=2``
      gate passed on 95-99% of bars.
    * ``z_stat`` standardises the residual by its **realised** scale (a
      recursive mean of squared log-residuals), not by the model-implied
      ``S = H·P·H' + R``. The model S is pinned by the configured q/r and does
      not track the asset's volatility: with the shipped q=1e-4/r=1e-3 it sits
      ~4x the realised sigma, so standardising by it alone would leave
      ``z_entry=2`` passing ~0% of bars — inert in the other direction.

    Consequence for the shipped ``z_entry=2.0`` / ``z_exit=0.5``: ``|z_stat|``
    has sd ≈ 1 and |z| > 2 on ~5% of real 1d bars (measured 5.0-5.9% for
    AAPL/MSFT/NVDA/AMZN/TSLA/SPY), so the entry is a genuine ~2-sigma event and
    the exit a genuine 0.5-sigma convergence — the textbook reading of those
    defaults. ``process_noise`` (q) now only sets smoothing: lower q = smoother
    level = more persistent residuals (stronger fade signal, slower to adapt).
    """

    def __init__(
        self,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-3,
        warmup_bars: int = 150,
    ) -> None:
        self._q = process_noise
        self._r = measurement_noise
        self._warmup_bars = warmup_bars

        # State [log price, drift per bar]; F/H/Q/R per run_filter (dt=1).
        self._x: np.ndarray = np.array([0.0, 0.0])
        self._P: np.ndarray = np.eye(2)
        self._n: int = 0
        self._residual: float = 0.0
        self._var_resid: float = 0.0  # recursive mean of squared log-residuals

    @property
    def residual(self) -> float:
        return self._residual

    @property
    def sigma(self) -> float:
        # Standardisation scale: sqrt of the recursive squared-residual mean.
        return float(np.sqrt(max(self._var_resid, _Z_VAR_FLOOR)))

    @property
    def predicted(self) -> float:
        return float(np.exp(self._x[0]))

    @property
    def velocity(self) -> float:
        return float(self._x[1])

    @property
    def n_steps(self) -> int:
        return self._n

    def _predict(self) -> None:
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self._x = F @ self._x
        Q = self._q * np.array([[1.0 / 3.0, 0.5], [0.5, 1.0]])
        self._P = F @ self._P @ F.T + Q

    def _update(self, z: float) -> None:
        H = np.array([[1.0, 0.0]])
        R = self._r
        S = float((H @ self._P @ H.T)[0, 0] + R)
        K = (self._P @ H.T) / max(S, 1e-24)
        innov = z - float((H @ self._x)[0])
        self._x = self._x + (K * innov).ravel()
        eye = np.eye(2)
        self._P = (eye - K @ H) @ self._P
        self._P = (self._P + self._P.T) / 2.0

    def observe(self, price: float) -> OnlineLevelResult:
        """Feed the latest close. Returns the online fade signal."""
        if price <= 0:
            return OnlineLevelResult(None, None, None, None, None, self._n, False)

        # Log domain (see the class docstring): a non-positive price is the only
        # domain error, and every quantity below is relative to the price level.
        log_p = float(np.log(price))

        if self._n == 0:
            self._x = np.array([log_p, 0.0])
            self._P = np.diag([1.0, 1.0])
            self._n = 1
            self._residual = 0.0
            # Mirror the batch reference, which predicts+updates the first bar
            # too: its innovation is zero by construction (the state starts at
            # the observation) but P shrinks the same way, so both paths then
            # evolve identically instead of drifting apart by one update.
            self._predict()
            self._update(log_p)
            return OnlineLevelResult(
                0.0, self.sigma, 0.0, price, 0.0, self._n, self._n >= self._warmup_bars
            )

        self._predict()
        self._residual = float(log_p - float(self._x[0]))
        zstat = self._standardize(self._residual)
        self._update(log_p)
        self._n += 1
        ready = self._n >= self._warmup_bars
        return OnlineLevelResult(
            residual=self._residual,
            sigma=self.sigma,
            z_stat=zstat,
            predicted=self.predicted,
            velocity=self.velocity,
            n_steps=self._n,
            ready=ready,
        )

    def _standardize(self, residual: float) -> float:
        """Residual / sqrt(recursive squared-residual mean), then fold it in.

        The scale is taken from the residuals *before* this one, so ``z_stat``
        is a one-step-ahead quantity and carries no look-ahead. The first
        residual only seeds the scale (no estimate exists yet) and is reported
        as 0. The EWMA is recursive with a slow ``_Z_VAR_ALPHA``, so the seed
        still carries a small residual weight at ``warmup_bars`` — the scale is
        an approximation there, not a converged estimate.
        """
        if self._var_resid <= 0.0:
            self._var_resid = max(residual * residual, _Z_VAR_FLOOR)
            return 0.0
        z = residual / self.sigma
        self._var_resid += _Z_VAR_ALPHA * (residual * residual - self._var_resid)
        return z
