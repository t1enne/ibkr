"""Chan (2013) Example 3.3 pairs filter — the published recursion, as published.

This is a deliberate, standalone reproduction, NOT a variant of
:class:`src.indicators.kalman.strategy.OnlinePairs`.  That module is a different
object: it runs in LOG prices, warm-starts ``[beta, alpha]`` by one-shot OLS, and
uses an isotropic random-walk ``Q`` with hand-set ``q``/``r``.  The published
example needs none of that — no warm start, no lookback, prices not logs, and a
zero prior covariance on the first bar.

Reference items are the numbered rules of ``reference.md`` §2 (Chan 2013 ch.3
Example 3.3, reconstructed there from three secondary sources):

  (2) Prices, not logs.  ``y_t`` = leg-1 close, ``x_t`` = leg-2 close, and the
      observation matrix is ``F_t = [x_t, 1]``, so the intercept ``alpha`` is a
      state alongside the hedge ratio ``beta``.
  (3) ``theta_t = theta_{t-1} + w_t``, ``w ~ N(0, Vw)``, ``Vw = delta/(1-delta) * I2``.
  (4) ``y_t = F_t theta_{t-1} + v_t``, ``v ~ N(0, Ve)`` with ``Ve`` in ABSOLUTE
      price-squared units — not a return variance, not rescaled by price level.
  (5) ``theta_0 = (0, 0)``, ``P_0 = 0``; on the FIRST bar the prior covariance is
      ``R = 0`` (not ``Vw``), thereafter ``R = P + Vw``.
  (6) No warm start and no lookback: the only guard published is "trade if
      days > 1", exposed here as ``ready``.
  (7) Predict: ``yhat_t = F_t theta_{t-1}``; innovation ``e_t = y_t - yhat_t``;
      innovation variance ``Q_t = F_t R F_t' + Ve`` and ``S_t = sqrt(Q_t)``.
  (8) Update: ``K_t = R F_t' / Q_t``; ``theta_t = theta_{t-1} + K_t e_t``;
      ``P_t = R - K_t F_t R``.

Two EXPERIMENT knobs exist so the same code can also express the repaired local
variant — and so the consequence of each deviation can be MEASURED.  Both sit off
the default path, which is the reference verbatim:

  ``price_domain="log"``
      Run the recursion on log prices (the domain of our own repaired filter)
      instead of the reference's prices.
  ``ve_mode="relative"``
      Scale ``Ve`` by the observed leg's price level squared — the defect class
      we repaired in our own filter (fixed price-unit noise => a ``z`` that is
      not scale free).  This knob exists to measure the consequence, NOT to fix
      the reference.  Squaring a log price level is meaningless, so
      ``price_domain="log"`` together with ``ve_mode="relative"`` is refused.

Except for those two knobs, the recursion below is item-for-item the reference.
The module outputs the filter state only: it emits no signal, size or backtest.
Trading rules (reference items 9-12) are the caller's, and item 10 (the exit) is
the one item this filter cannot pin down — the published reproductions disagree
on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

PriceDomain = Literal["absolute", "log"]
VeMode = Literal["absolute", "relative"]

# Reference values, items (3) and (4).
DELTA: float = 1e-4
VE: float = 1e-3


@dataclass(frozen=True)
class ChanPairsConfig:
    """Reference parameters (items 3-4) plus the two off-default experiment knobs.

    delta — random-walk rate of ``[beta, alpha]``; ``Vw = delta/(1-delta) * I2``.
    ve    — observation-noise variance.  Absolute price^2 under the default
            knobs; a log-price variance under ``price_domain="log"``.
    """

    delta: float = DELTA
    ve: float = VE
    price_domain: PriceDomain = "absolute"
    ve_mode: VeMode = "absolute"

    def __post_init__(self) -> None:
        if not 0.0 < self.delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {self.delta!r}")
        if not self.ve > 0.0:
            raise ValueError(f"ve must be positive, got {self.ve!r}")
        if self.price_domain == "log" and self.ve_mode == "relative":
            raise ValueError(
                "ve_mode='relative' squares a price level; it is only meaningful "
                "with price_domain='absolute'"
            )


@dataclass(frozen=True)
class ChanPairsState:
    """Filter state: ``theta = [beta, alpha]`` and the symmetric 2x2 ``P``.

    ``P`` is held as its three independent entries so the update stays exact
    (``P`` is symmetric by construction: ``R - K F R`` is symmetric when ``R`` is).
    """

    beta: float = 0.0
    alpha: float = 0.0
    p_beta_beta: float = 0.0
    p_beta_alpha: float = 0.0
    p_alpha_alpha: float = 0.0
    bars: int = 0

    @property
    def theta(self) -> tuple[float, float]:
        return (self.beta, self.alpha)

    @property
    def p(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            (self.p_beta_beta, self.p_beta_alpha),
            (self.p_beta_alpha, self.p_alpha_alpha),
        )


@dataclass(frozen=True)
class ChanPairsStep:
    """One bar of output: the innovation, its standardisation, and the state.

    innovation — ``e_t = y_t - F_t theta_{t-1}`` (item 7), in the working domain.
    s          — ``S_t = sqrt(Q_t)`` (item 7), the innovation's prediction sd.
    z          — ``e_t / S_t``, the reference's own standardisation; item 9
                 trades this against a band of width ``k * S_t`` (published k = 1).
    ready      — item 6: True once a previous bar exists (published "days > 1").
    """

    beta: float
    alpha: float
    innovation: float
    s: float
    z: float
    ready: bool
    bars: int


def _working_values(
    config: ChanPairsConfig, x_price: float, y_price: float
) -> tuple[float, float, float]:
    """Map a raw price pair to the recursion's ``(x, y)`` and the effective ``Ve``."""
    if not (math.isfinite(x_price) and math.isfinite(y_price)):
        raise ValueError(f"prices must be finite, got x={x_price!r} y={y_price!r}")
    if x_price <= 0.0 or y_price <= 0.0:
        raise ValueError(f"prices must be positive, got x={x_price!r} y={y_price!r}")

    if config.price_domain == "log":
        x, y = math.log(x_price), math.log(y_price)
    else:
        x, y = x_price, y_price

    # Item 4: Ve is absolute price^2 by default.  'relative' reproduces the
    # repaired defect: the same number scaled by the observed leg's level^2.
    ve_eff = (
        config.ve * y_price * y_price if config.ve_mode == "relative" else config.ve
    )
    return x, y, ve_eff


def step(
    state: ChanPairsState,
    config: ChanPairsConfig,
    x_price: float,
    y_price: float,
) -> tuple[ChanPairsState, ChanPairsStep]:
    """Advance the reference recursion by one bar.  Pure: state in, state out."""
    x, y, ve_eff = _working_values(config, x_price, y_price)
    vw = config.delta / (1.0 - config.delta)

    # Item 5: R = 0 on the first bar, R = P + Vw thereafter.
    if state.bars == 0:
        r_bb = r_ba = r_aa = 0.0
    else:
        r_bb = state.p_beta_beta + vw
        r_ba = state.p_beta_alpha
        r_aa = state.p_alpha_alpha + vw

    # Item 7: prediction, innovation, innovation variance.
    innovation = y - (state.beta * x + state.alpha)
    q = r_bb * x * x + 2.0 * r_ba * x + r_aa + ve_eff
    assert q > 0.0, "innovation variance must be positive (ve > 0 guarantees it)"
    s = math.sqrt(q)

    # Item 8: K = R F'/Q with F = [x, 1], and P = R - K F R written as the
    # rank-one form R - (R F')(F R)/Q (F R is the transpose of R F', R is symmetric).
    u_beta = r_bb * x + r_ba
    u_alpha = r_ba * x + r_aa
    beta = state.beta + u_beta * innovation / q
    alpha = state.alpha + u_alpha * innovation / q
    p_bb = r_bb - u_beta * u_beta / q
    p_ba = r_ba - u_beta * u_alpha / q
    p_aa = r_aa - u_alpha * u_alpha / q
    assert p_bb >= 0.0 and p_aa >= 0.0, "posterior variances cannot be negative"

    bars = state.bars + 1
    new_state = ChanPairsState(
        beta=beta,
        alpha=alpha,
        p_beta_beta=p_bb,
        p_beta_alpha=p_ba,
        p_alpha_alpha=p_aa,
        bars=bars,
    )
    return new_state, ChanPairsStep(
        beta=beta,
        alpha=alpha,
        innovation=innovation,
        s=s,
        z=innovation / s,
        ready=bars > 1,
        bars=bars,
    )


class ChanPairsFilter:
    """Stateful wrapper over :func:`step` — one filter per pair.

    Call :meth:`update` once per bar with the raw closes; nothing is stored but
    the recursion state, so a filter fed a prefix produces the same ``theta_t``
    as one fed the full series.
    """

    def __init__(self, config: ChanPairsConfig | None = None) -> None:
        self._config = config if config is not None else ChanPairsConfig()
        self._state = ChanPairsState()

    @property
    def config(self) -> ChanPairsConfig:
        return self._config

    @property
    def state(self) -> ChanPairsState:
        return self._state

    def update(self, x_price: float, y_price: float) -> ChanPairsStep:
        self._state, out = step(self._state, self._config, x_price, y_price)
        return out
