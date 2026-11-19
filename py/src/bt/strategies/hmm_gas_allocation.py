"""Paper-faithful HMM regime allocation -- Nystrup, Madsen & Lindström (2017).

Reference: "Dynamic portfolio optimization across hidden market regimes",
Quantitative Finance 18(1) 83-95. Scope here is the paper's **single-risky-asset
variant**: SPY vs 0% cash, daily bars. It reproduces the paper's METHOD, not its
1998-2015 printed result -- none of the paper's six indices, and no bond proxy
older than 2016-12, exist in this database.

The chain is the paper's (Sec. 3), not a state->asset lookup table:

    filtered state probabilities -> one-step forecast mean/variance
    -> constrained mean-variance weight -> fractional-holding trade next day

* The estimator is :class:`~src.indicators.hmm.score_driven.ScoreDrivenHMM` --
  2-state Gaussian HMM on daily log-returns, parameters re-estimated **every
  day** by the recursive score (GAS) update with ``N_eff = 260``, filtered
  ``alpha_t|t`` only (no smoother, no Viterbi, no frozen fit). It is constructed
  once per run in ``ctx.shared`` and fed exactly once per bar, warmup included,
  because the first 260 returns are the paper's ML/EM initialisation.
* The allocation is the paper's mean-variance step with its risk aversion
  ``gamma = 2``: ``max_{0<=w<=1} w*mu - gamma*w**2*variance``, a 1-D bounded
  scalar solve (``scipy.optimize.minimize_scalar``; no new dependency).
* Trades execute with one-day delay by construction: a decision taken on bar
  ``t``'s close is emitted as a next-open fill by the engine, which is the
  paper's "80% of the holding is sold on day t+1" convention. The emitted order
  is a fixed share delta computed at ``t`` (a fraction of the holding), so the
  long-only constraint survives the delay.

DEVIATION A2 (unavoidable): the paper charges ``kappa = 10 bp`` of traded
notional **inside the budget constraint** (``1'u + kappa'|u| <= 0``). This
engine has no budget/weight constraint at all -- cost is a fill-time price
haircut -- so the paper's cost-in-the-feasible-set cannot be expressed. Two
consequences, both stated in the report:

1. the no-trade band the paper's cost generates is LOST, so this rebuild trades
   small weight changes the paper would have left alone, and annual turnover is
   expected **above** the paper's 1.17 (its unpenalised, no-delay variants were
   4.16-4.61, so the honest comparison is against those);
2. the one fragment that IS expressible is the post-trade weight cap the cost
   implies: ``w <= 1/(1+kappa) = 0.999`` (the paper cannot hold more than the
   value left after paying the cost). That cap is applied, and the 10 bp is
   charged at fill time through ``spread_bps``.

Control arms (identical sample, paper Sec. 4.5):
* ``paper`` -- the adaptive mean-variance MPC; ``N_eff = 260``.
* ``rbaa``  -- rule-based RBAA: all-in stocks or all-in cash, switching only
  when the filtered probability that a regime change occurred exceeds 0.9998;
  its own estimator with ``N_eff = 2 years``. The one-bit mapping uses the
  low-variance state (stocks) vs the high-variance state (cash).
* ``static`` -- monthly rebalance to 69% SPY / 31% cash. The paper's SAA is
  69% stocks / 31% bonds; **no bond leg exists in this sample** (TLT starts
  2016-12) so the 31% is 0%-interest cash. That is a loud deviation: the static
  control is a different asset mix from the paper's, not a re-run of it.

Cash earns 0% by absence: the engine has no cash-rate field (same assumption as
the paper, but hard-coded, not a knob).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

from src.bt.strategies.dsl import StrategyContext, strategy
from src.bt.strategies.types import StrategyParams
from src.indicators.hmm.score_driven import (
    GasObservation,
    ScoreDrivenHMM,
    banded_weight,
    forecast_moments,
)

STRATEGY_TYPE = "hmm_gas_allocation"

_DAILY = "1d"
#: ctx.shared keys.
_MODEL = "gas_model"
_PREV_CLOSE = "prev_close"
_WEIGHT = "target_weight"
_MONTH = "rebalance_month"
#: Paper values: N_eff = 260 days for the MPC arm, = 2 years for RBAA.
_NEFF_MPC = 260.0
_NEFF_RBAA = 504.0
#: Fraction of the paper's SAA: 69% stocks, and 31% of *cash* (no bond leg).
_STATIC_WEIGHT = 0.69

Control = Literal["paper", "rbaa", "static"]


@dataclass(frozen=True)
class Params(StrategyParams):
    """Six knobs, none swept: the paper's values are the defaults."""

    #: Paper Sec. 3.1 stage cost weights variance by this constant (``gamma=2``
    #: is the reported mean-variance arm).
    risk_aversion: float = 2.0
    #: Proportional cost per unit of traded notional. Charged at fill time
    #: (``spread_bps = 2 * kappa * 1e4`` = 20 -> 10 bp per side); see DEVIATION
    #: A2 in the module docstring.
    kappa: float = 0.001
    #: Paper (23) trading aversion ``rho`` -- the no-trade band's width. 0.02 is
    #: the value of the paper's own ``MPC(gamma=0, rho=0.02)`` arm (Table 1).
    trading_aversion: float = 0.02
    #: Paper (25) look-ahead ``K``. 100 is the paper's own choice (Sec. 4.2): a
    #: 0.99 self-transition implies a 100-day expected sojourn, and a longer
    #: horizon was found not to change the results.
    horizon: int = 100
    #: Which of the three arms to run: ``paper`` | ``rbaa`` | ``static``.
    control: Control = "paper"
    #: RBAA only: switch when the filtered probability of a regime change on
    #: this bar exceeds this threshold (paper claims the switch rule only).
    rbaa_threshold: float = 0.9998


def _assert_single_symbol(ctx: StrategyContext) -> str:
    """The allocator is a one-risky-asset programme: SPY (or one symbol) only."""
    symbols = ctx.symbols
    assert len(symbols) == 1, f"{STRATEGY_TYPE} is a one-asset programme, got {symbols}"
    symbol = symbols[0]
    assert ctx.candle.symbol == symbol, "the evaluation clock must be the one symbol"
    assert (ctx.candle.interval or _DAILY) == _DAILY, "daily bars only"
    return symbol


def _max_weight(kappa: float) -> float:
    """Post-trade cap the paper's cost-in-budget implies: ``1 / (1 + kappa)``."""
    assert 0.0 <= kappa < 1.0
    return 1.0 / (1.0 + kappa)


def _held_weight(ctx: StrategyContext, symbol: str) -> float:
    """The book's CURRENT weight of ``symbol`` -- the paper's ``h_{t-1}/V``.

    Read off the live mark, not the last target: the band is measured from what
    is actually held (the paper's ``u_t`` is a trade out of the actual holding),
    so a drift beyond the band re-opens the decision on its own.
    """
    equity = ctx.current_equity()
    assert equity > 0.0, "a weight needs a positive equity base"
    weight = float(ctx.quantity(symbol) * ctx.price(symbol) / equity)
    return min(1.0, max(0.0, weight))


def _rebalance(ctx: StrategyContext, symbol: str, target_weight: float) -> None:
    """Move the book to ``target_weight`` of live equity (long-only, w in 0..1).

    One ``ctx.resize`` call: a net-delta ``rebalance`` fill against the existing
    lots (the engine's TWAC resize path), never a fresh lot per tick. Friction is
    therefore paid on the traded delta only, and the book holds ONE lot.
    """
    assert 0.0 <= target_weight <= 1.0
    ctx.resize(symbol, target_weight, reason="gas_target", tag="gas")


def _observe(ctx: StrategyContext, params: Params) -> GasObservation | None:
    """Feed this bar's log-return to the run's online GAS HMM.

    Called once per bar, warmup included: the first 260 returns are the ML/EM
    initialisation. Returns ``None`` until the model is ready (and on the first
    bar, which has no previous close).
    """
    symbol = ctx.symbols[0]
    close = ctx.price(symbol)
    previous = ctx.shared.get(_PREV_CLOSE)
    ctx.shared[_PREV_CLOSE] = close
    if previous is None:
        return None
    model: ScoreDrivenHMM = ctx.shared.setdefault(
        _MODEL,
        ScoreDrivenHMM(
            neff=_NEFF_RBAA if params.control == "rbaa" else _NEFF_MPC,
            em_obs=260,
        ),
    )
    return model.update(math.log(close / float(previous)))


def _paper_weight(obs: GasObservation, params: Params, held: float) -> float:
    """Paper arm: the K-step (25) programme with the (22) risk and (23) trading
    penalties -- i.e. the mean-variance optimum inside a NO-TRADE BAND.

    ``held`` is the book's current weight, so the band is the paper's ``|u_t|``
    penalty measured against the actual holding: a move worth less than
    ``rho / (2 gamma sum_k sigma_k^2)`` is not taken. The post-trade weight is
    capped at the cost-in-budget weight ``1/(1+kappa)`` (see DEVIATION A2).
    """
    means, variances = forecast_moments(obs, params.horizon)
    weight = banded_weight(
        means, variances, params.risk_aversion, params.trading_aversion, held
    )
    return min(weight, _max_weight(params.kappa))


def _rbaa_weight(obs: GasObservation, ctx: StrategyContext, params: Params) -> float:
    """RBAA arm: all-in stocks (low-vol state) or all-in cash, switch on a
    high-confidence regime-change probability only."""
    current = cast(float | None, ctx.shared.get(_WEIGHT))
    if current is None:
        # First decision: take the filtered argmax position.
        return 1.0 if obs.filtered[0] >= obs.filtered[1] else 0.0
    if obs.change_prob > params.rbaa_threshold:
        return 1.0 if obs.filtered[0] >= obs.filtered[1] else 0.0
    return current


def _static_weight(ctx: StrategyContext) -> float | None:
    """SAA arm: rebalance to 69/31 once per calendar month (else no trade)."""
    month = (ctx.timestamp.year, ctx.timestamp.month)
    if ctx.shared.get(_MONTH) == month:
        return None
    ctx.shared[_MONTH] = month
    return _STATIC_WEIGHT


@strategy(stateful=True)
def on_candle(ctx: StrategyContext) -> None:
    params: Params = ctx.params
    assert params.control in ("paper", "rbaa", "static"), params.control
    assert params.risk_aversion > 0.0
    assert 0.0 <= params.kappa < 1.0
    assert 0.0 < params.rbaa_threshold <= 1.0
    assert params.trading_aversion >= 0.0
    assert params.horizon >= 1
    symbol = _assert_single_symbol(ctx)

    # The model is fed on every bar, warmup included: those bars are the
    # paper's 260-observation initialisation. The static control never reads it.
    obs = None if params.control == "static" else _observe(ctx, params)
    if ctx.phase == "warmup":
        return

    if params.control == "static":
        target = _static_weight(ctx)
        if target is None:
            return
    else:
        if obs is None or not obs.ready:
            return
        target = (
            _paper_weight(obs, params, _held_weight(ctx, symbol))
            if params.control == "paper"
            else _rbaa_weight(obs, ctx, params)
        )

    ctx.shared[_WEIGHT] = target
    _rebalance(ctx, symbol, target)
