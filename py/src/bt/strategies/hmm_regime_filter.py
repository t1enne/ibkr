"""HMM regime filter — the QuantStart reference rule set, long/flat, daily.

Reference: "Market Regime Detection using Hidden Markov Models in QSTrader"
(QuantStart, 2017). Two pieces: an SMA 10/30 trend base, and a 2-state Gaussian
HMM on the symbol's own return features whose state gates **entries only**.

Rules, exactly as the reference states them
-------------------------------------------
1. **Entry** — flat, the frozen model's regime label is the lowest-variance
   state, and ``SMA(10) > SMA(30)``. No crossover event, no confidence
   threshold, no hysteresis: the reference votes on the state argmax alone.
2. **Exit** — ``SMA(10) < SMA(30)`` while long. **Never** gated by the regime,
   so a position opened in a benign regime can always be closed in a bad one.
3. **No forced exit** when the regime flips (the article leaves that variant as
   an exercise), so the book can sit through a flip.
4. **Full capital, long/flat, one symbol.** No shorts, no leverage.

``use_regime=False`` is the engine control the stage-1 review could only script:
it drops rule 1's regime clause and leaves rules 2-4 intact, so the book is the
plain SMA 10/30 long/flat trend follower and the HMM is never constructed.

The label: one ``OnlineRegime`` per run in ``ctx.shared`` (never module state),
fed exactly once per bar — warmup bars included, since those are the training
window. The model is fitted **once** and frozen, so the trading window is
out-of-sample, which is the reference's only strong property.

Deliberate deviations from the reference
----------------------------------------
* **State meaning is derived, not eyeballed.** The article hard-codes "state 0
  = desirable" after seeing which component 2008 fell in. Here components are
  canonicalised by ``rank_states_by_vol`` — the entry state is the one with the
  lowest fitted return variance — so a refit cannot swap the gate onto the
  wrong component.
* **Features are the repo's three-column model** (log return, annualised vol,
  annualised momentum) with ``diag`` covariance; the model has no
  returns-only/``full`` mode.
* Training window is 5 years of daily bars, not 1993-2004: no local SPY daily
  history exists before 2004.

Invariants: every read is cursor-truncated (bar t uses closes <= t); an
unfitted model reports -1 and admits **nothing**, so the book stays flat rather
than guessing; ``window_size`` counts DAILY bars and must not exceed the
config's warmup bar count, or the single fit never lands and the run is flat.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bt.strategies.dsl import StrategyContext, strategy
from src.bt.strategies.types import StrategyParams
from src.indicators.hmm.strategy import OnlineRegime, OnlineRegimeResult

STRATEGY_TYPE = "hmm_regime_filter"

#: The only interval this module is correct on: the features annualise with
#: ``bars_per_year=252``.
_DAILY = "1d"
#: Regime that admits entries. ``rank_states_by_vol`` orders components by fitted
#: return variance, so 0 is always the lowest-variance component — the derived
#: equivalent of the article's hard-coded "state 0".
_ENTRY_REGIME = 0
_SMA_FAST = 10
_SMA_SLOW = 30
_BARS_PER_YEAR = 252
_N_REGIMES = 2
_VOL_WINDOW = 20
_MOMENTUM_WINDOW = 10
#: ``ctx.shared`` key holding the per-run ``OnlineRegime``.
_REGIME = "hmm_regime"


@dataclass(frozen=True)
class Params(StrategyParams):
    """Two model knobs plus the neutral control switch. None is swept."""

    #: Fraction of initial capital per lot. The article is fully invested; 0.99
    #: leaves room for spread + slippage + commission so a max-size open is not
    #: refused by the cash check.
    size: float = 0.99
    #: Daily bars the single fit trains on. The reference trained on ~12 years;
    #: local SPY daily history starts 2004, so this is 5 years less the margin
    #: a 5y warmup leaves before ``trading_start``.
    window_size: int = 1200
    #: Consult the HMM label before an entry (rule 1). ``False`` is the plain
    #: SMA 10/30 long/flat book — a NEUTRAL control, not a variant: entries and
    #: exits are then the SMA legs alone and the model is never fed, fitted or
    #: read. ``True`` (the default) is the reference rule set unchanged.
    use_regime: bool = True


def _observe(ctx: StrategyContext, params: Params) -> OnlineRegimeResult:
    """Feed this bar's close to the run's online HMM and return its label.

    Called once per bar, warmup included — the model needs those bars to reach
    ``window_size`` before the trading window opens.
    """
    interval = ctx.candle.interval or _DAILY
    assert interval == _DAILY, f"hmm_regime_filter is daily-only, got {interval!r}"
    model: OnlineRegime = ctx.shared.setdefault(
        _REGIME,
        OnlineRegime(
            n_regimes=_N_REGIMES,
            window_size=params.window_size,
            vol_window=_VOL_WINDOW,
            momentum_window=_MOMENTUM_WINDOW,
            retrain_interval=10**9,  # irrelevant with fit_once, kept explicit
            random_state=42,
            bars_per_year=_BARS_PER_YEAR,
            fit_once=True,
        ),
    )
    return model.observe(ctx.state, ctx.candle.symbol, interval)


@strategy(stateful=True)
def on_candle(ctx: StrategyContext) -> None:
    params: Params = ctx.params
    assert 0.0 < params.size <= 1.0, "size is a capital fraction"
    symbol = ctx.candle.symbol

    # Feed BEFORE the warmup return: warmup bars are the fit's training window.
    # The control book never consults the model, so it is not even constructed.
    regime: OnlineRegimeResult | None = (
        _observe(ctx, params) if params.use_regime else None
    )
    if ctx.phase == "warmup":
        return

    fast = ctx.ta.sma(symbol, _SMA_FAST)
    slow = ctx.ta.sma(symbol, _SMA_SLOW)
    if not len(fast) or not len(slow):
        return
    # A NaN leg compares False both ways: no entry and no exit, which is the
    # flat/short-history case.
    trend_up = float(fast[-1]) > float(slow[-1])
    trend_down = float(fast[-1]) < float(slow[-1])

    regime_admits = not params.use_regime or (
        regime is not None and regime.value == _ENTRY_REGIME
    )
    if ctx.position(symbol) is None:
        if regime_admits and trend_up:
            # ``size_mode="equity"`` is a correctness requirement, not tuning:
            # capital mode sizes every entry off the INITIAL capital, so after a
            # loss the next max-size open exceeds available cash and the engine
            # rejects the fill. Equity mode is also the closer reading of "fully
            # invested" — 99% of what the book is worth at entry.
            ctx.long(
                symbol,
                size=params.size,
                size_mode="equity",
                reason="regime_trend_entry",
            )
        return

    # Exits are never gated by the regime (rule 2).
    if trend_down:
        ctx.close(symbol, reason="sma_exit")
