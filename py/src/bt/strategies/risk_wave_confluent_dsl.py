"""Regime-wave + adaptive-entropy-trend confluent rotation (daily, long-only).

Follow-up to the pure regime-gated book (risk_wave_rotation_dsl), which lost to
buy-and-hold: its weak spot was re-entering downtrends on a risk-on composite
cross (2022: three 0%-win resumptions). This version adds a *classical
trend-confirmation* stage from ``src.indicators.adaptive_entropy``: the book is
deployed only when BOTH agree --

  * the 5-leg composite wave (HYG/TLT, IEF/TLT, IWM/SPY, CPER/GLD, XLF/XLU —
    ``_risk_wave_common``) sits risk-on above ``z_enter`` (hysteresis to ``z_exit``), and
  * the SPY adaptive-entropy trend is bullish (``res.trend == 1``), i.e. close is
    above an entropy-adaptive band with structured (low-entropy) conviction.

and de-risked (flat) when EITHER flips bearish (regime < ``z_exit`` *or* SPY
entropy trend == -1). The entropy filter's job is to refuse re-entries into a
choppy/downtrend even when the slow credit/curve composite has ticked positive —
the exact false-risk-on resumption the pure gate caught on the chin.

Mechanics: regime recomputed each bar cursor-safely; the SPY entropy trend is
fed incrementally via ``OnlineAdaptiveEntropy`` held in ``ctx.shared`` (per
run/window, safe across split/sweep workers); equal-weight rotation across
``book_symbols`` reconciled every ``rebalance_days`` while risk-on. Fills at the
next bar's open. Benchmark = buy-and-hold SPY.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bt.strategies._risk_wave_common import closes_for_legs, composite_smooth
from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.types import StrategyParams
from src.indicators.adaptive_entropy import AdaptiveEntropyConfig, OnlineAdaptiveEntropy

STRATEGY_TYPE = "risk_wave_confluent_dsl"

_STATE_KEY = "risk_wave_confluent_state"


@dataclass(frozen=True)
class Params(StrategyParams):
    # --- regime wave thresholds (on the EWMA-smoothed composite z) ---
    # Keep z_enter near 0 (risk-neutral/on) and rely on the entropy trend for
    # timing, so the book is deployed during most of a healthy tape (higher
    # capital utilization) rather than whipsawed by a strict top-quartile band.
    z_enter: float = 0.0
    z_exit: float = -0.5
    # --- tradeable book / sizing (equity-scaled) ---
    book_symbols: tuple[str, ...] = ("SPY",)
    # Gross equity deployed when ON; split equally across warm members via
    # size_mode='equity' (each ctx.long sizes off live MTM equity, so the book
    # compounds PnL and stays near-full when it holds).
    target_allocation: float = 0.97
    rebalance_days: int = 21
    min_names: int = 3  # warm book members required to deploy
    min_history_bars: int = 130
    # Per-trade stop-loss for each book leg (fractional, e.g. 0.08 = 8% below
    # entry) — the per-position risk knob. 0.0 disables the stop (clean regime
    # overlay). Applied per ``ctx.long`` so each equal-weight lot keeps its own
    # downside protection rather than a single book-level ticket.
    stop_loss_pct: float = 0.0
    # --- adaptive-entropy trend confirmation ---
    trend_symbol: str = "SPY"
    entropy_lookback: int = 25


@dataclass
class _State:
    on: bool = False
    bars_since_sync: int = 0


def _warm_members(
    ctx: StrategyContext, book: tuple[str, ...], params: Params
) -> list[str]:
    """Book members past warmup (deployable today)."""
    warm: list[str] = []
    for sym in book:
        try:
            arr = ctx.ohlcv(sym).close.to_array()
        except Exception:
            continue
        if arr.size >= params.min_history_bars:
            warm.append(sym)
    return warm


def _flatten(ctx: StrategyContext, names: list[str], reason: str) -> None:
    """Close every open lot among ``names``."""
    for sym in names:
        if ctx.quantity(sym) > 1e-6:
            ctx.close(sym, reason=reason)


def _deploy_equal_weight(
    ctx: StrategyContext, names: list[str], params: Params
) -> None:
    """Close stale lots and re-open equal-weight slices of deployed equity."""
    _flatten(ctx, names, "[rvwf] rebalance equal weight")
    eq = ctx.current_equity()
    n = len(names)
    if n == 0 or eq * params.target_allocation <= 1:
        return
    per_eq = params.target_allocation / n
    for sym in sorted(names):
        try:
            px = ctx.price(sym)
        except Exception:
            continue
        if px > 0:
            ctx.long(
                sym,
                size=per_eq,
                sl=params.stop_loss_pct,
                reason=f"[rvwf] long {sym} (equal weight)",
                size_mode="equity",
            )


@strategy(bars="1d", stateful=True)
def on_candle(ctx: StrategyContext):
    # Warmup bars fill strategy state only — no fill can happen there, so the
    # DSL raises on any entry emitted during them. Accumulators above/below
    # still receive warmup bars because the engine feeds the store regardless.
    if ctx.phase == "warmup":
        return
    params: Params = ctx.params
    book = params.book_symbols
    if not book:
        return
    st: _State = ctx.shared.setdefault(_STATE_KEY, _State())
    es: OnlineAdaptiveEntropy = ctx.shared.setdefault(
        "rvwf_entropy",
        OnlineAdaptiveEntropy(AdaptiveEntropyConfig(lookback=params.entropy_lookback)),
    )

    # --- 1. regime wave ---
    closes = closes_for_legs(ctx)
    z = composite_smooth(closes)

    # --- 2. adaptive-entropy trend confirmation on ``trend_symbol`` ---
    trend = 0
    try:
        v = ctx.ohlcv(params.trend_symbol)
        if v.close.visible:
            c = float(v.close[-1])
            h = float(v.high[-1])
            lo = float(v.low[-1])
            res = es.observe(c, h, lo)
            trend = int(res.trend) if es.ready else 0
    except Exception:
        trend = 0

    # Deployability: enough warm book members.
    members = _warm_members(ctx, book, params)
    if len(members) < params.min_names:
        return

    if st.on:
        # De-risk when the composite falls OR SPY entropy turns bearish.
        if (z is not None and z < params.z_exit) or trend == -1:
            _flatten(ctx, members, "[rvwf] risk off (regime/trend)")
            st.on = False
            st.bars_since_sync = 0
            return
        # Periodic equal-weight reconciliation while risk-on.
        st.bars_since_sync += 1
        if st.bars_since_sync >= params.rebalance_days:
            _deploy_equal_weight(ctx, _warm_members(ctx, book, params), params)
            st.bars_since_sync = 0
        return

    # Flat: deploy only with composite risk-on AND bullish entropy trend.
    if z is not None and z > params.z_enter and trend == 1:
        _deploy_equal_weight(ctx, members, params)
        st.on = True
        st.bars_since_sync = 0
