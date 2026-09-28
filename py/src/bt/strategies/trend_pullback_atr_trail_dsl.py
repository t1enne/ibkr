"""Weekly-trend-confirmed mean reversion — progressive trail exit (DSL).

DSL-native port of ``trend_pullback_atr_trail`` following the declarative
:class:`StrategyContext` pattern (cf. ``ema_cross``). Same signal logic:

  * weekly trend filter (close > 50-week SMA),
  * daily overshoot entry (close <= MA20 - ATR_mult * ATR),
  * two-phase exit: wait for MA20 recovery, then a lowest-low trail stop.

Cross-candle state (cooldowns, in_trail, weekly-gate cache) lives in
``ctx.shared`` via ``@strategy(stateful=True)`` — the DSL replacement for the
``GLOBAL`` dict.

Sizing (matching the raw strategy): ``ctx.long`` sizes a fraction of *initial*
capital, but the raw strategy sizes ``qty = cash * risk_pct / (ATR * atr_mult)``
(current cash, ATR-risk scaled). To reproduce the exact same orders the port
back-solves the ``size`` argument from ``ctx.ta.atr`` + ``portfolio.cash`` so the
DSL emits identical share counts — the raw sizing formula is not directly
expressible as a bare ``ctx.long(size=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from src.bt.size.pure import risk_sized_qty
from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.series import SeriesView
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "trend_pullback_atr_trail_dsl"


@dataclass(frozen=True)
class Params(StrategyParams):
    atr_period: int = 14
    atr_mult: float = 1.5
    ma_period: int = 20
    weekly_ma_period: int = 50
    weekly_return_threshold: float = -99.0
    # ATR-risk sizing (matches the raw strategy exactly):
    #   qty = current_cash * risk_pct / (ATR * atr_mult)
    risk_pct: float = 0.01
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trail_lookback: int = 10
    warmup_bars: int = 60
    cooldown_bars: int = 3


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def _weekly_vals(closes: np.ndarray, window: int) -> list[float]:
    """Last ``window`` weekly buckets (5 trading days each), newest first.

    Matches ``src.bt.regime.gates._weekly_vals``.
    """
    arr = closes[-window * 5 :].reshape(window, 5)
    return [float(arr[-i - 1][-1]) for i in range(window)]


def _weekly_above_sma(ctx: StrategyContext, sym: str, params: Params) -> bool:
    """True if the symbol's weekly structure is bullish (close > weekly SMA)."""
    closes = ctx.ohlcv(sym).close.to_array()
    window = params.weekly_ma_period
    if len(closes) < window * 5 + 1:
        return False

    shared = ctx.shared
    key = f"{sym}|w{window}"
    week = (
        int(ctx.timestamp.isocalendar().year),
        int(ctx.timestamp.isocalendar().week),
    )
    cached = shared.get(key)
    if cached is not None and cached["week"] == week:
        weekly = list(cached["vals"])
        weekly[0] = float(closes[-1])
    else:
        weekly = _weekly_vals(closes, window)
        shared[key] = {"week": week, "vals": weekly}

    sma = float(sum(weekly[:window]) / window)
    return float(weekly[0]) > sma


def _is_oversold(ctx: StrategyContext, sym: str, params: Params) -> bool:
    o = ctx.ohlcv(sym)
    if len(o.close) < max(params.ma_period, params.atr_period) + 1:
        return False
    ma20 = _last(ctx.ta.sma(sym, params.ma_period))
    atr_val = _last(ctx.ta.atr(sym, params.atr_period))
    if np.isnan(ma20) or np.isnan(atr_val):
        return False
    return float(o.close[-1]) <= ma20 - params.atr_mult * atr_val


def _above_ma(ctx: StrategyContext, sym: str, params: Params) -> bool:
    o = ctx.ohlcv(sym)
    if len(o.close) < params.ma_period + 1:
        return False
    ma20 = _last(ctx.ta.sma(sym, params.ma_period))
    return not np.isnan(ma20) and float(o.close[-1]) > ma20


def _trail_broken(lows: SeriesView, lookback: int) -> bool:
    if len(lows) < lookback + 1:
        return False
    # NB mirrors raw ``lows.iloc[-lookback:-1].min()`` — the ``lookback-1``
    # bars *before* the current bar (index ``-1`` is excluded).
    prior = [float(lows[i]) for i in range(-lookback, -1)]
    return float(lows[-1]) < min(prior)


def _last(v: object) -> float:
    if isinstance(v, SeriesView):
        return float(v[-1])
    # Callers always pass a bare float (or float-valued SeriesView) here.
    return float(cast(float, v))


# ---------------------------------------------------------------------------
# DSL strategy
# ---------------------------------------------------------------------------


@strategy(bars="1d", stateful=True)
def on_candle(ctx: StrategyContext):
    # Warmup bars fill strategy state only — no fill can happen there, so the
    # DSL raises on any entry emitted during them. Accumulators above/below
    # still receive warmup bars because the engine feeds the store regardless.
    if ctx.phase == "warmup":
        return
    shared = ctx.shared
    shared.setdefault("cooldowns", {})
    shared.setdefault("in_trail", set())

    for sym in ctx.symbols:
        o = ctx.ohlcv(sym)

        pos_now = ctx.position(sym)
        if pos_now is not None:
            pid = pos_now.position_id

            # Phase 1: wait for MA20 recovery
            if pid not in shared["in_trail"]:
                if _above_ma(ctx, sym, ctx.params):
                    shared["in_trail"].add(pid)
                continue

            # Phase 2: trail mode — exit on trail-stop break
            if _trail_broken(o.low, ctx.params.trail_lookback):
                shared["in_trail"].discard(pid)
                shared["cooldowns"][sym] = ctx.params.cooldown_bars
                ctx.close(sym, reason="[trail] MA20 recovered -> trail broken")
            continue

        cd = shared["cooldowns"].get(sym, 0)
        if cd > 0:
            shared["cooldowns"][sym] = cd - 1
            continue

        if len(o.close) < ctx.params.warmup_bars:
            continue
        if not _weekly_above_sma(ctx, sym, ctx.params):
            continue
        if not _is_oversold(ctx, sym, ctx.params):
            continue

        atr_val = _last(ctx.ta.atr(sym, ctx.params.atr_period))
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        # ATR-risk sizing (matches the raw strategy exactly):
        #   qty = cash * risk_pct / (ATR * atr_mult)
        # Risk sizing is strategy-owned (risk_sized_qty); ctx.long computes
        # qty = size * initial_capital / price, so back-solve the size that
        # yields the raw's absolute share count.
        price = float(o.close[-1])
        cash = ctx.state.portfolio.cash
        qty = risk_sized_qty(
            equity=cash,
            price=price,
            stop_dist=atr_val * ctx.params.atr_mult,
            risk_pct=ctx.params.risk_pct,
        )
        size = qty * price / ctx.state.portfolio.initial_capital

        ctx.long(
            sym,
            size=size,
            sl=ctx.params.stop_loss,
            tp=ctx.params.take_profit,
            reason="[trail] 50wSMA up + ATR oversold",
        )
