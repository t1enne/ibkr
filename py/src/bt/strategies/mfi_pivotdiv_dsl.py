"""Pivot-pair MFI divergence (long + short), geometry-gated.

Unlike ``mfi_divergence_dsl``, which compares the ``min`` of two fixed halves
of a window (no turning points, no structure -- it fires on monotone legs), this
strategy only trades a **divergence wave**:

    swing low L1 -> intervening swing high -> swing low L2

with ``L2.price < L1.price`` and ``L2.mfi > L1.mfi + gap`` (bullish), i.e. a
lower low in price that the oscillator refuses to confirm. Bearish is the
mirror on swing highs. All the geometry lives in
:mod:`src.bt.strategies.divergence_geometry` and is unit-tested there; this
module only wires it to candles, sizing, and exits.

Optional regime gates (all off by default, swept per config):
``need_trend`` (symbol vs its own MA with the trade) and ``bench_up`` (SPY vs
its MA). ``cooldown_bars`` suppresses the stale-signal re-entry loop.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from src.bt.strategies.divergence_geometry import (
    DivergenceParams,
    PivotScanner,
    evaluate_from_pivots,
)
from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.types import StrategyParams
from src.bt.strategies.utils import gross_gate

STRATEGY_TYPE = "mfi_pivotdiv_dsl"
_STATE_KEY = "mfi_pivotdiv_state"


@dataclass(frozen=True)
class _State:
    best_long: float | None = None  # running best (lowest) close while long
    best_short: float | None = None  # running best (highest) close while short
    was_in_position: bool = False  # previous bar held a position
    last_exit_bar: int = -1  # bar index of the last close (-1 = never)
    lows: PivotScanner | None = None  # cached swing-low scanner (price init)
    highs: PivotScanner | None = None  # cached swing-high scanner


@dataclass(frozen=True)
class Params(StrategyParams):
    # -- MFI / divergence geometry --
    mfi_period: int = 14
    pivot_bars: int = 3  # fractal half-width for swings
    lookback: int = 90  # bars scanned for pivots
    min_gap_bars: int = 5  # min separation between the two swings
    max_gap_bars: int = 0  # max separation (0 = lookback)
    max_age_bars: int = 0  # P2 recency (0 = max_gap)
    min_osc_gap: float = 5.0  # absolute MFI improvement required
    min_osc_gap_frac: float = 0.0  # MFI improvement as fraction of its range
    require_intervening: bool = True  # demand the opposite swing between P1/P2
    min_intervening_atr: float = 0.0  # required retrace depth of that swing
    min_price_ext: float = 0.0  # min penetration of P1 by P2 (fraction)
    max_price_ext: float = 0.0  # max penetration (0 = unbounded)
    osc_extreme_side: bool = False  # require P1 in the MFI tail
    osc_extreme_floor: float = 35.0  # bullish: P1 MFI <= this
    osc_extreme_ceil: float = 65.0  # bearish: P1 MFI >= this
    # -- side admission --
    allow_longs: bool = True
    allow_shorts: bool = True
    # -- regime gates --
    need_trend: bool = False  # symbol close on the trade's side of its MA
    trend_ma_period: int = 200
    bench_gate: bool = False  # SPY on the trade's side of its MA
    benchmark: str = "SPY"
    bench_ma_period: int = 200
    # -- risk / exits --
    atr_period: int = 14
    trail_atr_mult: float = 1.5  # chandelier trail off the running extreme
    entry_stop_atr: float = 3.0  # structural stop, ATRs beyond entry
    risk_pct: float = 0.05
    warmup_bars: int = 60
    cooldown_bars: int = 0  # flat bars after an exit before re-entry
    max_gross_exposure: float = 1.0  # aggregate |notional| cap vs equity (1 = off)


@strategy(bars="1d", stateful=True)
def on_candle(ctx: StrategyContext):
    p: Params = ctx.params
    for sym in ctx.symbols:
        _process_symbol(ctx, p, sym)


def _process_symbol(ctx: StrategyContext, p: Params, sym: str) -> None:
    holder: dict[str, _State] = ctx.shared.setdefault(_STATE_KEY, {})
    state = holder.get(sym)
    if state is None:
        state = _State()
        holder[sym] = state

    o = ctx.ohlcv(sym)
    if o is None or len(o.close) <= p.warmup_bars:
        return

    n = len(o.close)
    if n == 0:
        return
    close = float(o.close.to_array()[-1])

    # Pivot scanners are the expensive-once, cheap-per-bar cache. They must be
    # refreshed BEFORE ``put`` closes over ``state``, else every write would
    # rebuild them from scratch and undo the optimisation. Feeding them on
    # every bar (even while in a position) keeps them warm across an exit.
    mfi = ctx.ta.mfi(sym, period=p.mfi_period).to_array()
    if mfi.size != n:
        return
    prices = o.close.to_array()
    state = _scanners(state, p, prices, mfi)
    holder[sym] = state
    assert state.lows is not None and state.highs is not None
    lows, highs = state.lows, state.highs

    def put(**kw: object) -> None:
        holder[sym] = dataclasses.replace(state, **kw)

    atr = ctx.ta.atr(sym, period=p.atr_period).last()
    if not np.isfinite(atr):
        atr = 0.0

    # ---- manage an open position: chandelier ATR trail on the active side ---
    qty = ctx.quantity(sym)
    if qty != 0:
        put(was_in_position=True)
        if qty > 0:
            if _trail(ctx, p, state, put, sym, close, atr, side="long"):
                return
        else:
            if _trail(ctx, p, state, put, sym, close, atr, side="short"):
                return
        return

    if state.was_in_position:
        state = dataclasses.replace(state, was_in_position=False, last_exit_bar=n)
        holder[sym] = state

    if _cooling(n, state.last_exit_bar, p.cooldown_bars):
        return

    # ---- pivot geometry, evaluated once per bar ----------------------------
    geo = DivergenceParams(
        pivot_bars=p.pivot_bars,
        lookback=p.lookback,
        min_gap_bars=p.min_gap_bars,
        max_gap_bars=p.max_gap_bars,
        max_age_bars=p.max_age_bars,
        min_osc_gap=p.min_osc_gap,
        min_osc_gap_frac=p.min_osc_gap_frac,
        require_intervening=p.require_intervening,
        min_intervening_atr=p.min_intervening_atr,
        min_price_ext=p.min_price_ext,
        max_price_ext=p.max_price_ext,
        osc_extreme_side=p.osc_extreme_side,
        osc_extreme_floor=p.osc_extreme_floor,
        osc_extreme_ceil=p.osc_extreme_ceil,
    )

    if p.allow_longs and _side_ok(ctx, p, sym, close, side="long"):
        long_pivot = evaluate_from_pivots(
            lows.pivots, highs.pivots, prices, mfi, "long", geo, atr=atr
        )
        if long_pivot is not None:
            _enter(ctx, p, sym, close, atr, "long", long_pivot.price, long_pivot.osc)
            return

    if p.allow_shorts and _side_ok(ctx, p, sym, close, side="short"):
        short_pivot = evaluate_from_pivots(
            highs.pivots, lows.pivots, prices, mfi, "short", geo, atr=atr
        )
        if short_pivot is not None:
            _enter(ctx, p, sym, close, atr, "short", short_pivot.price, short_pivot.osc)


def _scanners(state: _State, p: Params, prices: np.ndarray, mfi: np.ndarray) -> _State:
    """Return ``state`` with its pivot scanners created and fed up to date.

    Scanners are rebuilt whenever the pivot width changes, so a sweep that
    varies ``pivot_bars`` cannot inherit stale pivots.
    """
    rebuild = (
        state.lows is None
        or state.highs is None
        or state.lows.pivot_bars != p.pivot_bars
    )
    if rebuild:
        state = dataclasses.replace(
            state,
            lows=PivotScanner(p.pivot_bars, high=False),
            highs=PivotScanner(p.pivot_bars, high=True),
        )
    assert state.lows is not None and state.highs is not None
    state.lows.extend(prices, mfi)
    state.highs.extend(prices, mfi)
    return state


def _side_ok(
    ctx: StrategyContext, p: Params, sym: str, close: float, *, side: str
) -> bool:
    """Regime admission for ``side``. A gate that cannot be evaluated denies."""
    if p.need_trend:
        ok = _ma_side(ctx, sym, p.trend_ma_period, want_above=side == "long")
        if ok is not True:
            return False
    if p.bench_gate:
        ok = _ma_side(ctx, p.benchmark, p.bench_ma_period, want_above=side == "long")
        if ok is not True:
            return False
    assert close > 0, "close must be positive before signal generation"
    return True


def _ma_side(
    ctx: StrategyContext, sym: str, period: int, *, want_above: bool
) -> bool | None:
    """True when ``sym``'s close sits on ``want_above``'s side of its SMA."""
    if period <= 0:
        return None
    o = ctx.ohlcv(sym)
    if o is None or len(o.close) < period:
        return None
    cur = float(o.close.to_array()[-1])
    ma = ctx.ta.sma(sym, period=period).last()
    if not (np.isfinite(cur) and np.isfinite(ma)) or ma <= 0:
        return None
    return (cur > ma) is want_above


def _trail(
    ctx: StrategyContext,
    p: Params,
    state: _State,
    put,
    sym: str,
    close: float,
    atr: float,
    *,
    side: str,
) -> bool:
    """Ratchet the chandelier trail; close and return True when it trips."""
    long = side == "long"
    best = state.best_long if long else state.best_short
    if best is None or not (best > 0):
        put(**({"best_long": close} if long else {"best_short": close}))
        return False
    improved = (close > best + 1e-9) if long else (close < best - 1e-9)
    if improved:
        put(**({"best_long": close} if long else {"best_short": close}))
        return False
    if atr <= 0:
        return False
    hit = (
        (close <= best - p.trail_atr_mult * atr)
        if long
        else (close >= best + p.trail_atr_mult * atr)
    )
    if not hit:
        return False
    put(**({"best_long": None} if long else {"best_short": None}))
    ctx.close(
        sym,
        reason=(
            f"[pivotdiv] trail {side} {sym}: close {close:.2f} "
            f"{p.trail_atr_mult:.1f}ATR off best {best:.2f}"
        ),
    )
    return True


def _cooling(now: int, last_exit_bar: int, cooldown_bars: int) -> bool:
    """True while re-entry on this symbol is blocked by the exit cooldown."""
    if cooldown_bars <= 0 or last_exit_bar < 0:
        return False
    return now - last_exit_bar < cooldown_bars


def _enter(
    ctx: StrategyContext,
    p: Params,
    sym: str,
    close: float,
    atr: float,
    side: str,
    pivot_price: float,
    pivot_osc: float,
) -> None:
    if not (atr > 0 and close > 0 and ctx.state.portfolio.cash > 0):
        return
    if gross_gate(
        ctx.current_equity(),
        ctx.state.portfolio.positions,
        p.max_gross_exposure,
        p.risk_pct,
    ):
        return
    stop = p.entry_stop_atr * atr / close
    reason = (
        f"[pivotdiv] {side} {sym}: MFI swing div at {pivot_price:.2f} "
        f"(mfi {pivot_osc:.1f}); enter {close:.2f}"
    )
    if side == "long":
        ctx.long(sym, size=p.risk_pct, size_mode="equity", sl=stop, reason=reason)
    else:
        ctx.short(sym, size=p.risk_pct, size_mode="equity", sl=stop, reason=reason)
