"""Volume-Area breakout / breakdown — DSL strategy on the online Volume Profile.

Trades institutional order-flow structure: the rolling Value Area (POC + the
band of prices where most volume traded) is the "fair price" zone. A close
that breaks **above** the Value Area High with expanding volume and a
concurrent uptrend is order-flow exiting the fair-value range → leg long; the
mirror closes **below** the Value Area Low → leg short.

The profile is owned by an :class:`src.indicators.volume_profile.strategy.
OnlineVP` per symbol in ``ctx.shared``, fed once per candle (the cursor-safe
``state.candles`` read in :meth:`OnlineVP.observe`). Reconstructed fresh per
run/window so split/sweep/optimize windows are isolated.

Signal pipeline (per bar):
  1. Refresh each symbol's VP snapshot exactly once (store in shared so entry
     and exit read the same profile for the same bar).
  2. Breakout requires a real push, not a wick: close near the bar's extreme
     plus ``vol_expand_mult``-x average-volume confirmation, plus a trend
     concurrency filter (price on the correct side of its own rolling mean).
  3. Exits: revert back into the value area (tag stop) or an ATR trailing/
     fixed stop — both manual, since engine SL/TP fields are informational.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

from src.bt.state.types import Position
from src.indicators.volume_profile.strategy import OnlineVP
from src.bt.size.pure import risk_sized_qty
from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.series import SeriesView
from src.bt.strategies.types import StrategyParams
from src.bt.strategies.utils import gross_exposure, gross_gate

STRATEGY_TYPE = "vp_breakout_dsl"

# Keys for per-run state in ctx.shared (minted fresh per run/window).
_VP_KEY = "vp_profiles"  # dict[sym -> OnlineVP]
_SNAP_KEY = "vp_snaps"  # dict[sym -> VolumeProfileSnapshot] (once/bar)
_COOLDOWN_KEY = "cooldowns"

SizingMode = Literal["risk", "alloc"]
Direction = Literal["long", "short", "both"]


@dataclass(frozen=True)
class Params(StrategyParams):
    # -- direction gating --
    # Which side of the VP breakouts the strategy may trade. "both" keeps the
    # original long+short behaviour; "long"/"short" restrict entries so the
    # same code path serves the long leg, the bear (short) leg, and the
    # combined book — the plan's A/B design (no fork).
    direction: Direction = "both"
    # -- online volume profile --
    vp_window: int = 100
    num_bins: int = 50
    value_area_pct: float = 0.70
    vp_warmup: int = 60
    # -- signal confirmation --
    vol_period: int = 20
    vol_expand_mult: float = 1.3  # breakout volume >= this x avg
    wick_check: float = 0.5  # close must be in this top/bottom fraction of the bar
    trend_lookback: int = 50  # concurrency: price vs its own rolling mean
    # -- regime gate (Kaufman efficiency) --
    regime_lookback: int = 20
    regime_er: float = 0.0  # min |signed ER| to allow entry; 0.0 disables the gate
    # -- risk / sizing --
    sizing_mode: SizingMode = "risk"
    symbol_alloc: float = 0.20
    atr_period: int = 14
    atr_mult: float = 2.0  # stop = atr_mult * ATR against entry
    risk_pct: float = 0.005
    max_gross_exposure: float = (
        1.0  # aggregate |notional| cap, fraction of live equity (1.0 = off)
    )
    # -- misc --
    cooldown_bars: int = 5


# ---------------------------------------------------------------------------
# helpers (pure)
# ---------------------------------------------------------------------------


def _bar_position_close(close: float, high: float, low: float) -> float:
    """Where the close sits within the bar's range, 0 (low) .. 1 (high)."""
    rng = high - low
    if rng <= 1e-12:
        return 1.0
    return (close - low) / rng


def _avg_volume(ctx: StrategyContext, sym: str, params: Params) -> float:
    """Mean volume over the prior ``vol_period`` bars (excl. the current one)."""
    vol = ctx.ohlcv(sym).volume.to_array()
    n = len(vol)
    if n < params.vol_period + 1:
        return float("nan")
    return float(np.mean(vol[-params.vol_period : -1]))


def _trend_up(ctx: StrategyContext, sym: str, params: Params) -> bool:
    close = ctx.ohlcv(sym).close.to_array()
    n = len(close)
    if n < params.trend_lookback + 1:
        return False
    ma = float(np.mean(close[-params.trend_lookback :]))
    return float(close[-1]) > ma


def _efficiency_ratio(close: SeriesView, lookback: int) -> float:
    """Signed Kaufman efficiency ratio over the trailing ``lookback+1`` closes.

    ``sign(net_change) * (|net_change| / sum_of_abs_step_changes)`` in [-1, 1];
    positive = upward trend efficiency, negative = downward, ~0 = chop. NaN
    when insufficient data.
    """
    n = len(close)
    if n < lookback + 2:
        return float("nan")
    # Negative-index reads count back from the cursor tail, so this stays
    # correct under split/sweep truncation (positive indices would clamp to the
    # current bar and collapse the window to a single value).
    closes = [float(close[-1 - i]) for i in range(lookback + 1)]
    net = closes[0] - closes[-1]
    path = sum(abs(closes[i] - closes[i + 1]) for i in range(lookback))
    if path <= 1e-12:
        return 0.0
    return net / path


def _regime_ok(ctx: StrategyContext, sym: str, is_long: bool, params: Params) -> bool:
    """Direction-matched efficiency gate; disabled when ``regime_er <= 0``."""
    if params.regime_er <= 0.0:
        return True
    er = _efficiency_ratio(ctx.ohlcv(sym).close, params.regime_lookback)
    if math.isnan(er):
        return False
    if is_long:
        return er >= params.regime_er
    return er <= -params.regime_er


def _refresh_vp(ctx: StrategyContext, params: Params) -> None:
    """Feed each symbol's online VP once per candle; store the snapshots."""
    profs = ctx.shared.setdefault(_VP_KEY, {})
    snaps: dict[str, object] = {}
    for sym in ctx.symbols:
        prof = profs.get(sym)
        if not isinstance(prof, OnlineVP):
            prof = OnlineVP(
                num_bins=params.num_bins,
                value_area_pct=params.value_area_pct,
                window=params.vp_window,
                warmup_bars=params.vp_warmup,
            )
            profs[sym] = prof
        snaps[sym] = prof.observe(ctx.state, sym, ctx.interval)
    ctx.shared[_SNAP_KEY] = snaps


def _snap(ctx: StrategyContext, sym: str) -> object:
    return ctx.shared.get(_SNAP_KEY, {}).get(sym)


def _gross_exposure(
    base_equity: float,
    positions: Mapping[str, tuple[Position, ...]],
) -> float:
    """Back-compat shim -> :func:`src.bt.strategies.utils.gross_exposure`."""
    return gross_exposure(base_equity, positions)


def _size(ctx: StrategyContext, params: Params, price: float, atr_val: float) -> float:
    """0..1 capital fraction to deploy per ``sizing_mode``."""
    if params.sizing_mode == "alloc":
        target = params.symbol_alloc * ctx.state.portfolio.initial_capital
        cash = ctx.state.portfolio.cash
        notional = min(target, max(cash, 0.0))
        if notional <= 0 or price <= 0:
            return 0.0
        return min(max(notional / ctx.state.portfolio.initial_capital, 0.0), 1.0)
    if np.isnan(atr_val) or atr_val <= 0:
        return 0.0
    cash = ctx.state.portfolio.cash
    if cash <= 0 or price <= 0:
        return 0.0
    qty = risk_sized_qty(
        equity=cash,
        price=price,
        stop_dist=params.atr_mult * atr_val,
        risk_pct=params.risk_pct,
    )
    if qty <= 0:
        return 0.0
    size = qty * price / ctx.state.portfolio.initial_capital
    return min(max(size, 0.0), 1.0)


# ---------------------------------------------------------------------------
# strategy
# ---------------------------------------------------------------------------


@strategy(bars="1d", stateful=True)
def on_candle(ctx: StrategyContext):
    params: Params = ctx.params
    shared = ctx.shared
    shared.setdefault(_COOLDOWN_KEY, {})

    _refresh_vp(ctx, params)

    # ---- Exits first -------------------------------------------------------
    for sym in ctx.symbols:
        pos = ctx.position(sym)
        if pos is None:
            continue
        snap = _snap(ctx, sym)
        if snap is None or not getattr(snap, "ready", False):
            continue
        close = ctx.ohlcv(sym).close.last()
        vah = getattr(snap, "vah", None)
        val = getattr(snap, "val", None)
        is_long = pos.qty >= 0
        outcome = _exit(ctx, sym, is_long, close, vah, val, params)
        if outcome:
            reason, guard = outcome
            if guard is None:
                ctx.close(sym, reason=reason)
            else:
                # Model the intra-bar stop touch honestly: the engine fills a
                # guarded close at the adverse worse-of (stop, next open) rather
                # than delaying to next open and letting a gap blur the risk cap.
                ctx.close(sym, reason=reason, guard_price=guard)
            shared[_COOLDOWN_KEY][sym] = params.cooldown_bars

    # ---- Entries -----------------------------------------------------------
    for sym in ctx.symbols:
        cd = shared[_COOLDOWN_KEY].get(sym, 0)
        if cd > 0:
            shared[_COOLDOWN_KEY][sym] = cd - 1
            continue
        if ctx.position(sym) is not None:
            continue
        _enter(ctx, sym, params)


def _exit(
    ctx: StrategyContext,
    sym: str,
    is_long: bool,
    close: float,
    vah: object,
    val: object,
    params: Params,
) -> tuple[str, float | None] | None:
    """Revert into the value area (tag stop) or an ATR stop.

    Returns ``(reason, stop_guard)``; ``stop_guard`` is only set for the
    ATR-stop exit (the level the engine must fill at or worse — see
    ``execute_signal``). Direction is derived from the position inside the
    DSL's ``close``.
    """
    # Manual ATR stop (engine SL fields are informational).
    pos = ctx.position(sym)
    if pos is None:
        return None
    atr_val = ctx.ta.atr(sym, params.atr_period).last()
    stop_dist = (
        params.atr_mult * atr_val
        if atr_val > 0 and not np.isnan(atr_val)
        else float("inf")
    )
    o = ctx.ohlcv(sym)
    if is_long:
        if float(o.low[-1]) <= pos.entry_price - stop_dist:
            return ("[stop] ATR stop hit", pos.entry_price - stop_dist)
        if isinstance(vah, float) and close < vah:
            return ("[tag] long reverted into value area", None)
    else:
        if float(o.high[-1]) >= pos.entry_price + stop_dist:
            return ("[stop] ATR stop hit", pos.entry_price + stop_dist)
        if isinstance(val, float) and close > val:
            return ("[tag] short reverted into value area", None)
    return None


def _enter(ctx: StrategyContext, sym: str, params: Params) -> None:
    snap = _snap(ctx, sym)
    if snap is None or not getattr(snap, "ready", False):
        return
    o = ctx.ohlcv(sym)
    n = len(o.close)
    if n < max(params.vp_warmup, params.trend_lookback + 1, params.vol_period + 1):
        return

    close = o.close.last()
    high = float(o.high[-1])
    low = float(o.low[-1])
    price = close
    vah = getattr(snap, "vah", None)
    val = getattr(snap, "val", None)
    if not isinstance(vah, float) or not isinstance(val, float):
        return

    atr_val = ctx.ta.atr(sym, params.atr_period).last()
    if price <= 0 or np.isnan(atr_val) or atr_val <= 0:
        return

    avg_vol = _avg_volume(ctx, sym, params)
    if np.isnan(avg_vol) or avg_vol <= 0:
        return
    vol_expanding = float(o.volume[-1]) >= params.vol_expand_mult * avg_vol
    strength = _bar_position_close(close, high, low)

    long_breakout = (
        close > vah  # close (not a wick) breaks the value area top
        and strength >= params.wick_check  # held near the highs
        and vol_expanding  # order-flow confirmation
        and _trend_up(ctx, sym, params)  # concurrency: not against the trend
    )
    short_breakdown = (
        close < val
        and strength <= 1.0 - params.wick_check
        and vol_expanding
        and not _trend_up(ctx, sym, params)
    )
    long_breakout = long_breakout and _regime_ok(ctx, sym, True, params)
    short_breakdown = short_breakdown and _regime_ok(ctx, sym, False, params)
    # Direction gating ("short" suppresses long entries, "long" suppresses
    # short entries, "both" lets both through).
    long_breakout = long_breakout and params.direction in ("long", "both")
    short_breakdown = short_breakdown and params.direction in ("short", "both")
    if not long_breakout and not short_breakdown:
        return

    size = _size(ctx, params, price, atr_val)
    if size <= 0:
        return

    # Aggregate gross-exposure cap (Priority 4, "portfolio construction"). The
    # per-position size only risks ``risk_pct`` of *current cash* per name, so
    # without this the book has no portfolio-level notional ceiling when several
    # names are up concurrently. The gate compares gross + candidate against a
    # live-equity-relative cap; ``_size`` returns the candidate as a fraction of
    # *initial capital*, so rescale it onto the equity base first.
    equity = ctx.current_equity()
    init = ctx.state.portfolio.initial_capital
    candidate = size * init / equity if equity > 0 else 0.0
    if gross_gate(
        equity, ctx.state.portfolio.positions, params.max_gross_exposure, candidate
    ):
        return

    if long_breakout:
        ctx.long(
            sym,
            size=size,
            reason=f"[breakout] long close {close:.2f} > VAH {vah:.2f} on volume",
        )
    else:
        ctx.short(
            sym,
            size=size,
            reason=f"[breakdown] short close {close:.2f} < VAL {val:.2f} on volume",
        )
