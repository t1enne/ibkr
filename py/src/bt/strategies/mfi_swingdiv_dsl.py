"""Bearish MFI-divergence short on lag-confirmed fractal swing highs.

Shorts the newer of the two most recent higher-high pivots when MFI undercuts
it (divergence); a cold-fade momentum gate parks entries during hot
up-thrusts, and an entry trend gate requires price to have pulled back under
a recent high. Chandelier ATR trail exit, wide disaster SL. Per-symbol state,
no lookahead.
"""

from __future__ import annotations
import dataclasses
from dataclasses import dataclass
import numpy as np
from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "mfi_swingdiv_dsl"
_STATE_KEY = "mfi_swing_state"


@dataclass(frozen=True)
class _State:
    best: float | None = (
        None  # running best (lowest) close since entry (None while flat)
    )
    last_pivot_hi: float = 0.0  # most recent confirmed pivot close (cached)
    last_pivot_mfi: float = 0.0  # mfi at that pivot close
    last_pivot_i: int = -1  # its bar index (age guard)


@dataclass(frozen=True)
class Params(StrategyParams):
    # -- swing pivot detection --
    pivot_bars: int = 3  # lower closes needed on each side of a pivot (fractal range)
    mfi_period: int = 14
    ob_floor: float = 60.0  # prior pivot must be this MFI overbought
    mfi_gap: float = 3.0  # newest pivot must undercut prior MFI by this
    pivot_lookback: int = 90  # bars of history scanned for the newest pivot
    min_swing: float = 0.0  # swing high must lead the last local low by this (0 off)
    # -- cold-fade momentum gate (0 = off) --
    mom_lookback: int = 10  # closes back for the up-thrust ROC
    mom_gate_atr: float = 2.5  # skip short when ROC over mom_lookback >= this * ATR
    # -- entry trend gate (0 = off) --
    high_lookback: int = 50  # prior closes scanned for the recent high reference
    high_atr: float = 0.5  # close must be >= this many ATRs under the recent high
    # -- chandelier trailing management --
    atr_period: int = 14
    trail_atr_mult: float = 2.0  # ATRs off the running best close that bank the trade
    disaster_atr_mult: float = 4.0  # disaster SL in ATR above entry (whole bracket)
    # -- risk sizing --
    risk_pct: float = 0.05
    warmup_bars: int = 60


@strategy(bars="1d", stateful=True)
def on_candle(ctx: StrategyContext):
    p: Params = ctx.params
    # Fire on the last config symbol; loop all names so every ticker is traded.
    for sym in ctx.symbols:
        _process_symbol(ctx, p, sym)


def _process_symbol(ctx: StrategyContext, p: Params, sym: str) -> None:
    holder: dict[str, _State] = ctx.shared.setdefault(_STATE_KEY, {})
    state = holder.get(sym)
    if state is None:
        state = _State()
        holder[sym] = state

    def put(**kw: object) -> None:
        holder[sym] = dataclasses.replace(state, **kw)

    o = ctx.ohlcv(sym)
    if o is None or len(o.close) <= p.warmup_bars:
        return

    closes = o.close.to_array()
    n = len(closes)
    if n == 0:
        return
    close = float(closes[-1])  # this symbol's close for the current bar

    mfi_vals = ctx.ta.mfi(sym, period=p.mfi_period).to_array()
    if n != len(mfi_vals) or n <= 0:
        return

    atr = ctx.ta.atr(sym, period=p.atr_period).last()
    if not np.isfinite(atr):
        atr = 0.0

    # ---- while short: chandelier ATR trailing exit ----
    if ctx.quantity(sym) != 0:
        best = state.best
        if best is None or not (best > 0):
            put(best=close)
            return
        if close < best - 1e-9:
            put(best=close)  # ratchet the trail behind a fresh favorable close
            return
        if atr > 0 and close >= best + p.trail_atr_mult * atr:
            put(best=None)
            reason = (
                f"[swingdiv] trail tp {sym}: close {close:.2f} "
                f"{p.trail_atr_mult:.1f}ATR off best {best:.2f}"
            )
            ctx.close(sym, reason=reason)
        return

    n = len(closes)
    if n < p.warmup_bars + 2 * p.pivot_bars + 2:
        return
    win_start = max(0, n - p.pivot_lookback)
    window = closes[win_start:]
    wbase = win_start
    pivot_idx: list[
        tuple[int, float]
    ] = []  # absolute idx, close  of right-confimed swing high
    ws = len(window)
    D = p.pivot_bars
    for i in range(ws):
        hi_i = window[i]
        if not np.isfinite(hi_i):
            continue
        # D lower closes each side, right side fully behind the cursor.
        if i < D or not all(window[j] < hi_i for j in range(i - D, i)):
            continue
        if (i + D) >= (ws - 1) or not all(
            window[i + 1 + r] < hi_i for r in range(0, D)
        ):
            continue
        pivot_idx.append((wbase + i, float(hi_i)))
    if not pivot_idx:
        return

    if len(pivot_idx) < 2:
        return  # need at least two resolved swing highs for a divergence

    # use the two most recent pivots (should be newest = highest for a top)
    piv2 = pivot_idx[-2:]
    (i0, hi0), (i1, hi1) = piv2
    if not (i1 > i0 and hi1 > hi0):
        return  # not a higher-high sequence: either equal/newer-lower leg
    # pull MFI at each pivot close
    if not (np.isfinite(closes[i0]) and np.isfinite(closes[i1])):
        return
    m0 = float(mfi_vals[i0]) if np.isfinite(mfi_vals[i0]) else float("nan")
    m1 = float(mfi_vals[i1]) if np.isfinite(mfi_vals[i1]) else float("nan")
    if not (np.isfinite(m0) and np.isfinite(m1)):
        return
    divergent = m0 >= p.ob_floor and m1 <= m0 - p.mfi_gap
    if not divergent:
        put(last_pivot_hi=hi1, last_pivot_mfi=m1, last_pivot_i=i1)
        return

    # ---- cold-fade momentum gate: assess whether the up-thrust is still hot ----
    mom_hot = False
    if p.mom_lookback > 0 and p.mom_gate_atr > 0:
        lb = p.mom_lookback
        if n > lb and atr > 0 and np.isfinite(closes[n - 1 - lb]):
            roc = close - float(closes[n - 1 - lb])
            if roc >= p.mom_gate_atr * atr:
                mom_hot = True

    if mom_hot:
        # Park while thrust runs; re-fire once it cools.
        put(last_pivot_hi=hi1, last_pivot_mfi=m1, last_pivot_i=i1)
        return

    # Entry trend gate: no short while price presses a recent high (intact
    # uptrend). Require it pulled back, else park and re-check next bar.
    if p.high_lookback > 0 and n > p.high_lookback:
        recent_high = float(np.max(closes[n - 1 - p.high_lookback : n - 1]))
        if close >= recent_high - p.high_atr * atr:
            put(last_pivot_hi=hi1, last_pivot_mfi=m1, last_pivot_i=i1)
            return

    _enter_short(
        ctx,
        p,
        sym,
        close,
        atr,
        reason=(
            f"[swingdiv] short {sym}: 2nd-div top {hi1:.2f}@{m1:.1f} vs prior "
            f"{hi0:.2f}@{m0:.1f}; divergence, gap={p.mfi_gap:.1f}"
        ),
    )


def _enter_short(
    ctx: StrategyContext,
    p: Params,
    sym: str,
    close: float,
    atr: float,
    reason: str,
):
    cash = ctx.state.portfolio.cash
    if not (atr > 0 and close > 0 and cash > 0):
        return
    stop_dist = p.disaster_atr_mult * atr
    ctx.short(
        sym,
        size=p.risk_pct,
        size_mode="equity",
        sl=stop_dist / close,
        reason=reason,
    )
