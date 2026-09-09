"""Shallow-pullback long: fresh marginal low on drying volume in an uptrend.

The distilled core of the ``mfi_reversal_ls_dsl`` long side, with all MFI
machinery removed. Measured findings that drove the split:

  * The MFI oversold-trough level filter (``trough <= os_floor`` + lift-off
    ``gap``) is net **value-destroying** on top of the fresh-low core: every
    removal of MFI improved per-trade return, and the tighter the required
    trough, the worse forward returns -- it excludes the mild dips that bounce
    best. Its replacement (price/MFI *divergence*) lives in the sibling
    ``mfi_swingdiv_reversal_ls_dsl``.
  * What remains is a plain, defensible thesis: **buy a shallow dip in a
    strong name**. A fresh marginal low on drying volume, admitted only when
    the name's own trailing ROC is positive (uptrend -> a dip; a downtrend
    fresh low is a knife, not a dip).

Long-only by construction. Each name trades at most one position at a time;
per-symbol trail state lives in ``ctx.shared`` (fresh per run, safe across
worker processes). Every read is cursor-truncated -- no lookahead.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "pullback_freshlow_dsl"
_STATE_KEY = "pullback_freshlow_state"


@dataclass(frozen=True)
class _State:
    # running best (highest) close since a long entry; None while flat
    best_long: float | None = None


@dataclass(frozen=True)
class Params(StrategyParams):
    # -- trend gate (the robust edge; 0 look = off) --
    sym_roc_look: int = 120  # bars for the symbol's own ROC
    sym_roc_min: float = 0.05  # min ROC (fraction) to admit a long
    # -- pullback trigger --
    fresh_look: int = 5  # today's close must be a fresh low over this window
    leg_look: int = 40  # volume baseline window
    vol_dry: float = 1.0  # today's volume <= this x mean leg volume
    # -- risk / management --
    atr_period: int = 7
    trail_atr_mult: float = 1.2  # ATRs off the running high that bank a trade
    entry_stop_atr: float = 3.0  # initial stop, ATRs beyond entry
    risk_pct: float = 0.4
    warmup_bars: int = 60


@strategy(bars="1d", stateful=True)
def on_candle(ctx: StrategyContext):
    p: Params = ctx.params
    for sym in ctx.symbols:
        _process_symbol(ctx, p, sym)


def _roc(ctx: StrategyContext, sym: str, look: int) -> float:
    """Trailing ``look``-bar rate of change of ``sym``'s close (fraction).

    Cursor-truncated. NaN when history is short or a reference close is
    non-finite, so callers fail closed.
    """
    if look <= 0:
        return float("nan")
    o = ctx.ohlcv(sym)
    if o is None or len(o.close) <= look:
        return float("nan")
    arr = o.close.to_array()
    ref = float(arr[-1 - look])
    cur = float(arr[-1])
    if not (np.isfinite(ref) and np.isfinite(cur)) or ref <= 0:
        return float("nan")
    return cur / ref - 1.0


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
    close = float(closes[-1])

    atr = ctx.ta.atr(sym, period=p.atr_period).last()
    if not np.isfinite(atr):
        atr = 0.0

    # ---- manage an open position: chandelier ATR trail ----
    qty = ctx.quantity(sym)
    if qty > 0:
        _trail_long(ctx, p, state, put, sym, close, atr)
        return

    # ---- flat: decide admission from the symbol's own trend ----
    sym_roc = _roc(ctx, sym, p.sym_roc_look)
    if not np.isfinite(sym_roc) or sym_roc < p.sym_roc_min:
        return

    if _fresh_pullback(o, p, n):
        _enter(ctx, p, sym, close, atr)


def _trail_long(
    ctx: StrategyContext,
    p: Params,
    state: _State,
    put,
    sym: str,
    close: float,
    atr: float,
) -> None:
    best = state.best_long
    if best is None or not (best > 0):
        put(best_long=close)
        return
    if close > best + 1e-9:
        put(best_long=close)
        return
    if atr > 0 and close <= best - p.trail_atr_mult * atr:
        put(best_long=None)
        ctx.close(
            sym,
            reason=(
                f"[pullback] trail tp long {sym}: close {close:.2f} "
                f"{p.trail_atr_mult:.1f}ATR under best {best:.2f}"
            ),
        )


def _fresh_pullback(o, p: Params, n: int) -> bool:
    """True on a fresh marginal low over ``fresh_look`` on drying volume."""
    if n < p.warmup_bars + 2:
        return False
    closes = o.close.to_array()
    vols = o.volume.to_array()
    if len(closes) != n or len(vols) != n:
        return False

    fresh = closes[n - 1 - p.fresh_look : n - 1]
    if len(fresh) == 0 or not np.all(np.isfinite(fresh)):
        return False
    if not (closes[n - 1] < float(np.nanmin(fresh))):
        return False

    return _vol_dry(o, p, n)


def _vol_dry(o, p: Params, n: int) -> bool:
    """Current bar's volume <= ``vol_dry`` x the mean over the prior leg."""
    vols = o.volume.to_array()
    leg_start = max(0, n - p.leg_look - 1)
    leg = vols[leg_start : n - 1]
    if len(leg) == 0 or not np.all(np.isfinite(leg)):
        return False
    mean_leg = float(np.nanmean(leg))
    if mean_leg <= 0:
        return False
    cur_vol = float(vols[n - 1])
    if not np.isfinite(cur_vol) or cur_vol <= 0:
        return False
    return cur_vol <= p.vol_dry * mean_leg


def _enter(ctx: StrategyContext, p: Params, sym: str, close: float, atr: float) -> None:
    if not (atr > 0 and close > 0 and ctx.state.portfolio.cash > 0):
        return
    stop = p.entry_stop_atr * atr / close
    ctx.long(
        sym,
        size=p.risk_pct,
        size_mode="equity",
        sl=stop,
        reason=(
            f"[pullback] long {sym}: fresh marginal low {close:.2f} "
            f"on dried vol in uptrend (ROC>={p.sym_roc_min:.2f})"
        ),
    )
