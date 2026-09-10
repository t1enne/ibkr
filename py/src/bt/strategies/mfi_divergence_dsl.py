"""MFI divergence reversal, long+short.

Entry thesis: **price/flow divergence** -- price prints a lower low while MFI
prints a higher low (bullish absorption: sellers are being absorbed), or the
mirror (price higher high, MFI lower high = distribution).

Why this shape (measured, not assumed):

  * The original *level* filter ("MFI oversold trough in the window, now
    lifting") is net **value-destroying** on top of the raw reversal core: an
    MFI oversold trough appears on ~65% of bars, and requiring a *deeper*
    trough lowered forward returns. Removed.
  * MFI **divergence** is non-redundant with price and was robust across a 2D
    (``div_look`` x ``div_min_gap``) plateau -- the level filter was not.
  * Benchmark overlays (price-vs-EMA and bench-momentum gates) were swept and
    are **net-harmful** in every setting tested (best gated Sharpe well below
    ungated). Removed -- no regime gate.

Trend handling is the ``sym_roc_look`` / ``sym_roc_min`` sign gate: a name
net-rising admits longs (a divergence low in an uptrend is a dip), a name
net-falling admits shorts. ``sym_roc_look=0`` disables the sign gate. The
filter's *value is regime-dependent* (helps in sustained trend, neutral in
chop) -- pass the appropriate look per config rather than assuming one default.

Per-side chandelier ATR trailing exits and structural entry stops. Per-symbol
state in ``ctx.shared``; every read is cursor-truncated (no lookahead).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.types import StrategyParams
from src.bt.strategies.vp_breakout_dsl import _gross_exposure

STRATEGY_TYPE = "mfi_divergence_dsl"
_STATE_KEY = "mfi_divergence_state"


@dataclass(frozen=True)
class _State:
    # running best (lowest) close since a long entry; None while flat/long-off
    best_long: float | None = None
    # running worst (highest) close since a short entry; None while flat/short-off
    best_short: float | None = None


@dataclass(frozen=True)
class Params(StrategyParams):
    # -- MFI / divergence --
    mfi_period: int = 14
    div_look: int = 30  # divergence swing window (split prior|recent halves)
    div_min_gap: float = 8.0  # required MFI higher-low lead over the prior swing
    # -- side admission --
    allow_longs: bool = True
    allow_shorts: bool = True
    # -- per-symbol trend sign gate (0 look = off) --
    sym_roc_look: int = 0  # bars for the symbol's own ROC
    sym_roc_min: float = 0.0  # |ROC| (fraction) needed to admit a side
    # -- risk / management --
    atr_period: int = 14
    trail_atr_mult: float = 1.2  # ATRs off the running extreme that bank trade
    entry_stop_atr: float = 3.0  # initial stop, ATRs beyond entry
    risk_pct: float = 0.4
    warmup_bars: int = 60
    # -- drawdown control --
    # Per-name notional is ``risk_pct`` of live equity, so a correlated cluster
    # can dominate the book (measured: DD -49% on 6y at risk_pct=0.4).
    # ``risk_pct`` is the primary DD lever; this caps aggregate |gross| notional
    # as a fraction of initial capital (1.0 = off).
    max_gross_exposure: float = (
        1.0  # aggregate |notional| cap, fraction of initial capital (1.0 = off)
    )


@strategy(bars="1d", stateful=True)
def on_candle(ctx: StrategyContext):
    p: Params = ctx.params
    for sym in ctx.symbols:
        _process_symbol(ctx, p, sym)


def _roc(ctx: StrategyContext, sym: str, look: int) -> float:
    """Trailing ``look``-bar rate of change of ``sym``'s close (fraction).

    Cursor-truncated. NaN when history is short or the reference close is
    non-finite, so every caller fails closed.
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

    mfi_vals = ctx.ta.mfi(sym, period=p.mfi_period).to_array()
    if n != len(mfi_vals) or n <= 0:
        return

    atr = ctx.ta.atr(sym, period=p.atr_period).last()
    if not np.isfinite(atr):
        atr = 0.0

    # ---- manage an open position: chandelier ATR trail on the active side ----
    qty = ctx.quantity(sym)
    if qty > 0:
        _trail_long(ctx, p, state, put, sym, close, atr)
        return
    if qty < 0:
        _trail_short(ctx, p, state, put, sym, close, atr)
        return

    # ---- flat: decide the admissible side from the symbol's own trend ----
    sym_roc = _roc(ctx, sym, p.sym_roc_look) if p.sym_roc_look > 0 else 0.0
    if p.sym_roc_look > 0 and not np.isfinite(sym_roc):
        return
    long_ok = p.allow_longs and ((p.sym_roc_look == 0) or (sym_roc >= p.sym_roc_min))
    short_ok = p.allow_shorts and ((p.sym_roc_look == 0) or (sym_roc <= -p.sym_roc_min))

    # ---- entry: whichever divergence trigger fires for an admissible side ----
    if long_ok and _bull_divergence(o, p, mfi_vals, n):
        _enter(ctx, p, sym, close, atr, side="long")
        return
    if short_ok and _bear_divergence(o, p, mfi_vals, n):
        _enter(ctx, p, sym, close, atr, side="short")


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
                f"[mfi_div] trail tp long {sym}: close {close:.2f} "
                f"{p.trail_atr_mult:.1f}ATR under best {best:.2f}"
            ),
        )


def _trail_short(
    ctx: StrategyContext,
    p: Params,
    state: _State,
    put,
    sym: str,
    close: float,
    atr: float,
) -> None:
    best = state.best_short
    if best is None or not (best > 0):
        put(best_short=close)
        return
    if close < best - 1e-9:
        put(best_short=close)
        return
    if atr > 0 and close >= best + p.trail_atr_mult * atr:
        put(best_short=None)
        ctx.close(
            sym,
            reason=(
                f"[mfi_div] trail tp short {sym}: close {close:.2f} "
                f"{p.trail_atr_mult:.1f}ATR over best {best:.2f}"
            ),
        )


def _bull_divergence(o, p: Params, mfi_vals: np.ndarray, n: int) -> bool:
    """Long trigger: price lower-low while MFI higher-low (bullish absorption).

    Splits the last ``div_look`` bars into prior|recent halves and demands the
    recent half's price low is *below* the prior half's while its MFI low is
    *above* (by at least ``div_min_gap``) -- sellers are being absorbed.
    """
    look = p.div_look
    if n < p.warmup_bars + 2 or look < 2:
        return False
    closes = o.close.to_array()
    if len(closes) != n or len(mfi_vals) != n:
        return False
    half = look // 2
    prior_lo = closes[n - look : n - half]
    rec_lo = closes[n - half : n]
    prior_mo = mfi_vals[n - look : n - half]
    rec_mo = mfi_vals[n - half : n]
    if not (len(prior_lo) and len(rec_lo)):
        return False
    if not (
        np.all(np.isfinite(prior_lo))
        and np.all(np.isfinite(rec_lo))
        and np.all(np.isfinite(prior_mo))
        and np.all(np.isfinite(rec_mo))
    ):
        return False
    pl = float(np.nanmin(prior_lo))
    rl = float(np.nanmin(rec_lo))
    pm = float(np.nanmin(prior_mo))
    rm = float(np.nanmin(rec_mo))
    return bool(rl < pl and rm > pm + p.div_min_gap)


def _bear_divergence(o, p: Params, mfi_vals: np.ndarray, n: int) -> bool:
    """Short trigger: price higher-high while MFI lower-high (bearish distribution)."""
    look = p.div_look
    if n < p.warmup_bars + 2 or look < 2:
        return False
    closes = o.close.to_array()
    if len(closes) != n or len(mfi_vals) != n:
        return False
    half = look // 2
    prior_hi = closes[n - look : n - half]
    rec_hi = closes[n - half : n]
    prior_mo = mfi_vals[n - look : n - half]
    rec_mo = mfi_vals[n - half : n]
    if not (len(prior_hi) and len(rec_hi)):
        return False
    if not (
        np.all(np.isfinite(prior_hi))
        and np.all(np.isfinite(rec_hi))
        and np.all(np.isfinite(prior_mo))
        and np.all(np.isfinite(rec_mo))
    ):
        return False
    ph = float(np.nanmax(prior_hi))
    rh = float(np.nanmax(rec_hi))
    pm = float(np.nanmax(prior_mo))
    rm = float(np.nanmax(rec_mo))
    return bool(rh > ph and rm < pm - p.div_min_gap)


def _enter(
    ctx: StrategyContext,
    p: Params,
    sym: str,
    close: float,
    atr: float,
    side: str,
) -> None:
    if not (atr > 0 and close > 0 and ctx.state.portfolio.cash > 0):
        return
    # -- drawdown control: aggregate gross-notional cap (1.0 = off) --
    if 0.0 < p.max_gross_exposure < 1.0:
        equity = ctx.current_equity()
        init = ctx.state.portfolio.initial_capital
        # DSL sizes the candidate at ``risk_pct * equity`` of notional; express
        # it as a fraction of initial capital to match ``_gross_exposure``.
        candidate = p.risk_pct * equity / init if init > 0 else 0.0
        gross = _gross_exposure(init, ctx.state.portfolio.positions)
        if gross + candidate >= p.max_gross_exposure:
            return
    stop = p.entry_stop_atr * atr / close
    if side == "long":
        ctx.long(
            sym,
            size=p.risk_pct,
            size_mode="equity",
            sl=stop,
            reason=(
                f"[mfi_div] long {sym}: bullish MFI divergence {close:.2f} "
                f"(gap={p.div_min_gap:.1f})"
            ),
        )
    else:
        ctx.short(
            sym,
            size=p.risk_pct,
            size_mode="equity",
            sl=stop,
            reason=(
                f"[mfi_div] short {sym}: bearish MFI divergence {close:.2f} "
                f"(gap={p.div_min_gap:.1f})"
            ),
        )
