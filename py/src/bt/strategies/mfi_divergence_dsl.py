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
from src.bt.strategies.utils import gross_gate

STRATEGY_TYPE = "mfi_divergence_dsl"
_STATE_KEY = "mfi_divergence_state"


@dataclass(frozen=True)
class _State:
    # running best (lowest) close since a long entry; None while flat/long-off
    best_long: float | None = None
    # running worst (highest) close since a short entry; None while flat/short-off
    best_short: float | None = None
    # True while the previous bar held an open position; used to stamp the bar
    # at which this symbol went flat and start the re-entry cooldown.
    was_in_position: bool = False
    # bar index (len(close) of the bar) at which the position was closed; -1
    # means never / no cooldown armed. Entry is blocked while
    # ``n - last_exit_bar < cooldown_bars``.
    last_exit_bar: int = -1


@dataclass(frozen=True)
class Params(StrategyParams):
    # -- MFI / divergence --
    mfi_period: int = 14
    div_look: int = 30  # divergence swing window (split prior|recent halves)
    div_min_gap: float = 8.0  # required MFI higher-low lead over the prior swing
    # Required *price* extension between the two swings, as a fraction (0.05 =
    # recent swing extreme must overshoot the prior by 5%). This is the
    # load-bearing quality gate: an mfi-gap-only trigger fires on ~every bar
    # (measured mean forward return ~0), while demanding the price itself thrust
    # past the prior extreme makes the divergence non-trivial (mean fwd10 rises
    # monotonically with this floor). 0.0 disables it.
    div_min_price_ext: float = 0.0
    # -- side admission --
    allow_longs: bool = True
    allow_shorts: bool = True
    # Bars to stay flat after a position on the same symbol closes (any reason:
    # trail, stop, end) before re-entry is allowed. 0 disables the cooldown.
    # Suppresses re-shorting/re-longing the same persistent signal right after
    # an exit (measured: the stale-signal re-entry loop on V-reversal names was
    # the single largest per-name loss driver).
    cooldown_bars: int = 0
    # -- per-symbol trend sign gate (0 look = off) --
    sym_roc_look: int = 0  # bars for the symbol's own ROC
    sym_roc_min: float = 0.0  # |ROC| (fraction) needed to admit a side
    # -- moving-average regime gates --
    # Longs are admitted only when price is *above* its MA, shorts only when
    # below -- i.e. the divergence is traded with the prevailing trend, not
    # against it.
    # ``spy_ma_gate``: gate on the benchmark (SPY) vs its own MA (market regime).
    # ``symbol_ma_gate``: gate on the traded symbol vs its own MA.
    # ``need_ma_alignment``: require *both* the SPY and the symbol MA to agree
    # with the trade side (implies both individual gates regardless of their
    # flags). SPY and symbol lengths are swept independently via
    # ``spy_ma_period`` / ``symbol_ma_period``.
    spy_ma_gate: bool = False
    symbol_ma_gate: bool = False
    need_ma_alignment: bool = False
    spy_ma_period: int = 200
    symbol_ma_period: int = 200
    benchmark: str = "SPY"
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
    # as a fraction of **live equity** (1.0 = off).
    max_gross_exposure: float = (
        1.0  # aggregate |notional| cap, fraction of live equity (1.0 = off)
    )


@strategy(bars="1d", stateful=True)
def on_candle(ctx: StrategyContext):
    # Warmup bars fill strategy state only — no fill can happen there, so the
    # DSL raises on any entry emitted during them. Accumulators above/below
    # still receive warmup bars because the engine feeds the store regardless.
    if ctx.phase == "warmup":
        return
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
    if qty != 0:
        put(was_in_position=True)
        if qty > 0:
            _trail_long(ctx, p, state, put, sym, close, atr)
        else:
            _trail_short(ctx, p, state, put, sym, close, atr)
        return

    # Flat. If the previous bar held a position, the close happened since then
    # (trail, stop, or end): stamp this bar and start the cooldown window.
    if state.was_in_position:
        state = dataclasses.replace(state, was_in_position=False, last_exit_bar=n)
        holder[sym] = state
    # ---- flat: decide the admissible side from the symbol's own trend ----
    sym_roc = _roc(ctx, sym, p.sym_roc_look) if p.sym_roc_look > 0 else 0.0
    if p.sym_roc_look > 0 and not np.isfinite(sym_roc):
        return
    long_ok = p.allow_longs and ((p.sym_roc_look == 0) or (sym_roc >= p.sym_roc_min))
    short_ok = p.allow_shorts and ((p.sym_roc_look == 0) or (sym_roc <= -p.sym_roc_min))

    # ---- moving-average regime gates ----
    spy_up = _above_ma(ctx, p.benchmark, p.spy_ma_period)
    sym_up = _above_ma(ctx, sym, p.symbol_ma_period)
    long_ok, short_ok = _ma_side_ok(
        long_ok,
        short_ok,
        spy_up,
        sym_up,
        p.spy_ma_gate,
        p.symbol_ma_gate,
        p.need_ma_alignment,
    )

    # ---- cooldown: block re-entry too soon after the last exit on this name ----
    if _cooling(n, state.last_exit_bar, p.cooldown_bars):
        return

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


def _cooling(now: int, last_exit_bar: int, cooldown_bars: int) -> bool:
    """True while a re-entry on this symbol must be blocked.

    ``now`` is the current bar index (``len(close)``) and ``last_exit_bar`` the
    bar at which the last position closed (``-1`` = never held one). Blocks
    while fewer than ``cooldown_bars`` bars have elapsed since the exit; a
    non-positive ``cooldown_bars`` disables the gate entirely.
    """
    if cooldown_bars <= 0 or last_exit_bar < 0:
        return False
    return now - last_exit_bar < cooldown_bars


def _above_ma(ctx: StrategyContext, sym: str, period: int) -> bool | None:
    """True when ``sym``'s latest close is above its ``period``-bar SMA.

    Returns ``None`` when history is too short or either value is non-finite,
    so every caller fails closed (a gate must not admit a side it cannot check).
    """
    if period <= 0:
        return None
    o = ctx.ohlcv(sym)
    if o is None or len(o.close) < period:
        return None
    cur = float(o.close.to_array()[-1])
    ma = ctx.ta.sma(sym, period=period).last()
    if not (np.isfinite(cur) and np.isfinite(ma)) or ma <= 0:
        return None
    return cur > ma


def _ma_side_ok(
    long_ok: bool,
    short_ok: bool,
    spy_up: bool | None,
    sym_up: bool | None,
    spy_ma_gate: bool,
    symbol_ma_gate: bool,
    need_ma_alignment: bool,
) -> tuple[bool, bool]:
    """Apply the MA regime gates to the admissible sides (pure).

    ``spy_up`` / ``sym_up`` are the ``_above_ma`` results (``None`` = unknown,
    fails closed). ``need_ma_alignment`` overrides both individual flags and
    requires benchmark and symbol to trend with the side.
    """
    if need_ma_alignment:
        return (
            long_ok and spy_up is True and sym_up is True,
            short_ok and spy_up is False and sym_up is False,
        )
    if spy_ma_gate:
        long_ok = long_ok and spy_up is True
        short_ok = short_ok and spy_up is False
    if symbol_ma_gate:
        long_ok = long_ok and sym_up is True
        short_ok = short_ok and sym_up is False
    return long_ok, short_ok


def _bull_divergence(o, p: Params, mfi_vals: np.ndarray, n: int) -> bool:
    """Long trigger: price lower-low while MFI higher-low (bullish absorption).

    Splits the last ``div_look`` bars into prior|recent halves and demands the
    recent half's price low is *below* the prior half's while its MFI low is
    *above* (by at least ``div_min_gap``) -- sellers are being absorbed. The
    price low must also clear ``div_min_price_ext`` below the prior low, so a
    divergence against an essentially flat low does not fire.
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
    if not (rl < pl and rm > pm + p.div_min_gap):
        return False
    # Price must actually extend below the prior low by the required fraction:
    # a divergence against a barely-lower low is noise, not absorption.
    if p.div_min_price_ext > 0:
        ext = (pl - rl) / pl if pl > 0 else 0.0
        if ext < p.div_min_price_ext:
            return False
    return True


def _bear_divergence(o, p: Params, mfi_vals: np.ndarray, n: int) -> bool:
    """Short trigger: price higher-high while MFI lower-high (bearish distribution).

    Mirrors the bull gate, including the ``div_min_price_ext`` requirement.
    """
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
    if not (rh > ph and rm < pm - p.div_min_gap):
        return False
    # Price must actually extend above the prior high by the required fraction.
    if p.div_min_price_ext > 0:
        ext = (rh - ph) / ph if ph > 0 else 0.0
        if ext < p.div_min_price_ext:
            return False
    return True


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
    # -- drawdown control: aggregate gross-notional cap vs live equity --
    # The candidate lot is ``risk_pct`` of live equity, already equity-relative.
    equity = ctx.current_equity()
    if gross_gate(
        equity, ctx.state.portfolio.positions, p.max_gross_exposure, p.risk_pct
    ):
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
