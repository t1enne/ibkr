"""MFI-exhaustion short (marginal high on dried volume + MFI rollover).

Core MFI bearish-divergence idea, traded as an *exhaustion top* rather than a
lag-confirmed 2nd fractal pivot. Short the fresh marginal higher high that
prints on *drying* positive volume while MFI has topped and rolls down from a
recent overbought peak. Price extends but participation thins -> the buyers
that would sustain the breakout are absent.

Rule (all must hold at the current bar, flat; no trend/regime gate):
  1. close prints a fresh marginal high over ``fresh_look`` closes;
  2. MFI peaked >= ``ob_floor`` within the last ``mfi_top_look`` bars;
  3. MFI has rolled off that peak by >= ``mfi_gap`` (bearish divergence:
     price HH while money-flow strength LH);
  4. current bar (the one making the high) volume is dry -- <= ``vol_dry`` x
     the mean volume over the leg that formed the run (buyers exhausted).

Chandelier ATR trail exit with a structural stop above entry.
"""

from __future__ import annotations
import dataclasses
from dataclasses import dataclass
import numpy as np
from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "mfi_exhaustion_dsl"
_STATE_KEY = "mfi_exh_state"


def _meltup_upday_mag(closes: np.ndarray, look: int) -> float:
    """Mean % size of up-close days over the trailing ``look`` closed bars.

    A stable, per-name "melt-up intensity" reading: ripper names (PLTR/MU/AMD)
    print persistently larger avg up-days (>= ~2.1-2.2%) than rollover names whose
    exhaustion tops genuinely fade (NVO/SNPS/QCOM ~1.4-2.1%), and the gap holds
    across regimes. Returns 0.0 (admissible) when there is not enough closed
    history or no up-days in the window.
    """
    if look < 2 or len(closes) <= look:
        return 0.0
    # trailing ``look`` fully-closed bars (drop the current, unconfirmed candle)
    window = closes[-look - 1 : -1]
    base = window[:-1].copy()
    base[base <= 0] = np.nan
    rets = np.diff(window) / base  # per-bar fractional change
    ups = rets[np.isfinite(rets) & (rets > 0)]
    if ups.size == 0 or not np.any(np.isfinite(ups)):
        return 0.0
    return float(np.nanmean(ups) * 100.0)


@dataclass(frozen=True)
class _State:
    best: float | None = None  # running best (lowest) close since entry (None flat)


@dataclass(frozen=True)
class Params(StrategyParams):
    # -- exhaustion trigger --
    mfi_period: int = 14
    mfi_top_look: int = 20  # bars back to find the overbought MFI peak
    ob_floor: float = 70.0  # peak MFI must have reached at least this
    mfi_gap: float = 8.0  # current MFI must be this far under the peak
    fresh_look: int = 10  # marginal-high: current close tops last N closes
    leg_look: int = 40  # length of the run forming the top (volume baseline)
    vol_dry: float = 0.7  # current bar vol <= this x mean leg volume
    # -- regime gate (0 = off): don't short a melt-up far above its longer
    #    structural trend (that is where exhaustion-top fades die). NOTE: these
    #    price-vs-structure gates are known to delete the GOOD fresh-breakdown
    #    reversals (a breakdown top prints near/above its 200d) -- prefer the
    #    ``meltup_*`` character gate below, which is uncorrelated with them.
    regime_sma: int = 200  # longer structure (e.g. 200d) used to gauge regime
    regime_max_ext: float = 6.0  # allow shorts while close <= sma + this x ATR
    # -- melt-up "ripper" gate (default off: meltup_upday = 0.0). A persistent
    #    per-name character filter that blocks SHORTING names whose daily up-moves
    #    run too large -- the melt-up "ripper" cohort (PLTR/MU/AMD/...) that keeps
    #    ripping after every exhaustion top, so shorts on it bleed. Unlike the
    #    structure gates above, up-day magnitude does NOT co-vary with price-vs-MA
    #    at the top, so genuine rollover names (NVO/SNPS/QCOM/NXPI/ADI), whose
    #    exhaustion tops also print near/above the 200d, stay admissible. Measured
    #    trailing over ``meltup_look`` closed bars (no lookahead).
    meltup_look: int = 200  # closes over which to average up-day move size
    meltup_upday: float = 0.0  # reject short when avg up-day >= this %% (0 = off)
    # -- chandelier trailing management --
    atr_period: int = 14
    trail_atr_mult: float = 2.0  # ATRs off running best close that bank trade
    entry_stop_atr: float = 3.0  # initial stop -- ATRs above entry (bracket)
    # -- risk sizing --
    risk_pct: float = 0.05
    warmup_bars: int = 60


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

    # ---- exit while short: chandelier ATR trailing ----
    if ctx.quantity(sym) != 0:
        best = state.best
        if best is None or not (best > 0):
            put(best=close)
            return
        if close < best - 1e-9:
            put(best=close)
            return
        if atr > 0 and close >= best + p.trail_atr_mult * atr:
            put(best=None)
            reason = (
                f"[mfi_exh] trail tp {sym}: close {close:.2f} "
                f"{p.trail_atr_mult:.1f}ATR off best {best:.2f}"
            )
            ctx.close(sym, reason=reason)
        return

    # ---- regime gate (0 period = off) ----
    if p.regime_sma > 0:
        if n <= p.regime_sma:
            return
        sma_arr = ctx.ta.sma(sym, period=p.regime_sma).to_array()
        if len(sma_arr) != n:
            return
        sma_now = float(sma_arr[-1])
        if not np.isfinite(sma_now):
            return
        # cap extension above the long structure: reject far-extended melt-ups
        if atr > 0 and close > sma_now + p.regime_max_ext * atr:
            return

    # ---- melt-up "ripper" character gate (default off) ----
    if p.meltup_upday > 0.0:
        # trailing mean up-day magnitude over closed bars (excludes the current,
        # unconfirmed bar so a gap/overnight move on the entry day cannot veto).
        _m = _meltup_upday_mag(closes, p.meltup_look)
        if _m >= p.meltup_upday:
            return

    # ---- entry: exhaustion trigger (regime-gated) ----
    if state.best is None and _exhaustion_top(o, p, mfi_vals, n):
        _enter_short(ctx, p, sym, close, atr)


def _exhaustion_top(o, p: Params, mfi_vals: np.ndarray, n: int) -> bool:
    """True when current bar is a light-volume marginal high off an MFI rollover."""
    if n < p.warmup_bars + 2:
        return False
    closes = o.close.to_array()
    vols = o.volume.to_array()
    if len(closes) != n or len(vols) != n:
        return False

    # (1) fresh marginal close high over the last `fresh_look` closes (excl current)
    fresh_span = closes[n - 1 - p.fresh_look : n - 1]
    if not np.all(np.isfinite(fresh_span)) or len(fresh_span) == 0:
        return False
    if not (closes[n - 1] > float(np.nanmax(fresh_span))):
        return False

    # (2+3) MFI peaked >= ob_floor within mfi_top_look bars (behind current),
    #       now rolled off the peak by >= mfi_gap.
    mfi_hist = mfi_vals[n - 1 - p.mfi_top_look : n]  # includes current
    past = mfi_hist[:-1]
    if not np.all(np.isfinite(past)) or len(past) == 0:
        return False
    peak = float(np.nanmax(past))
    cur_mfi = float(mfi_hist[-1])
    if not np.isfinite(peak):
        return False
    if peak < p.ob_floor:
        return False
    if cur_mfi > peak - p.mfi_gap:
        return False  # MFI has not rolled far enough yet

    # (4) volume dryness: the advancing bar's volume <= vol_dry x mean of the leg
    leg_start = max(0, n - p.leg_look - 1)
    leg_vols = vols[leg_start : n - 1]
    if len(leg_vols) == 0 or not np.all(np.isfinite(leg_vols)):
        return False
    mean_leg = float(np.nanmean(leg_vols))
    if mean_leg <= 0:
        return False
    cur_vol = float(vols[n - 1])
    if not np.isfinite(cur_vol) or cur_vol <= 0:
        return False
    if cur_vol > p.vol_dry * mean_leg:
        return False

    return True


def _enter_short(
    ctx: StrategyContext, p: Params, sym: str, close: float, atr: float
) -> None:
    cash = ctx.state.portfolio.cash
    if not (atr > 0 and close > 0 and cash > 0):
        return
    stop_dist = p.entry_stop_atr * atr
    ctx.short(
        sym,
        size=p.risk_pct,
        size_mode="equity",
        sl=stop_dist / close,
        reason=(
            f"[mfi_exh] short {sym}: marginal HH {close:.2f} on dried vol "
            f"with MFI rollover (gap={p.mfi_gap:.1f})"
        ),
    )
