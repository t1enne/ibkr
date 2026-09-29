"""VWATR divergence strategy -- long only. Same skeleton as vwatr_dsl; adds
THREE divergence exits vwatr_dsl does not use. No entry filter (entry features
tested correlate ~0 with trade return), so this module changes ONLY the exit and
the re-entry gate.

VWATR itself (``smooth(TR * volume) / smooth(volume)``) lives in the shared TA
layer now: ``ctx.ta.vwatr`` and ``ctx.ta.vwatr_baseline``. The ``atr_ratio``
divergence path uses ``ctx.ta.plain_atr`` -- a plain rolling-mean ATR, NOT the
Wilder-smoothed ``ctx.ta.atr``.

LEDGER -- sizing, not gating, is the lever. Read before touching params.
CONFIG: strats/pass/vwatr_div_exp8_6y_risk0.08.json (8 high-vol names, 1d, 6y).
Beats SPY by a large multiple at SR > 1 at roughly HALF the drawdown of an
equal-weight buy-and-hold of the same names. High kurtosis (fat tails).
`risk_pct` was the real bug -- hardcoded low, never swept. Its 6y response is
UNIMODAL with an INTERIOR peak (peak != boundary), so the param is real; but the
shipped value sits on that plateau within noise and the risk-adjusted optimum
(Calmar) is LOWER than the return-max point. Treat it as "in the plateau", never
an argmax -- do not re-tune to it. Contrast `decel_ratio`, MONOTONE-decaying
toward its boundary (always want smaller) = degenerate knob. Compounding
(`size_mode="equity"`) supplies the exponential curve SHAPE, but the signal stays
profitable at the smallest sizing, so sizing SCALES the risk-adjusted edge rather
than manufacturing it. WALK-FORWARD passes (`bt split --folds 2|3`); `bt optimize`
on risk_pct FAILS as expected -- thin early IS makes it pick a grid boundary.
TWO ADVERSE FACTS, neither fatal, both must be disclosed:
  - TAIL CONCENTRATION. Profit factor is unremarkable and the MEDIAN trade is far
    below the MEAN: a handful of winners carries most of the gross profit. Drop
    the top few and total return collapses toward flat -- a fat-right-tail bet,
    not a broad edge. Symbol/year bootstraps CANNOT see this (they resample names
    and regimes, never trades).
  - SYMBOL DEPENDENCE. Drop-one shows the worst single-symbol omission drops SR to
    around SPY's: one name IS load-bearing. (Drafts claimed "no single symbol is
    load-bearing / worst = drop ENPH"; both WRONG -- ENPH is among the LEAST
    damaging to drop.)
REGIME: most of the P&L comes from the last third of the window -- a late-window
/ high-beta-bull result, not all-weather.

NO GATE EXISTS -- proven by a pre-registered search; don't repeat it. Dozens of
external macro/breadth/vol series were tested against the mid-sample flat hole and
NONE cleared p<0.05 (artifacts: strats/wip/vwatr_div_gate_research/). WHY none CAN
work: the hole is FLAT, not losing -- the missing mass is the FAT RIGHT TAIL. A
gate filters ENTRIES; it cannot create an absent melt-up. Sizing scales a tail
that IS present -- hence risk_pct works and every gate died. Dead gate ideas, all
failing the trade-deletion / fold test: SPY SMA(200)+vol filter, VWATR coil
precondition, `baseline_gate` both ways, `macro_gate` (gains only by deleting
trades or acting as a calendar switch -- DD-per-trade WORSENS). Any future search
must control for the CALENDAR PLACEBO: the hole is a mid-sample era, so every slow
level series "separates" by construction -- a raw level variable at small |corr|
is a date bet in disguise.

WHAT THE EXITS DO.
1. SLOPE DIVERGENCE -- REAL MECHANISM, DEGENERATE PARAM. Base uses only
   sign(slope); here slope is measured twice and the exit fires when price is at
   running-best AND slope_now < decel_ratio*slope_prev. vs the base it RAISES
   trade count and win rate while holding DD. NOT a disguised tighter trail
   (tighter vwatr_mult is strictly WORSE) -- it fires on slope rollover AT the
   high, not a price give-back. `decel_ratio` = fraction of prior slope still
   required (0.5 = slope must halve); LARGER fires EARLIER/more often, NOT never
   (slope_now is usually <= 0 when decelerating, so `decel_ratio >= 1` fires
   near-always -- the old "never fires" claim was WRONG). Monotone-declining from
   the low boundary => boundary value.
2/3. PRICE-UNIT DIVERGENCE (VWATR/close, VWATR/ATR) and BASELINE DIVERGENCE both
   FAILED: the former because a 1-bar delta flips sign constantly (noise, not
   divergence); the latter because `below` is re-timing and `above` worsens DD.
WIDER TRAIL IS WORSE -- a third independent contradiction of vwatr_dsl's "wider
monotonically better across 10 names". Flagged, not reconciled. Judge the TAIL
(kurtosis / worst DD per symbol), not only the mean.

HYGIENE. `decel_ratio=0.0` COLLINEAR with `div_exit=False` -> `__post_init__`
RAISES. Grid over `strategy_params` only. Opt-in `macro_gate` (default OFF; spec
format on Params) needs its symbols in config `symbols` before the benchmark,
never traded. `rearm_bars` DELETED (dead code). Use `bt split --folds 2|3` --
`oos_length="auto"` makes long windows / 4 folds = ~1y slices, and thin early IS
makes the optimizer pick boundary values and lose OOS.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from src.bt.size.pure import risk_sized_qty
from src.bt.state import ActionType
from src.bt.strategies.dsl import StrategyContext, strategy
from src.bt.strategies.types import (
    Marker,
    Overlay,
    Panel,
    PlotSpec,
    StrategyParams,
    ms,
    sparse,
)

STRATEGY_TYPE = "vwatr_div_dsl"
_STATE_KEY = "vwatr_div_state"

# Guard for the volume-weighted denominator (also used by ``plot``'s ratio
# panels): a window whose volume sums to ~0 has no meaningful VWATR.
_EPS = 1e-12


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(num.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > _EPS)
    out[ok] = num[ok] / den[ok]
    return out


@dataclass(frozen=True)
class _State:
    best: float | None = None


@dataclass(frozen=True)
class Params(StrategyParams):
    # -- VWATR (unchanged from vwatr_dsl) --
    vwatr_period: int = 14
    vwatr_base_win: int = 60
    vwatr_mult: float = 2.0
    slope_bars: int = 2
    breakout_look: int = 20
    risk_pct: float = 0.02
    # -- divergence layer --
    # 1: slope divergence exit. ``decel_ratio`` = fraction of the prior slope
    #    still required to hold. Must be > 0: the guard is ``decel_ratio > 0``,
    #    so 0.0 is collinear with ``div_exit=False`` (a degenerate grid point).
    decel_ratio: float = 0.5
    div_exit: bool = True  # enable the slope-divergence exit
    # 2: price-unit divergence exit. "none" | "pct_ratio" | "atr_ratio"
    norm_mode: str = "none"
    # 3: baseline divergence gate on entry. "none" | "above" | "below"
    baseline_gate: str = "none"
    # 4: MACRO REGIME gate (opt-in, default off => baseline unchanged).
    #    Each entry "SYMBOL:WINDOW:THRESHOLD" rejects an entry when the series'
    #    z-score vs its own trailing WINDOW bars is BELOW THRESHOLD. Series must
    #    be present in config.symbols and named with the GATE_ prefix.
    macro_gate: tuple[str, ...] = ()
    _gate_cache: dict = dataclasses.field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        # ``decel_ratio <= 0`` is collinear with ``div_exit=False`` (the exit
        # guard is ``p.decel_ratio > 0``). Reject it rather than let a grid
        # silently sweep a point that disables the very feature it names.
        if self.div_exit and self.decel_ratio <= 0:
            raise ValueError(
                "decel_ratio must be > 0 when div_exit=True; "
                "decel_ratio=0 is collinear with div_exit=False"
            )


def _macro_gate_open(ctx: StrategyContext, specs: tuple[str, ...]) -> bool:
    """True when EVERY macro gate spec passes on the current bar.

    Each spec is ``"SYMBOL:WINDOW:THRESHOLD"``: the series' close is z-scored
    against its own trailing ``WINDOW`` bars and the gate rejects (returns False)
    when that z-score is below ``THRESHOLD``. The z-score uses only bars up to
    and including the cursor, so it is look-ahead free; a series with too little
    history blocks the entry (conservative) rather than silently admitting it.
    """
    for spec in specs:
        try:
            sym, win_s, thr_s = spec.split(":")
            win, thr = int(win_s), float(thr_s)
        except ValueError:
            return False
        o = ctx.ohlcv(sym)
        if o is None or len(o.close) < win + 1:
            return False
        arr = o.close.to_array()
        window = arr[-win:]
        if not np.all(np.isfinite(window)):
            return False
        mu = float(np.mean(window))
        sd = float(np.std(window, ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            return False
        z = (float(arr[-1]) - mu) / sd
        if not (z >= thr):
            return False
    return True


@strategy(bars="1d", stateful=True)
def on_candle(ctx: StrategyContext) -> None:
    if ctx.phase == "warmup":
        return
    p: Params = ctx.params
    for sym in ctx.symbols:
        if sym.startswith("GATE_"):
            continue  # synthetic macro series: fed for the gate, never traded
        _process_symbol(ctx, p, sym)


def _slope_decelerating(
    vwatr_arr: np.ndarray, n: int, slope_bars: int, decel_ratio: float
) -> bool:
    """VWATR slope at t vs the slope one ``slope_bars`` window earlier."""
    if n <= 2 * slope_bars:
        return False
    i = n - 1
    v0 = float(vwatr_arr[i])
    v1 = float(vwatr_arr[i - slope_bars])
    v2 = float(vwatr_arr[i - 2 * slope_bars])
    if not (np.isfinite(v0) and np.isfinite(v1) and np.isfinite(v2)):
        return False
    slope_now = v0 - v1
    slope_prev = v1 - v2
    # Only meaningful while expansion is still positive; if VWATR is already
    # falling the trail (not this read) is the exit.
    if slope_prev <= 0:
        return False
    return slope_now < decel_ratio * slope_prev


def _norm_diverging(
    vwatr_arr: np.ndarray,
    atr_arr: np.ndarray,
    close: np.ndarray,
    n: int,
    norm_mode: str,
) -> bool:
    """True when VWATR is rising in dollars while falling as % of price.

    Compares the *change* of VWATR against the *change* of the normalized form
    over the same bar, so a level difference (always positive) cannot be
    mistaken for divergence.
    """
    if n < 3:
        return False
    if norm_mode == "pct_ratio":
        norm = _safe_div(vwatr_arr, close)
    elif norm_mode == "atr_ratio":
        norm = _safe_div(vwatr_arr, atr_arr)
    else:
        return False
    d_v = vwatr_arr[n - 1] - vwatr_arr[n - 2]
    d_n = norm[n - 1] - norm[n - 2]
    if not (np.isfinite(d_v) and np.isfinite(d_n)):
        return False
    # absolute dollars expanding, percentage range contracting
    return d_v > 0 and d_n < 0


def _process_symbol(ctx: StrategyContext, p: Params, sym: str) -> None:
    holder: dict[str, _State] = ctx.shared.setdefault(_STATE_KEY, {})
    state = holder.get(sym) or _State()
    holder[sym] = state

    # Macro regime gate: evaluated once per distinct macro_gate spec per bar,
    # cached in ctx.shared. A blocked bar blocks every symbol identically --
    # the gate is a regime read, not a per-name filter.
    gate_ok = True
    if p.macro_gate:
        gcache: dict = ctx.shared.setdefault("_macro_gate_state", {})
        # _macro_gate_memo maps the spec tuple -> (bar_count, verdict).
        n_now = len(ctx.ohlcv(sym).close) if ctx.ohlcv(sym) is not None else 0
        memo = gcache.get("memo")
        if memo is None or memo[0] != n_now:
            memo = (n_now, _macro_gate_open(ctx, p.macro_gate))
            gcache["memo"] = memo
        gate_ok = memo[1]

    def put(**kw: object) -> None:
        nonlocal state
        state = dataclasses.replace(state, **kw)
        holder[sym] = state

    o = ctx.ohlcv(sym)
    if o is None or len(o.close) == 0:
        return

    close = o.close.to_array()
    n = len(close)

    px = float(close[-1])
    if not np.isfinite(px) or px <= 0:
        return

    vwatr_arr = ctx.ta.vwatr(sym, p.vwatr_period).to_array()
    vwatr = float(vwatr_arr[-1]) if n else float("nan")
    if not np.isfinite(vwatr) or vwatr <= 0:
        return

    base_now = float(ctx.ta.vwatr_baseline(sym, p.vwatr_period, p.vwatr_base_win)[-1])

    # ---- exit while long ----
    if ctx.quantity(sym) != 0:
        best = state.best
        if best is None or not np.isfinite(best):
            put(best=px)
            return
        if px > best:
            put(best=px)
            best = px

        reason: str | None = None

        # 1) slope divergence under a new high
        if (
            p.div_exit
            and p.decel_ratio > 0
            and px >= best
            and _slope_decelerating(vwatr_arr, n, p.slope_bars, p.decel_ratio)
        ):
            reason = (
                f"[vwatr-div] slope decel {sym}: close {px:.2f} at best, "
                f"slope below {p.decel_ratio:.2f}x prior"
            )
        # 2) price-unit divergence
        elif p.norm_mode in ("pct_ratio", "atr_ratio"):
            # NOTE: the ``atr_ratio`` denominator is a PLAIN rolling-mean ATR
            # over ``vwatr_period`` (``ctx.ta.plain_atr``), deliberately NOT
            # Wilder-smoothed like ``ctx.ta.atr``. Swapping in ``ctx.ta.atr``
            # would change the exit and every downstream number.
            atr_arr = (
                ctx.ta.plain_atr(sym, p.vwatr_period).to_array()
                if p.norm_mode == "atr_ratio"
                else vwatr_arr
            )
            if _norm_diverging(vwatr_arr, atr_arr, close, n, p.norm_mode):
                reason = (
                    f"[vwatr-div] {p.norm_mode} diverge {sym}: close {px:.2f} "
                    f"at best, vwatr +{vwatr_arr[n - 1] - vwatr_arr[n - 2]:.4f} "
                    f"but ratio falling"
                )

        if reason is not None:
            put(best=None)
            ctx.close(sym, reason=reason)
            return

        # base chandelier trail
        stop_distance = p.vwatr_mult * vwatr
        if px <= best - stop_distance:
            put(best=None)
            ctx.close(
                sym,
                reason=(
                    f"[vwatr] trail {sym}: close {px:.2f} "
                    f"{stop_distance:.2f} ({stop_distance / px:.2%}) off best {best:.2f}"
                ),
            )
        return

    # ---- entry: rising expansion + structural breakout ----
    if not np.isfinite(base_now) or base_now <= 0:
        return

    # 3) baseline divergence gate
    if p.baseline_gate == "above" and not (vwatr > base_now):
        return
    if p.baseline_gate == "below" and not (vwatr < base_now):
        return

    # 4) macro regime gate (no-op unless macro_gate is set)
    if not gate_ok:
        return

    if n <= p.slope_bars:
        return
    vwatr_prev = float(vwatr_arr[n - 1 - p.slope_bars])
    if not np.isfinite(vwatr_prev):
        return
    if not (vwatr > vwatr_prev):
        return

    prior = close[n - 1 - p.breakout_look : n - 1]
    if len(prior) < p.breakout_look or not np.all(np.isfinite(prior)):
        return
    prior_high = float(np.max(prior))
    if not (px > prior_high):
        return

    stop_price = p.vwatr_mult * vwatr
    equity = ctx.current_equity()
    qty = risk_sized_qty(
        equity=equity, price=px, stop_dist=stop_price, risk_pct=p.risk_pct
    )
    if qty <= 0:
        return
    size = min(qty * px / equity, 1.0) if equity > 0 else 0.0
    if size <= 0:
        return

    reason = (
        f"[vwatr-div] entry {sym}: close {px:.2f} > {prior_high:.2f}, "
        f"vwatr {vwatr:.3f} ({vwatr / base_now:.2f}x base {base_now:.3f}), "
        f"rising vs {vwatr_prev:.3f} over {p.slope_bars}b, "
        f"risk {p.risk_pct:.1%} qty {qty:.2f} stop {stop_price:.2f}"
    )

    put(best=px)
    ctx.long(sym, size=size, size_mode="equity", reason=reason)


def _shift(values: np.ndarray, bars: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if bars < len(values):
        out[bars:] = values[: len(values) - bars]
    return out


def _as_series(index, values: np.ndarray):
    import pandas as pd

    return pd.Series(values, index=index)


def plot(ctx: StrategyContext, params: Params) -> PlotSpec:
    """Post-run chart spec. The cursor is at the terminal bar here, so every
    ``ctx.ta`` series is the full visible history (not a truncation).
    """
    sym = ctx.candle.symbol
    df = ctx.state.candles.get((sym, ctx.interval))
    if df is None or len(df) < params.vwatr_period + params.vwatr_base_win:
        return PlotSpec()

    close = ctx.ta.close(sym).to_array()
    vwatr_arr = ctx.ta.vwatr(sym, params.vwatr_period).to_array()
    baseline = ctx.ta.vwatr_baseline(sym, params.vwatr_period, params.vwatr_base_win)
    ratio = _safe_div(vwatr_arr, baseline.to_array())
    pct = _safe_div(vwatr_arr, close)
    slope = vwatr_arr - _shift(vwatr_arr, params.slope_bars)
    slope_prev = _shift(slope, params.slope_bars)

    stop_band = params.vwatr_mult * vwatr_arr
    upper = close + stop_band
    lower = close - stop_band

    look = params.breakout_look
    channel = np.full(close.shape, np.nan, dtype=np.float64)
    if len(close) > look:
        from numpy.lib.stride_tricks import sliding_window_view

        channel[look:] = sliding_window_view(close, look).max(axis=1)[:-1]

    markers: list[Marker] = []
    for trade in ctx.state.portfolio.trades:
        if trade.symbol != sym or trade.position != ActionType.long:
            continue
        markers.append(
            Marker(
                ts=ms(trade.entry_time),
                price=float(trade.entry_price),
                kind="pivot_low",
            )
        )

    return PlotSpec(
        overlays=(
            Overlay(
                sparse(_as_series(df.index, channel)), "breakout_high", "line", "price"
            ),
            Overlay(sparse(_as_series(df.index, upper)), "trail_hi", "line", "price"),
            Overlay(sparse(_as_series(df.index, lower)), "trail_lo", "line", "price"),
        ),
        panels=(
            Panel(
                sparse(_as_series(df.index, slope)),
                f"vwatr slope ({params.slope_bars}b)",
                (0.0,),
            ),
            Panel(
                sparse(_as_series(df.index, slope_prev)), "vwatr slope (prior)", (0.0,)
            ),
            Panel(
                sparse(_as_series(df.index, pct)), "vwatr / close (price-unit norm)", ()
            ),
            Panel(sparse(_as_series(df.index, ratio)), "vwatr / baseline", ()),
        ),
        markers=tuple(markers),
    )
