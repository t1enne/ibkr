"""Pivot-pair MFI divergence (long + short), cash-funded entries.

Unlike ``mfi_divergence_dsl``, which compares the ``min`` of two fixed halves
of a window (no turning points, no structure -- it fires on monotone legs), this
strategy only trades a **divergence wave**:

    swing low L1 -> intervening swing high -> swing low L2

with ``L2.price < L1.price`` and ``L2.mfi > L1.mfi + gap`` (bullish), i.e. a
lower low in price that the oscillator refuses to confirm. Bearish is the
mirror on swing highs. Every geometric gate -- gap bands, price-extension band,
intervening depth, oscillator tail -- lives in
:mod:`src.bt.strategies.divergence_geometry` with its own defaults; this module
exposes only the four shape knobs it actually varies (``pivot_bars``,
``lookback``, ``min_gap_bars``, ``min_osc_gap``) and never re-derives a gate.

Capital deployment has no exposure cap. The bar's candidates are ranked by MFI
divergence magnitude (P2 osc minus P1 osc, in points) and at most
``rank_topk`` per side are entered, strongest first -- ranking stops a
market-wide dip from opening the whole universe at once. The lot fraction is
then derived from cash (:func:`_affordable`): a bar is traded only when the
cash left can fund its **whole ranked cohort**, so the portfolio layer never
has to scale or reject a fill and no lot is silently shrunk. With no gross cap
the book still cannot exceed cash -- this engine funds shorts out of cash too,
so gross |notional| <= cash <= equity by construction.

Sizing is participation-modulated, with no knob for it: the lot is
``risk_pct`` scaled by a SHRINK-ONLY multiplier in ``[_AMP_FLOOR, 1]`` derived
from the signal bar's volume percentile within its trailing ``_VOL_WIN``
(:func:`_size_amp`). Dry volume on the reversal is the absorption the thesis
wants; heavy volume is distribution into the bounce.

No VIX/regime gate. Four reads were built and REMOVED after measurement: a
VIX level-percentile band (loses Sharpe monotonically in both directions), VIX
dynamics (calm / healing / spike / roll-over -- only ``calm`` helped, and only
in-sample), a VIX term-structure sizing leg (VIX3M contango percentile, Sharpe
1.06 -> 0.92), and every short-admission gate tried on the long-only survivor
(VIX rank band, benchmark bear tape, own online-HMM high-vol state) -- the short
leg is negative standalone under all of them and adds only sub-standard-error
Sharpe to the combined book. See ``strats/README.md``; do not re-add any of them
without new evidence, and note that the short side would need a different EXIT
rule to be worth re-probing, not a different admission gate.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from src.bt.strategies.divergence_geometry import (
    DivergenceParams,
    PivotScanner,
    Side,
    evaluate_from_pivots,
)
from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.types import StrategyParams

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
class _Bar:
    """One symbol's cursor-truncated bar, shared by the decision helpers."""

    sym: str
    n: int  # bars visible (bar index of "now")
    close: float
    atr: float
    prices: np.ndarray  # close series, last element = now
    mfi: np.ndarray  # MFI series, same length
    volumes: np.ndarray  # volume series, same length


@dataclass(frozen=True)
class Params(StrategyParams):
    # -- MFI / divergence geometry (the only shape knobs this module exposes) --
    mfi_period: int = 14
    pivot_bars: int = 3  # fractal half-width for swings
    lookback: int = 90  # bars scanned for pivots
    min_gap_bars: int = 5  # min separation between the two swings
    min_osc_gap: float = 5.0  # absolute MFI improvement required
    # -- side admission --
    allow_longs: bool = True
    allow_shorts: bool = True
    # -- risk / exits --
    atr_period: int = 14
    trail_atr_mult: float = 1.5  # chandelier trail off the running extreme
    entry_stop_atr: float = 3.0  # structural stop, ATRs beyond entry
    risk_pct: float = 0.05  # lot notional as a fraction of live equity
    warmup_bars: int = 60
    cooldown_bars: int = 0  # flat bars after an exit before re-entry
    # -- capital deployment --
    # >0 = at most this many candidates PER SIDE per bar, strongest MFI
    # divergence first, and the cohort they form must be fundable from cash
    # (0 = no cohort cap: enter the strongest candidates cash still covers).
    rank_topk: int = 0


@dataclass(frozen=True)
class _Entry:
    """A ranked entry candidate: the side, its divergence magnitude, and the
    lot reference values ``_enter`` needs."""

    sym: str
    side: Side
    score: float  # MFI gap in points between the two pivots (higher = stronger)
    close: float
    atr: float
    pivot_price: float
    pivot_osc: float
    vol_pct: float  # percentile of this bar's volume in the trailing window


@strategy(stateful=True)
def on_candle(ctx: StrategyContext):
    # Warmup bars fill strategy state only — no fill can happen there, so the
    # DSL raises on any entry emitted during them.
    if ctx.phase == "warmup":
        return
    p: Params = ctx.params
    # Every symbol is scanned and managed on this bar; only the ENTRY emission
    # is deferred to the end of the cross-section, so ranking and the cash
    # affordability test see the whole cohort before the first lot is emitted.
    buckets: dict[Side, list[_Entry]] = {"long": [], "short": []}
    for sym in ctx.symbols:
        entry = _process_symbol(ctx, p, sym)
        if entry is not None:
            buckets[entry.side].append(entry)
    cohort = buckets["long"] + buckets["short"]
    sides = int(p.allow_longs) + int(p.allow_shorts)
    pad = max((e.atr / e.close for e in cohort if e.close > 0), default=0.0)
    allowance = _affordable(
        cash=float(ctx.state.portfolio.cash),
        equity=ctx.current_equity(),
        risk_pct=p.risk_pct,
        sides=sides,
        pad=pad,
    )
    if allowance <= 0:
        return
    topk = p.rank_topk if p.rank_topk > 0 else allowance
    if allowance < topk:
        # The whole ranked cohort must be fundable, not just its first lots:
        # otherwise the tail of a burst gets diluted or refused by the engine.
        return
    for entry in _rank(buckets["long"], topk) + _rank(buckets["short"], topk):
        _enter(ctx, p, entry, p.risk_pct * _size_amp(entry.vol_pct))


#: Bars of history the participation percentile is measured over (structural,
#: ~6 months of sessions). Long enough to describe the name's own recent volume
#: climate rather than one event; a 20-bar window measured strictly worse.
_VOL_WIN: int = 120
#: Floor of the size modulator: a fully "wet" bar still trades at this fraction
#: of ``risk_pct``, so a mis-signed reading costs money rather than closing the
#: book.
_AMP_FLOOR: float = 0.25


def _pct_rank(values: np.ndarray) -> float:
    """Fraction of the window strictly below its last element (0..1).

    ``0.5`` for an empty/degenerate window so a missing reading is neutral.
    """
    if values.size < 2:
        return 0.5
    last = float(values[-1])
    prior = values[:-1]
    prior = prior[np.isfinite(prior)]
    if prior.size == 0 or not np.isfinite(last):
        return 0.5
    return float(np.count_nonzero(prior < last)) / float(prior.size)


def _vol_pct(volumes: np.ndarray) -> float:
    """Percentile of the signal bar's volume within the trailing ``_VOL_WIN``."""
    if volumes.size < 2:
        return 0.5
    return _pct_rank(volumes[-_VOL_WIN:])


def _size_amp(vol_pct: float) -> float:
    """Shrink-only size multiplier in ``[_AMP_FLOOR, 1]`` from participation.

    Thin volume on the reversal bar is the absorption the thesis actually wants
    (sellers exhausted, no distribution into the bounce), so a dry bar is sized
    at the full ``risk_pct`` and a heavy-volume bar down to ``_AMP_FLOOR``.
    Shrink-only by construction: it modulates exposure, it never re-levers, so
    the cash-affordability path and the edge-per-unit-risk reading stay honest.
    """
    dry = 1.0 - max(0.0, min(1.0, vol_pct))
    return _AMP_FLOOR + (1.0 - _AMP_FLOOR) * dry


def _rank(entries: list[_Entry], topk: int) -> list[_Entry]:
    """Strongest MFI divergence first, deterministic symbol tie-break."""
    ranked = sorted(entries, key=lambda e: (-e.score, e.sym))
    return ranked[:topk] if topk > 0 else ranked


def _affordable(
    *, cash: float, equity: float, risk_pct: float, sides: int, pad: float
) -> int:
    """How many full-size lots this bar can fund, PER ADMITTED SIDE.

    A lot is ``risk_pct * equity`` of notional and fills at the NEXT open, not at
    the close it was sized on, so each is padded by the cohort's largest one-bar
    move (``pad`` = max ATR/close -- a data-derived gap reserve, not a knob).
    Cash is split evenly across ``sides`` so a long-rich bar cannot starve the
    short book. Entering only what this returns keeps the bar's whole cohort
    inside cash by construction: the engine's shared cohort scale stays at 1 and
    no fill is scaled or rejected.
    """
    if equity <= 0 or cash <= 0 or risk_pct <= 0 or sides <= 0:
        return 0
    lot = risk_pct * equity * (1.0 + max(pad, 0.0))
    return int(cash / (sides * lot))


def _geometry(p: Params) -> DivergenceParams:
    """The strategy-facing subset; every other gate keeps its module default."""
    return DivergenceParams(
        pivot_bars=p.pivot_bars,
        lookback=p.lookback,
        min_gap_bars=p.min_gap_bars,
        min_osc_gap=p.min_osc_gap,
    )


def _process_symbol(ctx: StrategyContext, p: Params, sym: str) -> _Entry | None:
    """Advance one symbol on this bar: manage an open lot, else return its entry
    candidate (``None`` when flat with no trigger)."""
    holder: dict[str, _State] = ctx.shared.setdefault(_STATE_KEY, {})
    bar = _prepare(ctx, p, sym, holder)
    if bar is None:
        return None
    state = holder[sym]
    qty = ctx.quantity(sym)
    if qty != 0:
        _save(holder, sym, state, was_in_position=True)
        _trail(ctx, p, holder, state, bar, side="long" if qty > 0 else "short")
        return None
    if _exit_blocked(holder, sym, state, p, bar.n):
        return None
    return _candidate(p, holder[sym], bar)


def _prepare(
    ctx: StrategyContext, p: Params, sym: str, holder: dict[str, _State]
) -> _Bar | None:
    """Build this symbol's :class:`_Bar`, or ``None`` while it is still warming.

    Pivot scanners are the expensive-once, cheap-per-bar cache. They must be
    refreshed BEFORE ``_save`` closes over ``state``, else every write would
    rebuild them from scratch and undo the optimisation. Feeding them on every
    bar (even while in a position) keeps them warm across an exit.
    """
    state = holder.get(sym) or _State()
    holder[sym] = state
    o = ctx.ohlcv(sym)
    if o is None or len(o.close) <= p.warmup_bars:
        return None
    n = len(o.close)
    if n == 0:
        return None
    mfi = ctx.ta.mfi(sym, period=p.mfi_period).to_array()
    if mfi.size != n:
        return None
    prices = o.close.to_array()
    holder[sym] = _scanners(state, p, prices, mfi)
    atr = ctx.ta.atr(sym, period=p.atr_period).last()
    return _Bar(
        sym=sym,
        n=n,
        close=float(prices[-1]),
        atr=atr if np.isfinite(atr) else 0.0,
        prices=prices,
        mfi=mfi,
        volumes=o.volume.to_array(),
    )


def _save(holder: dict[str, _State], sym: str, state: _State, **kw: object) -> None:
    """Write a derived ``_State`` back to the per-run book."""
    holder[sym] = dataclasses.replace(state, **kw)


def _exit_blocked(
    holder: dict[str, _State], sym: str, state: _State, p: Params, n: int
) -> bool:
    """True while an exit cooldown still blocks re-entry on this symbol.

    The bar that discovers the flat book stamps ``last_exit_bar`` (the trail
    close or the stop fired on the previous bar), so the cooldown starts here.
    """
    last = n if state.was_in_position else state.last_exit_bar
    if state.was_in_position:
        _save(holder, sym, state, was_in_position=False, last_exit_bar=n)
    if p.cooldown_bars <= 0 or last < 0:
        return False
    return n - last < p.cooldown_bars


def _candidate(p: Params, state: _State, bar: _Bar) -> _Entry | None:
    """Evaluate the pivot pair on this bar for whichever sides are admitted."""
    assert state.lows is not None and state.highs is not None
    lows, highs = state.lows.pivots, state.highs.pivots
    geo = _geometry(p)
    if p.allow_longs:
        new = evaluate_from_pivots(
            lows, highs, bar.prices, bar.mfi, "long", geo, atr=bar.atr
        )
        if new is not None:
            return _Entry(
                bar.sym,
                "long",
                new.osc - lows[-2].osc,
                bar.close,
                bar.atr,
                new.price,
                new.osc,
                _vol_pct(bar.volumes),
            )
    if p.allow_shorts:
        new = evaluate_from_pivots(
            highs, lows, bar.prices, bar.mfi, "short", geo, atr=bar.atr
        )
        if new is not None:
            return _Entry(
                bar.sym,
                "short",
                highs[-2].osc - new.osc,
                bar.close,
                bar.atr,
                new.price,
                new.osc,
                _vol_pct(bar.volumes),
            )
    return None


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


def _trail(
    ctx: StrategyContext,
    p: Params,
    holder: dict[str, _State],
    state: _State,
    bar: _Bar,
    *,
    side: Side,
) -> None:
    """Ratchet the chandelier trail; close the lot when it trips."""
    long = side == "long"
    best = state.best_long if long else state.best_short
    key = "best_long" if long else "best_short"
    if best is None or not (best > 0):
        _save(holder, bar.sym, state, **{key: bar.close})
        return
    improved = (bar.close > best + 1e-9) if long else (bar.close < best - 1e-9)
    if improved:
        _save(holder, bar.sym, state, **{key: bar.close})
        return
    if bar.atr <= 0:
        return
    hit = (
        (bar.close <= best - p.trail_atr_mult * bar.atr)
        if long
        else (bar.close >= best + p.trail_atr_mult * bar.atr)
    )
    if not hit:
        return
    _save(holder, bar.sym, state, **{key: None})
    ctx.close(
        bar.sym,
        reason=(
            f"[pivotdiv] trail {side} {bar.sym}: close {bar.close:.2f} "
            f"{p.trail_atr_mult:.1f}ATR off best {best:.2f}"
        ),
    )


def _enter(ctx: StrategyContext, p: Params, entry: _Entry, size: float) -> None:
    """Emit one candidate at the bar's cash-derived lot fraction."""
    close, atr = entry.close, entry.atr
    if not (atr > 0 and close > 0 and 0 < size <= 1):
        return
    stop_dist = p.entry_stop_atr * atr
    reason = (
        f"[pivotdiv] {entry.side} {entry.sym}: MFI swing div at "
        f"{entry.pivot_price:.2f} (mfi {entry.pivot_osc:.1f}); enter {close:.2f}"
    )
    if entry.side == "long":
        ctx.long(
            entry.sym,
            size=size,
            size_mode="equity",
            sl=close - stop_dist,
            reason=reason,
        )
    else:
        ctx.short(
            entry.sym,
            size=size,
            size_mode="equity",
            sl=close + stop_dist,
            reason=reason,
        )
