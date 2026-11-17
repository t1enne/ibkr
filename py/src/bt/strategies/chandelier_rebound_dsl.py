"""Chandelier rebound — an armed-reclaim mean-reversion book (long only).

Intent: a lead-in stretch (close below the slow MA while the fast MA is above it,
and the close far below its own z-mean) *arms* a symbol; the entry fires when the
close then reclaims the slow MA, betting the reclaim reverts rather than runs on.
Arm state lives in ``ctx.shared``, fresh per run window (no bleed across splits).

Mechanism, in order
-------------------
1. **Arm.** ``close < SMA(ma_slow)``, ``SMA(ma_fast) > SMA(ma_slow)`` and ``z <
   z_threshold``; EITHER fast/slow cross disarms — the arm means fast > slow.
2. **Trend gate.** ``trend_ma > 0`` admits only above ``SMA(trend_ma)``; ``0``
   disables it and reads no extra series. An unreadable trend admits nothing.
3. **Trigger.** ``crossover(close, SMA(ma_slow))`` while armed, in trade phase,
   flat, finite positive signal-bar ATR. One lot per symbol.
4. **Size.** ``size`` of live MTM equity, times ``1 + vol_size_gain`` when
   ``volume[-1] / SMA(volume, vol_ma) >= vol_thresh`` (baseline = the bars before
   the entry bar), capped at ``4 * size``. Unconfirmable volume means normal size.
5. **Regime floor.** ``vix_min`` admits only while the latest ``GATE_VIX`` close
   is at/above it. Fail CLOSED, and the refusal does NOT spend the arm.
6. **Exit — chandelier only.** Track the highest HIGH since entry; close when a
   completed bar's CLOSE reaches ``extreme - chandelier_atr * ATR``. That
   entry-time level is the only stop; ``trail_ratchet`` keeps it monotone.
7. **Validation.** A level not strictly below the signal bar's close, or on a
   non-finite / non-positive ATR or width, REFUSES the order (arm spent).

Measured and removed
--------------------
Each lost its measurement on the real book; none returns. Short leg: negative
standalone. ``bracket``/``decay`` exits (``sl_atr``/``tp_atr``/``exit_z``):
negative. MFE-tightened width: worse in every cell. ``vix_size_gain``: worse, and
rejected fills. ``vix_threshold``: degraded the book. ``rs_max``: kept worse
trades than it dropped. ``fund_qoq_min``: no separation. ``max_new_per_bar``:
worse than uncapped at every cap tried.

Invariants
----------
* ATR is the SIGNAL bar's; levels are fixed at signal time and the engine fills
  at the NEXT open, so a gap beyond a level is possible and NOT refused.
* ``ta.stdev`` is the **population** (ddof=0) deviation; pandas' ddof=1 default
  would shift every z-score silently.
* The arm is spent whether or not an order is emitted, except rule 5's refusal.
* **Chandelier lookahead.** Bar t widens the channel with its OWN high and tests
  its OWN close; no read goes beyond ``[-1]``. The exit fills at the next open.
* ``_TRAIL``/``_TRAIL_LEVEL`` are keyed by symbol, cleared together by
  ``_clear_trail_state`` at the only two places a lot can die: ``qty == 0``
  (including an ENGINE-ended lot) and right after a trail exit. The level store
  exists only when ``trail_ratchet`` is on (asserted at both sites).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.bt.strategies.dsl import StrategyContext, strategy
from src.bt.strategies.series import SeriesView
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "chandelier_rebound_dsl"

# Read-only macro series live in the feed under this prefix (codebase-wide
# convention, see `pullback_sx_dsl` / `vwatr_div_dsl`): they are data, never
# trade targets, so the symbol loop skips them.
_GATE_PREFIX = "GATE_"

# ctx.shared keys. ``_ARMED`` holds the symbols armed on an earlier bar:
# presence IS the flag (Pine's boolean). Re-arming is idempotent.
_ARMED = "armed"
_GATE_SYMBOL = "gate_symbol"
#: Chandelier trail state: ``sym -> highest high since entry``. Seeded with the
#: SIGNAL bar's high at emission time so the first fill bar continues the
#: channel instead of restarting it.
_TRAIL = "chandelier_trail"
#: ``trail_ratchet`` state: ``sym -> the last level EMITTED``, so the next bar
#: can only ratchet it tighter. Exists only when the knob is on; cleared with
#: ``_TRAIL`` whenever the lot goes flat.
_TRAIL_LEVEL = "chandelier_trail_level"

#: ``close_reason`` tag on a chandelier trail exit, so the exit mix is readable
#: in the report next to the engine's own ``sl``/``tp``/``end`` reasons.
_CHANDELIER_EXIT_REASON = "chandelier"

#: ``vix_min``'s OFF sentinel. The gate's value is a VIX LEVEL, so it is never
#: negative: ``-1.0`` cannot be a real threshold and is free to mean OFF, which
#: also means the GATE_ series is never read.
_VIX_MIN_OFF = -1.0
#: Lowest level that can be a real threshold. Anything between the OFF sentinel
#: and ``0.0`` is an authoring mistake, so it is ASSERTED against rather than
#: silently read as off or on.
_VIX_MIN_ENABLE = 0.0

#: ``trail_ratchet``'s OFF sentinel. A bool needs no RANGE, so what is policed
#: is its wiring: the level store exists exactly when the knob is ``True``
#: (asserted at both the resolution site and the reader), which is what keeps
#: the OFF path free of extra state.
_TRAIL_RATCHET_OFF = False

Side = Literal["long", "short"]


@dataclass(frozen=True)
class Params(StrategyParams):
    """Pine's five ``input.*`` values plus the execution/gate knobs."""

    ma_fast: int = 10
    ma_slow: int = 50
    atr_len: int = 14
    z_len: int = 20
    z_threshold: float = -1.0
    #: Fraction of live equity per lot. Pine's ``default_qty_value=10`` is the
    #: DEFAULT of this param, not a hardcoded truth.
    size: float = 0.1
    #: Volume confirmation: baseline window, counted in COMPLETED bars strictly
    #: before the entry bar (the entry bar is the numerator only).
    vol_ma: int = 20
    #: ``volume[-1] / SMA(volume, vol_ma)`` at/above this confirms the entry.
    vol_thresh: float = 1.5
    #: Confirmed size multiplier: ``size * (1 + vol_size_gain)``, capped at
    #: ``4 * size``. ``<= 0`` switches the feature OFF and the volume series is
    #: never read.
    vol_size_gain: float = 0.0
    #: Slow-MA trend filter: longs admitted only above it. ``0`` disables the
    #: gate and costs no extra indicator read.
    trend_ma: int = 200
    #: Trailing channel width in ATRs — the only exit this module has.
    chandelier_atr: float = 3.0
    #: Monotone channel. ``True`` keeps the emitted level from ever loosening
    #: when ATR expands: the level is ``max(prior, extreme - width)``. Default
    #: OFF, and the level store is then never created, so the OFF path adds no
    #: state at all.
    trail_ratchet: bool = _TRAIL_RATCHET_OFF
    #: High-volatility regime floor: an entry is admitted only while the latest
    #: ``GATE_VIX`` close is at/above this level. Default ``_VIX_MIN_OFF``
    #: (sentinel) = OFF, and the GATE_ series is then never read. Any value
    #: ``>= _VIX_MIN_ENABLE`` enables it.
    vix_min: float = _VIX_MIN_OFF
    allow_longs: bool = True


@dataclass(frozen=True)
class _Entry:
    """An entry candidate for this bar, plus what the exit needs."""

    sym: str
    side: Side
    close: float  # signal bar close — the reference every level anchors to
    atr: float  # signal bar ATR, same bar, same units as the exit level
    high: float  # signal bar high — the long chandelier channel's seed
    low: float  # signal bar low — part of the finite-input check on the level


@dataclass(frozen=True)
class _Trigger:
    """An armed symbol whose trigger fired on this bar.

    ``order`` is the candidate to emit, or ``None`` when a gate (phase, trend,
    open position, unusable ATR, regime floor) suppressed the order — Pine
    spends the arm either way.

    ``spend_arm`` is ``False`` only for the regime floor's refusal: that is an
    absence of data, not a verdict on the price premise, so it must not burn a
    signal the price side still owns.
    """

    sym: str
    order: _Entry | None
    spend_arm: bool = True


@strategy(stateful=True)
def on_candle(ctx: StrategyContext) -> None:
    """Arm/disarm, trail, and emit this bar's entries."""
    params: Params = ctx.params
    armed: set[str] = ctx.shared.setdefault(_ARMED, set())
    trail: dict[str, float] = ctx.shared.setdefault(_TRAIL, {})
    # The level store is resolved CONDITIONALLY: ``setdefault`` writes
    # ``ctx.shared``, so creating it unconditionally would put a key in the run
    # state for a knob that is off. ``None`` therefore means OFF everywhere
    # downstream, and nothing is ever written for it.
    trail_level: dict[str, float] | None = (
        ctx.shared.setdefault(_TRAIL_LEVEL, {}) if params.trail_ratchet else None
    )
    # The OFF path's own invariant: a store exists exactly when its knob is on,
    # which is what makes "no extra state when OFF" checkable rather than merely
    # claimed.
    assert (trail_level is not None) == params.trail_ratchet, (
        "per-symbol level store must exist exactly when trail_ratchet is on"
    )
    # One VIX read per bar, not per symbol: the floor is a market-wide state.
    vix_close = _bar_vix_close(ctx, params)
    # A NaN/non-finite floor level admits nothing this bar, which is the
    # fail-closed behaviour; `None` means the gate is off.
    vix_min_ok = _vix_min_admits(vix_close, params)
    long_allowed = params.allow_longs
    for sym in ctx.symbols:
        if sym.startswith(_GATE_PREFIX):
            continue
        trigger = _process_symbol(
            ctx, params, sym, armed, trail, trail_level, long_allowed, vix_min_ok
        )
        if trigger is None:
            continue
        # Arm spending, in Pine's order-independent sense: a trigger spends the
        # arm whether or not an order is emitted. The ONE exception is the
        # regime floor's refusal (``spend_arm``), which keeps it.
        if trigger.spend_arm:
            armed.discard(trigger.sym)
        if trigger.order is not None:
            _enter(ctx, params, trigger.order, trail)


def _bar_vix_close(ctx: StrategyContext, params: Params) -> float | None:
    """Latest ``GATE_VIX`` close for THIS bar, or ``None`` when the gate is off.

    ``None`` is returned BEFORE resolving the series when the floor is off, so
    the OFF path stays byte-identical and pays no read at all. An absent or NaN
    series yields NaN rather than ``None``: for the floor that is fail-CLOSED
    (see ``_vix_min_admits``).
    """
    if params.vix_min < _VIX_MIN_ENABLE:
        return None
    sym = _gate_symbol(ctx)
    if sym is None:
        return float("nan")
    closes = ctx.ta.close(sym)
    assert len(closes) > 0, "a resolved GATE_ series has no visible bars"
    return closes[-1]


def _gate_symbol(ctx: StrategyContext) -> str | None:
    """The feed's ``GATE_`` series name, or ``None`` while none is visible.

    ``ctx.symbols`` is derived from the candle store's *accumulated* rows, so a
    GATE_ series whose history starts inside the warmup window is absent for
    the feed's first bars. Resolution is therefore per-bar (cached once found):
    a config that asks for a VIX floor but carries no ``GATE_`` series at all
    fails CLOSED — zero trades, visible in the report — rather than trading
    ungated while claiming to be gated.
    """
    sym = ctx.shared.get(_GATE_SYMBOL)
    if sym is not None:
        return sym
    found = next((s for s in ctx.symbols if s.startswith(_GATE_PREFIX)), None)
    if found is not None:
        ctx.shared[_GATE_SYMBOL] = found
    return found


def _process_symbol(
    ctx: StrategyContext,
    params: Params,
    sym: str,
    armed: set[str],
    trail: dict[str, float],
    trail_level: dict[str, float] | None,
    long_allowed: bool,
    vix_min_ok: bool,
) -> _Trigger | None:
    """Advance one symbol on this bar; return its fired trigger or ``None``.

    Order matters (module docstring): disarm on either fast/slow cross, then
    re-arm on compression, then trigger. Each step sees the previous one's
    result, which is why this is not a single-pass rewrite.
    """
    closes = ctx.ta.close(sym)
    # A cross needs the previous bar's MAs, so ``ma_slow`` bars are not enough:
    # ``ma_slow + 1`` visible closes guarantee both ``[-1]`` and ``[-2]`` reads
    # land on real (non-NaN) SMA values. The z window reads ``z_len`` closes.
    if len(closes) < params.ma_slow + 1 or len(closes) < params.z_len:
        return None

    ma_fast = ctx.ta.sma(sym, params.ma_fast)
    ma_slow = ctx.ta.sma(sym, params.ma_slow)
    atr_cur = ctx.ta.atr(sym, params.atr_len)[-1]

    # 0. Chandelier trail runs BEFORE every MA guard: the exit is driven by the
    # trail and this bar's close alone, so an unreadable MA must never be able
    # to swallow an exit that the channel already triggered.
    _chandelier_exit(ctx, params, sym, closes[-1], atr_cur, trail, trail_level)
    close_cur = closes[-1]
    # SMA is NaN until its window fills; a NaN comparison is always False in
    # Pine too, so bail (fail closed) rather than compare against NaN.
    if not (np.isfinite(close_cur) and np.isfinite(ma_slow[-1])):
        return None
    if not (np.isfinite(ma_fast[-1]) and np.isfinite(ma_fast[-2])):
        return None
    if not np.isfinite(ma_slow[-2]):
        return None

    trend_ok = _trend_admits(ctx, params, sym, close_cur)
    if trend_ok is None:
        return None
    # 1 and 2: disarm, then re-arm.
    _update_arm(armed, params, sym, closes, ma_fast, ma_slow)

    # A non-positive or NaN ATR cannot produce a sane exit level; fail closed
    # rather than emit a level nobody can trust.
    can_enter = (
        ctx.phase == "trade"
        and ctx.position(sym) is None
        and np.isfinite(atr_cur)
        and atr_cur > 0.0
    )

    # 3. Trigger: the arm is consumed whether or not an order is emitted (the
    # caller spends it from this trigger record).
    if _cross_over(closes, ma_slow) and sym in armed:
        return _admitted_trigger(
            ctx,
            sym,
            close_cur,
            atr_cur,
            can_enter and long_allowed and trend_ok,
            vix_min_ok,
        )
    return None


def _trend_admits(
    ctx: StrategyContext, params: Params, sym: str, close_cur: float
) -> bool | None:
    """Slow-MA trend gate on this (the entry) bar.

    ``None`` means "cannot be judged on this bar" (history shorter than
    ``trend_ma``, or a NaN trend SMA) and the caller fails closed: no arm
    update, no entry — a regime that cannot be read admits nothing.
    """
    if params.trend_ma <= 0:
        return True
    closes = ctx.ta.close(sym)
    if len(closes) < params.trend_ma:
        return None
    trend_cur = ctx.ta.sma(sym, params.trend_ma)[-1]
    if not np.isfinite(trend_cur):
        return None
    return close_cur > trend_cur


def _update_arm(
    armed: set[str],
    params: Params,
    sym: str,
    closes: SeriesView,
    ma_fast: SeriesView,
    ma_slow: SeriesView,
) -> None:
    """Disarm on either fast/slow cross, then re-arm on compression.

    The arm condition asserts ``ma_fast > ma_slow``, so EITHER cross falsifies
    the premise the arm encodes — hence a symmetric reset. Order is Pine's: the
    disarm must be visible to the re-arm on the same bar, or the bar where a
    cross under immediately re-arms would be lost.
    """
    if _cross_over(ma_fast, ma_slow) or _cross_under(ma_fast, ma_slow):
        armed.discard(sym)
    close_cur = closes[-1]
    if close_cur < ma_slow[-1] and ma_fast[-1] > ma_slow[-1]:
        if _z_score(closes, params.z_len, close_cur) < params.z_threshold:
            armed.add(sym)


def _admitted_trigger(
    ctx: StrategyContext,
    sym: str,
    close_cur: float,
    atr_cur: float,
    price_gates_open: bool,
    vix_min_ok: bool,
) -> _Trigger:
    """Build this bar's trigger record, running the regime floor last.

    The floor is read ONLY where an order would otherwise have been emitted: a
    bar suppressed by the price-side gates spends its arm exactly as before, so
    enabling the gate cannot change the arm bookkeeping of a bar it never
    adjudicated. A bar the floor itself refuses KEEPS the arm
    (``spend_arm=False``).
    """
    if not price_gates_open:
        return _Trigger(sym, None)
    if not vix_min_ok:
        return _Trigger(sym, None, spend_arm=False)
    return _Trigger(sym, _signal_bar_entry(ctx, sym, close_cur, atr_cur))


def _vix_min_admits(vix_close: float | None, params: Params) -> bool:
    """High-volatility regime floor; ``True`` = admitted.

    ``True`` immediately when the gate is OFF, so the book is byte-identical to
    the un-gated behaviour. The level is resolved once per bar by
    ``_bar_vix_close`` and handed in.

    FAIL CLOSED: no visible GATE_ series or a NaN level admits nothing this bar.
    The caller turns that into a refusal that does NOT spend the arm.
    """
    assert params.vix_min == _VIX_MIN_OFF or (params.vix_min >= _VIX_MIN_ENABLE), (
        f"vix_min has no meaning below {_VIX_MIN_ENABLE} except the OFF sentinel"
    )
    # OFF sentinel: return before touching the level (which the caller has then
    # not even resolved).
    if params.vix_min < _VIX_MIN_ENABLE:
        return True
    assert vix_close is not None, "floor enabled but no bar-level VIX was resolved"
    if not np.isfinite(vix_close):
        return False
    return vix_close >= params.vix_min


def _signal_bar_entry(
    ctx: StrategyContext,
    sym: str,
    close_cur: float,
    atr_cur: float,
) -> _Entry:
    """Pack this bar's candidate record (the bar is the signal bar)."""
    bars = ctx.ohlcv(sym)
    return _Entry(sym, "long", close_cur, atr_cur, bars.high[-1], bars.low[-1])


def _chandelier_exit(
    ctx: StrategyContext,
    params: Params,
    sym: str,
    close_cur: float,
    atr_cur: float,
    trail: dict[str, float],
    trail_level: dict[str, float] | None,
) -> None:
    """Extend the trailing channel, then close the lot if this bar broke it.

    LOOKAHEAD INVARIANT (the reason this is a close-triggered exit and not a
    resting stop order): on bar t the channel is widened with bar t's OWN high,
    and the exit test reads bar t's OWN close. The extreme and the close are the
    same completed bar, and the channel is never widened with a bar the strategy
    has not finished seeing — every read is ``[-1]`` of a cursor-truncated
    ``SeriesView``, so an uncompleted or future bar is structurally
    unreachable. The exit itself is emitted through ``ctx.close`` and fills at
    the engine's next open: the engine's own convention, not a price fabricated
    here.

    ``qty == 0`` clears the symbol's trail. That is the only cleanup hook the
    strategy gets for a lot the ENGINE ended (stop-out, final-bar close), so it
    must run on every flat bar rather than only after a trail exit — otherwise
    a stale extreme would seed the next position on that symbol. The same hook
    clears the ratcheted level.

    THE RATCHET OFF IS STRUCTURAL, not merely arithmetic. With ``trail_ratchet``
    ``False`` there is no prior level to ratchet against, so ``trail_level``
    arrives as ``None``, is never created and is never read, and the emitted
    level is the plain ``extreme - chandelier_atr * ATR``.
    """
    assert params.chandelier_atr > 0.0, (
        f"chandelier_atr must be positive, got {params.chandelier_atr}"
    )
    # Paired with ``on_candle``'s assertion: the store exists exactly when the
    # knob that owns it is on. Guards against a caller wiring one without the
    # other, which would silently disable the feature it thinks it enabled.
    assert (trail_level is not None) == params.trail_ratchet, (
        "chandelier level store must exist exactly when trail_ratchet is on"
    )
    # One lot per symbol (no pyramiding) and the short leg is gone, so a held
    # lot is necessarily long. Assert rather than mirror: a negative quantity
    # (a seeded short book) must fail loud instead of being trailed as a long,
    # which would exit it at a level that means nothing for that position.
    qty = ctx.quantity(sym)
    if qty == 0.0:
        _clear_trail_state(trail, trail_level, sym)
        return
    assert qty > 0.0, "only long lots are trailed (the short leg was removed)"
    bars = ctx.ohlcv(sym)
    bar_high = bars.high[-1]
    prev = trail.get(sym)
    # ``prev is None`` on a bar that already holds the lot means the position
    # pre-dates this run (a seeded live book): seed from the oldest extreme we
    # can actually see instead of inventing an entry bar.
    extreme = bar_high if prev is None else max(prev, bar_high)
    trail[sym] = extreme
    if not (np.isfinite(extreme) and np.isfinite(atr_cur) and atr_cur > 0.0):
        return
    level = extreme - params.chandelier_atr * atr_cur
    if trail_level is not None:
        # The level is monotone. ``ATR`` expanding would otherwise push the
        # level AWAY from price mid-trade, re-opening risk the book has already
        # paid for; ``max`` with the prior level can only tighten.
        prior = trail_level.get(sym)
        if prior is not None:
            level = max(prior, level)
        trail_level[sym] = level
        # Positive AND negative space, before write and after read: the emitted
        # level never loosens against the prior one.
        stored = trail_level[sym]
        assert prior is None or stored >= prior, "ratcheted chandelier level loosened"
        assert np.isfinite(stored), "ratcheted chandelier level is not finite"
    if not np.isfinite(close_cur):
        return
    # Close-basis test on purpose: an intrabar touch of the channel is noise at
    # daily resolution, while a close through it is the trend actually turning.
    if close_cur <= level:
        _clear_trail_state(trail, trail_level, sym)
        ctx.close(sym, reason=_CHANDELIER_EXIT_REASON)


def _clear_trail_state(
    trail: dict[str, float],
    trail_level: dict[str, float] | None,
    sym: str,
) -> None:
    """Drop every piece of per-symbol chandelier state (the lot is gone).

    THE SINGLE OWNER OF THE CLEANUP DISCIPLINE. The two stores describe ONE
    position, so they must die together: a surviving extreme would restart the
    channel, and a surviving level would ratchet the next position against a
    level it never reached. Called from exactly the two places a lot can
    disappear — the ``qty == 0`` branch (which is also how a lot the ENGINE
    ended is noticed) and immediately after a trail exit is emitted. ``None``
    means the ratchet is OFF and its store was never created, so nothing is
    written for it.
    """
    trail.pop(sym, None)
    if trail_level is not None:
        trail_level.pop(sym, None)


def _enter(
    ctx: StrategyContext,
    params: Params,
    entry: _Entry,
    trail: dict[str, float],
) -> None:
    """Emit the order for one candidate, or refuse it."""
    assert entry.side == "long", "only the long leg is emitted"
    size = _entry_size(ctx, params, entry.sym)
    # The entry-time channel level is the ONLY stop: it is anchored to the
    # signal bar's high, which necessarily sits above the close the order is
    # priced off, so no separate stop bracket is kept alongside it and no target
    # is armed.
    level = _chandelier_entry_level(entry, params)
    if level is None:
        return
    # Seed the trail with the SIGNAL bar's high — the same bar the stop is
    # anchored to — so the first bar that actually holds the lot continues the
    # channel instead of restarting it one bar late.
    trail[entry.sym] = entry.high
    ctx.long(
        entry.sym,
        size=size,
        size_mode="equity",
        sl=level,
        tp=None,
        reason="[rebound] armed + close reclaimed slow MA",
    )


def _entry_size(ctx: StrategyContext, params: Params, sym: str) -> float:
    """Order size for one candidate: ``size`` times the volume confirmation.

    An unconfirmed entry keeps plain ``size``: this is a sizing decision, so its
    failure mode is "trade the normal size", never "skip the trade".
    """
    assert params.vol_ma > 0, f"vol_ma must be positive, got {params.vol_ma}"
    assert params.vol_size_gain >= 0.0, (
        f"vol_size_gain must be non-negative, got {params.vol_size_gain}"
    )
    base = params.size
    # OFF: return before touching the volume series so sizing is byte-identical
    # to the un-confirmed behaviour.
    if params.vol_size_gain <= 0.0:
        return base
    mult = _vol_confirm_mult(ctx, params, sym)
    # Positive AND negative space of the outcome: the confirmation may never
    # shrink a lot, and an unconfirmed one leaves it exactly at the base size.
    assert mult >= 1.0, "a confirmation multiplier must never be below 1"
    if mult <= 1.0:
        return base
    sized = min(base * mult, 4.0 * base)
    # Positive AND negative space of the cap: a confirmed lot is strictly larger
    # than an unconfirmed one, yet never above the 4x ceiling.
    assert sized > base, "confirmed size did not exceed the base size"
    assert sized <= 4.0 * base, "confirmed size exceeded its 4x cap"
    return sized


def _vol_confirm_mult(ctx: StrategyContext, params: Params, sym: str) -> float:
    """Volume-confirmation multiplier; ``1.0`` = unconfirmed.

    ``vol_ratio = volume[-1] / SMA(volume, vol_ma)``; at/above ``vol_thresh``
    the multiplier is ``1 + vol_size_gain``. Returns ``1.0`` immediately when
    the feature is OFF, so the volume series is never read. A history too short
    for the baseline, a zero/NaN baseline, or a NaN ratio is "no confirmation",
    never "no trade".

    NO-LOOKAHEAD INVARIANT: the baseline walks ``range(-vol_ma - 1, -1)``, the
    ``vol_ma`` bars STRICTLY BEFORE this one, so the entry bar's own volume is
    the numerator and can never dilute its own baseline. Every read is ``[-1]``
    or lower of a cursor-truncated ``SeriesView``.
    """
    if params.vol_size_gain <= 0.0:
        return 1.0
    volume = ctx.ohlcv(sym).volume
    if len(volume) < params.vol_ma + 1:
        return 1.0
    baseline = float(
        np.mean(
            np.fromiter(
                (volume[i] for i in range(-params.vol_ma - 1, -1)),
                dtype=np.float64,
                count=params.vol_ma,
            )
        )
    )
    if not np.isfinite(baseline) or baseline <= 0.0:
        return 1.0
    ratio = float(volume[-1]) / baseline
    if not np.isfinite(ratio) or ratio < params.vol_thresh:
        return 1.0
    return 1.0 + params.vol_size_gain


def _chandelier_entry_level(entry: _Entry, params: Params) -> float | None:
    """Entry-time chandelier level for a candidate, or ``None`` to REFUSE it.

    ``high - k*ATR`` on the signal bar (the same bar the order anchors to).
    Refusal, never clamping, covers one class of defect: levels that cannot be
    trusted at all (non-finite input, a non-positive ATR or multiplier) and a
    level that does not sit below the signal bar's close — a bar whose high is
    more than ``k`` ATRs above its close would otherwise emit a "stop" already
    through the entry.
    """
    assert params.chandelier_atr > 0.0, "negative chandelier width"
    ref, atr = entry.close, entry.atr
    if not all(np.isfinite(v) for v in (ref, atr, entry.high, entry.low)):
        return None
    if atr <= 0.0 or params.chandelier_atr <= 0.0:
        return None
    level = entry.high - params.chandelier_atr * atr
    return level if np.isfinite(level) and level < ref else None


def _z_score(closes: SeriesView, z_len: int, close_cur: float) -> float:
    """Pine ``(close - ta.sma(close, n)) / ta.stdev(close, n)`` for this bar.

    ``ta.stdev`` is the population (ddof=0) deviation — NOT pandas' ddof=1
    default. NaN when the window has zero deviation (Pine yields NaN there too,
    and every comparison against NaN is False).
    """
    window = np.fromiter(
        (closes[i] for i in range(-z_len, 0)), dtype=np.float64, count=z_len
    )
    std = float(np.std(window, ddof=0))
    if not np.isfinite(std) or std <= 0.0:
        return float("nan")
    return (close_cur - float(np.mean(window))) / std


def _cross_over(a: SeriesView, b: SeriesView) -> bool:
    """Pine ``ta.crossover(a, b)``: ``a[-2] <= b[-2] and a[-1] > b[-1]``.

    Local rather than ``ctx.cross_over`` because the NaN guard is explicit here:
    Pine's crossover is NaN-strict (both operands finite on both bars), while
    ``ctx.cross_over`` would return False silently for the same reason — this
    keeps the intent visible at the call site.
    """
    a_prev, a_cur = a[-2], a[-1]
    b_prev, b_cur = b[-2], b[-1]
    if not all(np.isfinite(v) for v in (a_prev, a_cur, b_prev, b_cur)):
        return False
    return a_prev <= b_prev and a_cur > b_cur


def _cross_under(a: SeriesView, b: SeriesView) -> bool:
    """Pine ``ta.crossunder(a, b)``: ``a[-2] >= b[-2] and a[-1] < b[-1]``.

    The mirror of ``_cross_over``, same NaN-strict guard (see there for why this
    is local rather than ``ctx.cross_under``).
    """
    a_prev, a_cur = a[-2], a[-1]
    b_prev, b_cur = b[-2], b[-1]
    if not all(np.isfinite(v) for v in (a_prev, a_cur, b_prev, b_cur)):
        return False
    return a_prev >= b_prev and a_cur < b_cur
