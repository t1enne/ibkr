"""5-Candle Bias + Displacement Confirmation (DSL).

Encodes the spec as a state-machine on closed candles:

  * Bias     : previous 5 closed candles (relative offsets -3..-7 from a
               candidate confirmation at offset -1). Long bias iff >= threshold
               of the 5 are bullish AND close[-3] > close[-7] (net > 0).
               Short mirrored.
  * Pullback : the candle immediately after the bias window (-2), contra the
               bias direction, with body >= ``pullback_body_mult`` x the median
               body of the 5 bias candles. Strictly one candle.
  * Confirm  : the candle immediately after the pullback (-1, the current,
               already closed bar), in the bias direction. Body >=
               ``confirm_body_mult`` x pullback body, and close passes through
               the pullback open. Optional ATR volatility band on confirm body.
  * Entry    : single-shot at the confirmation bar (no chasing). The engine
               fills at NEXT open => one-bar execution lag is the realised
               slippage.
  * Stop     : long = pullback low -``stop_atr_mult``*ATR; short = pullback
               high +``stop_atr_mult``*ATR. 1R = |entry - stop|.
  * TP       : fixed ``tp_r`` x R bracket from entry.
  * Risk     : qty = cash*``risk_pct`` / stop_distance. When ``dyn_sizing`` is
               on, high-confidence setups (full bias-window alignment and/or a
               decisive confirmation) upsize that ``risk_pct`` via
               ``dyn_risk_mult`` so stronger setups get more capital. One live
               position per symbol; ``cooldown_bars`` blocks immediate reload of
               a symbol. ``max_positions`` = global open-position cap (0 =
               unlimited).

All values read are from fully-closed bars (SeriesView is cursor-truncated),
so there is no lookahead. SL/TP/stops are set from the confirmation-close
reference and filled next open; realised R therefore shifts slightly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bt.strategies.dsl import strategy, StrategyContext, OhlcvView
from src.bt.strategies.types import StrategyParams
from src.bt.strategies.series import SeriesView
from src.bt.size.pure import risk_sized_qty
from src.bt.state.types import ActionType
from src.indicators.adaptive_entropy.online import OnlineAdaptiveEntropy
from src.indicators.adaptive_entropy.types import AdaptiveEntropyConfig

STRATEGY_TYPE = "five_candle_bias_displacement"

_REGIME_KEY = "spy_ae_regime"  # ctx.shared[_REGIME_KEY] -> {'model','fed','result'}


@dataclass(frozen=True)
class Params(StrategyParams):
    lookback: int = 5  # bias window length
    bias_threshold: int = 4  # >= this many aligned in window
    pullback_body_mult: float = 0.5  # pullback body vs median bias body
    confirm_body_mult: float = 2.0  # confirmation body vs pullback body
    atr_period: int = 14
    stop_atr_mult: float = 0.1
    atr_filter_on: bool = False  # optional vol filter on confirm body
    atr_filter_lo: float = 0.5
    atr_filter_hi: float = 2.0
    tp_r: float = 2.0
    risk_pct: float = 0.005
    max_positions: int = 12
    warmup_bars: int = 60
    cooldown_bars: int = 3
    side: str = "both"  # "long" | "short" | "both"
    # -- regime gate (default off; keeps ungated runs byte-identical) -----
    regime_gate_mode: str = "none"  # "none" | "200sma" | "ae"
    gate_symbol: str = "SPY"  # observer whose series gates every entry
    gate_sma: int = 200  # window for the 200sma gate (daily closes)
    ae_lookback: int = 40  # AE entropy lookback (matches pass config)
    ae_num_bins: int = 10  # AE log-return histogram bins (pass config)
    regime_trend_min: int = 1  # require AE trend >= 1 (bull) to enter
    regime_min_strength: float = 0.0  # 0 disables the AE strength check
    # -- exit policy ---------------------------------------------------------
    # "fixed" = the legacy fixed ``tp_r`` R-multiple take-profit bracket
    # (engine-enforced, byte-identical to the pre-trail baseline).
    # "trail" = no fixed TP; the initial protective SL stays armed and a
    # chandelier trail (running-high/free extreme minus ``trail_atr_mult`` x
    # ATR) ratchets UP once the trade is ``activate_profit_R`` x R in profit.
    # Winners ride until the trail breaks; losers only ever exit at the SL.
    exit_mode: str = "fixed"  # "fixed" | "trail"
    trail_atr_mult: float = 3.0  # chandelier distance (x ATR) from extreme
    activate_profit_R: float = 1.0  # require this much (x R) before trailing
    # -- dynamic confidence sizing (default off; flat ``risk_pct`` fallback) -
    # Upsizes the per-trade risk on HIGHER-CONFIDENCE setups so more capital
    # flows to the most decisive entries (the ones that historically ride the
    # trail far). ``risk_pct`` stays the BASE risk for a neutral pass; strong
    # setups scale it up. Confidence = two prior-decided, monotone signals:
    #   * full_alignment : the entire ``lookback`` bias window aligned (max
    #     conviction runaway drift; not just meeting ``bias_threshold``).
    #   * decisive_confirm : confirmation body >= ``decisive_confirm_mult`` x
    #     the pullback body (require a decisive flag-day move, not the bare
    #     ``confirm_body_mult``>= floor pass).
    # True signals compose multiplicatively; the net multiplier is clamped to
    # [min_dyn_risk_mult, max_dyn_risk_mult]. Defaults (off) keep every run
    # byte-identical to the flat-``risk_pct`` baseline.
    dyn_sizing: bool = False  # True => boost risk on high-confidence setup
    full_alignment_boost: float = 1.5  # mult when all ``lookback`` aligned
    decisive_confirm_floor: float = 3.0  # confirm mult (x pb body) deemed decisive
    decisive_confirm_boost: float = 1.5  # mult when confirm >= that floor
    min_dyn_risk_mult: float = 1.0  # never under-size a pass (keep equity drift)
    max_dyn_risk_mult: float = 3.0  # cap single-trade oversize (concentration guard)


# --------------------------------------------------------------------------
# regime gates (GATE_A bare 200-SMA, GATE_B adaptive-entropy on SPY)
# --------------------------------------------------------------------------


def sma_gate_ok(ctx: StrategyContext, sym: str, period: int) -> bool:
    """GATE_A: long entries only while ``sym`` close > its ``period``-SMA.

    Pure function of closed data via ``ctx.ohlcv(sym)`` + ``ctx.ta.sma`` (both
    cursor-truncated, no lookahead). Requires at least ``period`` closes and a
    defined SMA value. Defaults to *false* (stay out) when the indicator is
    not yet warm so a gate never opens on a bare observation.
    """
    o = ctx.ohlcv(sym)
    if len(o.close) < period + 1:
        return False
    sma = ctx.ta.sma(sym, period)
    if sma is None or len(sma) == 0:
        return False
    last = float(sma[-1])
    if not np.isfinite(last):
        return False
    return float(o.close[-1]) > last


def _regime_model(ctx: StrategyContext) -> OnlineAdaptiveEntropy:
    """Return the per-run online AE model, minted once and fed per bar."""
    shared = ctx.shared
    holder = shared.setdefault(_REGIME_KEY, {})
    model = holder.get("model")
    if model is None:
        cfg = AdaptiveEntropyConfig(
            lookback=ctx.params.ae_lookback,
            num_bins=ctx.params.ae_num_bins,
        )
        model = OnlineAdaptiveEntropy(cfg)
        holder["model"] = model
    return model


def regime_allows_entry(ctx: StrategyContext) -> bool:
    """GATE_B: gate on the adaptive-entropy trend of ``gate_symbol`` (SPY).

    Ported verbatim from the proven momentum_compression_breakout AE gate
    (its SPY gate is load-bearing per strats/README). Feeds each *new* closed
    bar of the observer symbol into the online AE model (chronological,
    gap-safe) and allows entry only when the current AE trend is at least
    ``regime_trend_min`` (>=1 = bull). No AE history yet / trend sub-threshold
    => stay out.
    """
    params: Params = ctx.params
    o = ctx.ohlcv(params.gate_symbol)
    n_total: int = len(o.close)
    if n_total < params.ae_lookback + 1:
        return False
    model = _regime_model(ctx)
    holder = ctx.shared[_REGIME_KEY]
    fed: int = holder.get("fed", 0)
    if n_total > fed:
        close_arr = o.close.to_array()
        high_arr = o.high.to_array()
        low_arr = o.low.to_array()
        for i in range(fed, n_total):
            holder["result"] = model.observe(close_arr[i], high_arr[i], low_arr[i])
        holder["fed"] = n_total
    res = holder.get("result")
    if res is None:
        return False
    if res.trend < params.regime_trend_min:
        return False
    if (
        params.regime_min_strength > 0.0
        and res.trend_strength < params.regime_min_strength
    ):
        return False
    return True


def regime_gate_ok(ctx: StrategyContext) -> bool:
    """Dispatch the configured gate; "none" always allows."""
    mode = getattr(ctx.params, "regime_gate_mode", "none")
    if mode == "none":
        return True
    gsym = getattr(ctx.params, "gate_symbol", "SPY")
    if mode == "200sma":
        return sma_gate_ok(ctx, gsym, getattr(ctx.params, "gate_sma", 200))
    if mode == "ae":
        return regime_allows_entry(ctx)
    # unknown mode: be conservative and block (never silently wide-open)
    return False


# --------------------------------------------------------------------------
# primitive reads (relative offsets on cursor-truncated series)
# --------------------------------------------------------------------------


def _open(o: OhlcvView, i: int) -> float:
    return float(o.open[i])


def _high(o: OhlcvView, i: int) -> float:
    return float(o.high[i])


def _low(o: OhlcvView, i: int) -> float:
    return float(o.low[i])


def _close(o: OhlcvView, i: int) -> float:
    return float(o.close[i])


def _body(o: OhlcvView, i: int) -> float:
    return _close(o, i) - _open(o, i)


def atr_now(ctx: StrategyContext, sym: str, period: int) -> float:
    v = ctx.ta.atr(sym, period)
    return float(v[-1]) if isinstance(v, SeriesView) and len(v) > 0 else float("nan")


def dyn_risk_mult(
    params: Params, aligned: int, cf_body: float, pb_body: float
) -> float:
    """Multiplier on base ``risk_pct`` for a setup's confidence level.

    Confidence is decided *up front* (params), not fit to outcomes: two
    monotone, indicator-free signals of setup decisiveness compose
    multiplicatively, floored at 1.0 so a neutral pass keeps the flat baseline
    risk (preserving the strategy's equity-curve drift) and a high-confidence
    pass is the only thing that upsizes. ``max_dyn_risk_mult`` caps oversize as
    a concentration guard (cash already binds concurrency long before the cap).

    Signal A - full alignment: ``aligned == params.lookback`` (the entire bias
    window agrees with the displace) -> strongest runaway-drift conviction.

    Signal B - decisive confirmation: ``abs(cf_body)`` >= ``params.decisive_confirm_floor``
    x ``abs(pb_body)``, i.e. a flag-day one-candle conviction well past the bare
    ``confirm_body_mult`` pass threshold, not an every-setup marginal print.
    """
    if not getattr(params, "dyn_sizing", False):
        return 1.0
    mult = 1.0
    if params.full_alignment_boost > 1.0 and aligned >= params.lookback:
        mult *= params.full_alignment_boost
    pb = abs(pb_body)
    if (
        params.decisive_confirm_boost > 1.0
        and pb > 0
        and params.decisive_confirm_floor > 0
        and abs(cf_body) >= params.decisive_confirm_floor * pb
    ):
        mult *= params.decisive_confirm_boost
    lo = params.min_dyn_risk_mult
    hi = params.max_dyn_risk_mult
    return max(lo, min(mult, hi))


# --------------------------------------------------------------------------
# setup evaluation at current (closed) bar
# --------------------------------------------------------------------------


def _open_position(
    ctx: StrategyContext,
    sym: str,
    want_long: bool,
    params: Params,
    gate_ok: bool = True,
):
    """Emit an entry for one direction if the closed-bar stack qualifies.

    ``gate_ok`` is the timestamp-level regime-gate verdict computed once in
    ``on_candle``. It only constrains *long* entries (the gate is a bull/
    risk-on gate); short entries are unconstrained.
    """
    if want_long and not gate_ok:
        return
    o = ctx.ohlcv(sym)
    lb = params.lookback

    # 1) bias window = the 5 candles before the pullback → relative
    #    pullback sits at -2, confirmation at -1, so bias = -3 .. -(3+lb-1)
    bodies = [_body(o, -(3 + k)) for k in range(lb)]
    aligned = (
        sum(1 for b in bodies if (b > 0) if want_long)
        if want_long
        else sum(1 for b in bodies if (b < 0))
    )
    net_move = _close(o, -3) - _close(o, -(3 + (lb - 1)))
    bias_ok = aligned >= params.bias_threshold and (
        (net_move > 0) if want_long else (net_move < 0)
    )
    if not bias_ok:
        return
    med_body = float(np.median([abs(b) for b in bodies]))

    # 2) pullback (candle after bias; contra direction) — single candle
    pb_body = _body(o, -2)
    pulldir_ok = (pb_body < 0) if want_long else (pb_body > 0)
    if not pulldir_ok:
        return
    if abs(pb_body) < params.pullback_body_mult * med_body:
        return
    pb_open = _open(o, -2)
    pb_extreme = _low(o, -2) if want_long else _high(o, -2)  # contraside extreme

    # 3) confirmation (current closed bar, bias direction, sized, through open)
    cf_body = _body(o, -1)
    confdir_ok = (cf_body > 0) if want_long else (cf_body < 0)
    if not confdir_ok:
        return
    if abs(cf_body) < params.confirm_body_mult * abs(pb_body):
        return
    cfc = _close(o, -1)
    through_open = (cfc > pb_open) if want_long else (cfc < pb_open)
    if not through_open:
        return

    # optional ATR volatility band
    if params.atr_filter_on:
        a = atr_now(ctx, sym, params.atr_period)
        if np.isnan(a) or a <= 0:
            return
        cb = abs(cf_body)
        if not (params.atr_filter_lo * a <= cb <= params.atr_filter_hi * a):
            return

    # 4) levels & risk sizing from confirmation close reference
    entry_ref = cfc
    atr_val = atr_now(ctx, sym, params.atr_period)
    if np.isnan(atr_val) or atr_val <= 0:
        return

    # dynamic confidence sizing: scale base ``risk_pct`` up on the stronger
    # setups (full alignment / decisive confirm); flat 1.0 when disabled.
    eff_risk = params.risk_pct * dyn_risk_mult(params, aligned, cf_body, pb_body)

    if want_long:
        stop_level = pb_extreme - params.stop_atr_mult * atr_val
        stop_dist = entry_ref - stop_level
        if stop_dist <= 0:
            return
        sl_pct = stop_dist / entry_ref
    else:
        stop_level = pb_extreme + params.stop_atr_mult * atr_val
        stop_dist = stop_level - entry_ref
        if stop_dist <= 0:
            return
        sl_pct = stop_dist / entry_ref
    tp_pct = (params.tp_r * stop_dist) / entry_ref

    qty = risk_sized_qty(
        equity=ctx.state.portfolio.cash,
        price=entry_ref,
        stop_dist=stop_dist,
        risk_pct=eff_risk,
    )
    if qty <= 0:
        return
    size = qty * entry_ref / ctx.state.portfolio.initial_capital

    shared_last = ctx.shared.setdefault("last", {})
    idx = len(o.close)
    if idx - shared_last.get(sym, -(10**9)) < params.cooldown_bars:
        return  # reload guard (cooldown after prior open/close of the symbol)

    # Exit policy: in "trail" mode we disable the fixed TP bracket (tp=None)
    # and let the trail manager exit winners; the engine SL floor stays armed.
    open_tp_pct = tp_pct if getattr(params, "exit_mode", "fixed") == "fixed" else None

    if want_long:
        sl_abs = entry_ref * (1 - sl_pct)
        ctx.long(
            sym,
            size=size,
            sl=sl_pct,
            tp=open_tp_pct,
            reason="[bias5-displace] LONG",
            tag="b5l",
        )
    else:
        sl_abs = entry_ref * (1 + sl_pct)
        ctx.short(
            sym,
            size=size,
            sl=sl_pct,
            tp=open_tp_pct,
            reason="[bias5-displace] SHORT",
            tag="b5s",
        )
    shared_last[sym] = idx

    # Cross-candle trail bookkeeping (mirrors Position.stop_loss which the
    # engine sets from the same cfc anchor: engine SL_low == stop_level). The
    # manager only reads closed bars; it appends the entry-time note here so
    # the next candle can begin trailing/monotonic ratcheting from bar 0.
    if getattr(params, "exit_mode", "fixed") == "trail":
        trail = ctx.shared.setdefault("_trail", {})
        trail[sym] = {
            "side": "long" if want_long else "short",
            "entry": entry_ref,  # confirmation-close reference (cfc)
            "r_dist": stop_dist,  # 1R in price points
            "stop": sl_abs,  # engine-identical initial stop price
            "extreme": entry_ref,  # running extreme, seeded at entry
            "activated": False,  # becomes True after +activate_R x R
            "active_trail": None,  # persisted defensive trail level
            "done": False,  # close already emitted this bar
        }


# --------------------------------------------------------------------------
# trail exit policy (chandelier, closed-bar only)
# --------------------------------------------------------------------------


def _manage_trail(
    ctx: StrategyContext,
    sym: str,
    position,
    o: OhlcvView,
    params: Params,
) -> None:
    """Self-managed chandelier exit for exit_mode="trail".

    The engine SL (Position.stop_loss) stays armed from entry as a hard floor;
    this routine RIDES a winner by ratcheting a defensive ``active_trail`` up
    once the trade is ``activate_profit_R`` x R in profit, and exits via
    ``ctx.close`` (next-open fill guarded at the trail level) only when a NEW
    CLOSED bar breaks that level. Conservative by construction:

    * all reads are cursor-truncated (closed bars) -> no lookahead,
    * the exit is decided against a trail level persisted from the *prior*
      closed bar (no same-bar self-comparison against a level this bar lowered
      using this bar's high),
    * `ctx.close(... guard_price=act)` fills at worse-of(act, next_open), i.e.
      a bar that dips below the trail but gaps/opens higher still fills at the
      intended trail price (a stop-market sell) - never at a better price.
    """
    holder = ctx.shared.setdefault("_trail", {})
    rec = holder.get(sym)
    if rec is None:
        return  # no trail record (only should happen if state got GC'd)
    if rec["done"]:
        # A close is already queued (fills next open); wait for the lot to go
        # flat rather than double-emitting before the fill lands.
        return

    is_long = position.type == ActionType.long  # both mirror-ed paths below
    atr_val = atr_now(ctx, sym, params.atr_period)
    hi = _high(o, -1)
    lo = _low(o, -1)
    cl = _close(o, -1)
    stop = rec["stop"]

    if is_long:
        # 1) exit against the trail level persisted from the prior close.
        #    Only self-close when the dip pierced the TRAIL (act) but did NOT
        #    also go through the engine SL (stop) on this same bar: if lo also
        #    <= stop the engine's armed SL owns the exit (fills at stop, the
        #    correct worst case). Emitting our own close in that same-bar case
        #    would leave a stale close for a position the engine already closed
        #    (engine _close_position raises when the pid is gone) -> so we let
        #    the engine handle breaches at/through the SL and NEVER stack a
        #    second close on the same bar.
        act = rec.get("active_trail")
        if act is not None and act > stop and stop < lo <= act:
            # ctx.close fills next-open, guarded at the trail level (worse-of
            # act / next_open from the position side) => realistic intra-bar stop.
            ctx.close(sym, reason="[trail] chandelier hit", guard_price=act)
            rec["done"] = True
            return
        # 2) roll the running high-water extreme
        if hi > rec["extreme"]:
            rec["extreme"] = hi
        # 3) activation: once CLOSE is fully >= activate_R x R above entry
        if not rec["activated"] and not np.isnan(atr_val) and atr_val > 0:
            threshold = rec["entry"] + params.activate_profit_R * rec["r_dist"]
            if cl >= threshold and hi >= rec["entry"] + rec["r_dist"]:
                rec["activated"] = True
        # 4) ratchet the defensive trail up (monotonic, floored at the SL)
        #    -- only safe to do AFTER the exit check above so we never test a
        #    fresh level on the same bar that just produced it.
        if rec["activated"] and not np.isnan(atr_val) and atr_val > 0:
            base = max(rec["extreme"] - params.trail_atr_mult * atr_val, stop)
            cur = rec.get("active_trail") or stop
            rec["active_trail"] = base if base > cur else cur
    else:
        # mirror for shorts (chandelier below an upper extreme); the engine
        # SL floor (upper stop) owns breaches at/through stop, the strategy
        # self-closes only on trail breaks strictly below the stop.
        act = rec.get("active_trail")
        if act is not None and act < stop and hi >= act and hi < stop:
            ctx.close(sym, reason="[trail] chandelier hit", guard_price=act)
            rec["done"] = True
            return
        if lo < rec["extreme"]:
            rec["extreme"] = lo
        if not rec["activated"] and not np.isnan(atr_val) and atr_val > 0:
            threshold = rec["entry"] - params.activate_profit_R * rec["r_dist"]
            if cl <= threshold and lo <= rec["entry"] - rec["r_dist"]:
                rec["activated"] = True
        if rec["activated"] and not np.isnan(atr_val) and atr_val > 0:
            base = min(rec["extreme"] + params.trail_atr_mult * atr_val, stop)
            cur = rec.get("active_trail") or stop
            rec["active_trail"] = base if base < cur else cur


# --------------------------------------------------------------------------
# DSL entrypoint (single per-timestamp pass over every symbol)
# --------------------------------------------------------------------------


@strategy(bars="1d", stateful=True)
def on_candle(ctx: StrategyContext):
    # Warmup bars fill strategy state only — no fill can happen there, so the
    # DSL raises on any entry emitted during them. Accumulators above/below
    # still receive warmup bars because the engine feeds the store regardless.
    if ctx.phase == "warmup":
        return
    params = ctx.params
    if not isinstance(params, Params):
        params = Params.from_dict(ctx.params)

    # Regime gate: computed once per timestamp so every symbol sees the same
    # SPY verdict for the bar. With mode "none" (default) this is a no-op and
    # ungated runs stay byte-identical to the baseline. Observer symbol (SPY)
    # stays a normal tradeable in the universe here -- the gate is a filter on
    # SPY's own series, which is self-consistent for SPY longs and keeps the
    # A/B to a single variable (the gate).
    gate_ok = regime_gate_ok(ctx)

    # open-position cap (global)
    if params.max_positions:
        live = sum(len(ctx.state.portfolio.positions.get(s, ())) for s in ctx.symbols)
        if live >= params.max_positions:
            return

    trailling = params.exit_mode == "trail"

    longs_first = params.side in ("both", "long")
    shorts_too = params.side in ("both", "short")

    for sym in ctx.symbols:
        o = ctx.ohlcv(sym)
        if len(o.close) < max(params.warmup_bars, params.lookback + 2):
            continue
        entry_pos = ctx.position(sym)
        if entry_pos is not None:
            if trailling:
                _manage_trail(ctx, sym, entry_pos, o, params)
            continue

        if longs_first:
            _open_position(ctx, sym, True, params, gate_ok=gate_ok)
        if shorts_too:
            _open_position(ctx, sym, False, params, gate_ok=gate_ok)
