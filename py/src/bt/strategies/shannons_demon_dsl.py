"""Shannon's Demon — volatility-harvesting via periodic rebalancing (DSL).

DSL-native port of ``shannons_demon`` that rebalances with the
**add/trim** model instead of the netting ``rebalance`` action:

* **add**  — an underweight leg is topped up with ``ctx.long(..., tag=...)``,
  opening a *fresh* lot (Pine ``entry`` semantics).
* **trim** — an overweight leg sheds shares with ``ctx.partial_close`` across
  its lots (oldest-first), realizing PnL on the released shares and keeping each
  survivor lot's cost basis intact.

Reads aggregate by default: ``ctx.quantity(sym)`` (net signed), ``ctx.avg_entry``,
``ctx.position_ids`` — so the strategy targets per-symbol weights without
hand-managing the underlying lot splits. Each add contributes its own trade and
each partial close realizes a closed-trade slice; ``n_trades`` therefore tracks
honest activity (entries + trims + final closes) rather than the raw strategy's
single editable ``rebalance`` position.

Only long legs are traded (target weights are all positive). Supports the same
params as the raw strategy (rebalance_frequency, target_weights, position_size,
drift_tolerance, warmup_bars, cash_leg, trend_gate_*).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "shannons_demon_dsl"


@dataclass(frozen=True)
class Params(StrategyParams):
    """Mirrors ``shannons_demon.Params``.

    ``target_weights`` is a tuple so it survives ``StrategyParams.from_dict``;
    the overridden ``from_dict`` normalises a list from JSON to a tuple.
    """

    rebalance_frequency: int = 21
    target_weights: tuple[float, ...] = (0.5, 0.5)
    position_size: float = 1.0
    drift_tolerance: float = 0.05
    warmup_bars: int = 21
    cash_leg: bool = False
    trend_gate_enabled: bool = False
    trend_gate_lookback: int = 200
    trend_gate_threshold: float = 0.10
    # Sub-share threshold for an add/trim to be worth doing (dust guard).
    min_trade_shares: float = 1e-6

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        d = dict(d)
        if "target_weights" in d and isinstance(d["target_weights"], list):
            d["target_weights"] = tuple(d["target_weights"])
        return super().from_dict(d)


# ---------------------------------------------------------------------------
# pure helpers (cursor-safe, no engine mutation)
# ---------------------------------------------------------------------------


def _legs(ctx: StrategyContext, params: Params) -> list[str]:
    """The symbols to trade: first symbol (cash leg) or first two symbols."""
    symbols = list(ctx.symbols)
    if params.cash_leg:
        return [symbols[0]] if symbols else []
    return symbols[:2] if len(symbols) >= 2 else []


def _target_weights(legs: list[str], params: Params) -> dict[str, float]:
    """{symbol: effective target weight} after position_size scaling."""
    tw = params.target_weights
    if len(tw) != len(legs):
        tw = (0.5, 0.5) if len(legs) == 2 else (1.0,)
    return {sym: w * params.position_size for sym, w in zip(legs, tw)}


def _portfolio_value(ctx: StrategyContext, legs: list[str]) -> float:
    """Total portfolio value = open leg positions (live close) + cash."""
    pos_value = 0.0
    for sym in legs:
        price = ctx.price(sym)
        for p in ctx.state.portfolio.positions.get(sym, ()):
            pos_value += abs(p.qty) * price
    return pos_value + ctx.state.portfolio.cash


def _current_weights(
    ctx: StrategyContext, legs: list[str], total: float
) -> dict[str, float]:
    """{symbol: current weight} of each held leg vs total portfolio value."""
    if total <= 0:
        return {}
    weights: dict[str, float] = {}
    for sym in legs:
        qty = ctx.quantity(sym)
        if abs(qty) > 0:
            weights[sym] = (abs(qty) * ctx.price(sym)) / total
    return weights


def _drift_exceeds(
    current: dict[str, float],
    target: dict[str, float],
    tolerance: float,
) -> bool:
    return any(abs(current.get(k, 0.0) - tw) > tolerance for k, tw in target.items())


def _ratio_trending(ctx: StrategyContext, legs: list[str], params: Params) -> bool:
    """True if legs[0]/legs[1] ratio deviates from its SMA beyond threshold."""
    if len(legs) < 2:
        return False
    a = ctx.ohlcv(legs[0]).close.to_array()
    b = ctx.ohlcv(legs[1]).close.to_array()
    n = min(len(a), len(b))
    if n < params.trend_gate_lookback + 1:
        return False
    ratio = a[-n:] / b[-n:]
    sma = float(np.mean(ratio[-params.trend_gate_lookback :]))
    if not np.isfinite(sma) or sma <= 0:
        return False
    return abs(float(ratio[-1]) / sma - 1) > params.trend_gate_threshold


def _lot_qty(ctx: StrategyContext, sym: str, pid: str) -> float:
    """Current quantity of the lot ``pid`` in ``sym`` (0 if not held)."""
    for p in ctx.state.portfolio.positions.get(sym, ()):
        if p.position_id == pid:
            return float(p.qty)
    return 0.0


def _rebalance_leg(
    ctx: StrategyContext,
    sym: str,
    target_qty: float,
    tag: str,
) -> None:
    """Drive an aggregate leg position (``ctx.quantity``) toward ``target_qty``.

    * underweight → ``ctx.long`` a fresh lot sized to the deficit,
    * overweight  → ``ctx.partial_close`` across lots, oldest-first.

    Never full-closes and reopens a held leg — the add/trim path avoids the
    close/reopen churn (and its doubled trade count).
    """
    params = ctx.params
    current = ctx.quantity(sym)
    delta = target_qty - current
    if -params.min_trade_shares <= delta <= params.min_trade_shares:
        return

    initial = ctx.state.portfolio.initial_capital
    price = ctx.price(sym)

    if delta > 0:
        # Add: size is a fraction of initial capital; ctx.long emits
        # size*initial_capital/price shares == delta.
        size = delta * price / initial if initial > 0 else 0.0
        if size > 0:
            ctx.long(sym, size=size, tag=tag, reason=f"rebalance add {sym}")
        return

    # Trim: shed -delta shares across stacked lots, oldest position first so
    # long-held basis clears before recent adds. partial_close args are a
    # fraction (0,1] of the target lot; reduce `need` as shares are released.
    need = -delta
    for pid in ctx.position_ids(sym):
        if need <= params.min_trade_shares:
            break
        lot = _lot_qty(ctx, sym, pid)
        if lot <= 0:
            continue
        frac = min(need / lot, 1.0)
        if frac <= 0:
            continue
        ctx.partial_close(sym, qty=frac, lot=pid, reason=f"rebalance trim {sym}")
        need -= lot * frac


# ---------------------------------------------------------------------------
# DSL strategy
# ---------------------------------------------------------------------------


@strategy(bars="1d", stateful=True)
def on_candle(ctx: StrategyContext):
    # Warmup bars fill strategy state only — no fill can happen there, so the
    # DSL raises on any entry emitted during them. Accumulators above/below
    # still receive warmup bars because the engine feeds the store regardless.
    if ctx.phase == "warmup":
        return
    shared = ctx.shared
    shared.setdefault("bar_idx", 0)
    shared.setdefault("last_rebalance", -ctx.params.rebalance_frequency)
    shared.setdefault("tags", {})

    legs = _legs(ctx, ctx.params)
    if not legs:
        return

    if len(ctx.ohlcv(legs[0]).close) < ctx.params.warmup_bars:
        return

    shared["bar_idx"] += 1
    bars_since = shared["bar_idx"] - shared["last_rebalance"]
    if bars_since < ctx.params.rebalance_frequency:
        return

    total = _portfolio_value(ctx, legs)
    if total <= 0:
        return

    target = _target_weights(legs, ctx.params)

    # First deployment: open each leg at its target weight.
    if not ctx.state.portfolio.positions:
        shared["last_rebalance"] = shared["bar_idx"]
        for sym in legs:
            w = target.get(sym, 0.0)
            price = ctx.price(sym)
            if w <= 0 or price <= 0:
                continue
            shared["tags"][sym] = f"sh-{sym}"
            ctx.long(
                sym,
                size=w * total / ctx.state.portfolio.initial_capital,
                tag=shared["tags"][sym],
                reason=f"deploy {sym}",
            )
        return

    current = _current_weights(ctx, legs, total)
    if not _drift_exceeds(current, target, ctx.params.drift_tolerance):
        return

    if ctx.params.trend_gate_enabled and _ratio_trending(ctx, legs, ctx.params):
        return

    shared["last_rebalance"] = shared["bar_idx"]
    for sym in legs:
        w = target.get(sym, 0.0)
        if w <= 0 or ctx.price(sym) <= 0:
            continue
        tag = shared["tags"].setdefault(sym, f"sh-{sym}")
        target_qty = (w * total) / ctx.price(sym)
        _rebalance_leg(ctx, sym, target_qty, tag)
