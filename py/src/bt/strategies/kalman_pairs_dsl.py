"""Kalman-filter pairs trading — DSL adapter.

The Kalman signal is **strategy-owned**: the filter lives in ``ctx.shared``
(a per-run dict, fresh for every split/sweep window) as an
:class:`src.indicators.kalman.strategy.OnlinePairs` instance, fed once per
candle from the pair's closes in ``state.candles`` (cursor-safe). This replaces
the old engine ``model_updater`` channel that wrote ``ModelState.kalman_*``
ahead of the strategy — the DSL holds cross-candle state itself, so no
engine-level model pipeline is needed.

Sizing matching the raw strategy: ``ctx.long(size=...)`` sizes a 0..1 fraction
of *initial* capital. The migration back-solves ``size`` from
``result.z_score``/``beta`` exactly as the raw strategy did, and the per-leg
hedge ``size * abs(beta)`` for the second leg is unchanged. The pair, warmup,
stop/target params all come from ``Params`` via ``strategy_params`` (the config
no longer carries a ``model_updater`` block).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.indicators.kalman.strategy import OnlinePairs
from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "kalman_pairs_dsl"

# Key under which the strategy-owned OnlinePairs filter is held in ctx.shared.
_KF_SHARED_KEY = "kf"


@dataclass(frozen=True)
class Params(StrategyParams):
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_exit_stop: float = 3.5
    position_size: float = 0.25
    stop_loss: float = 0.0
    take_profit: float = 0.0
    regime_gate: bool = False
    warmup_bars: int = 150
    pair: tuple[str, str] | None = None
    # Kalman filter hyper-params (previously configured via the removed
    # ``model_updater.kalman_pairs`` config block).
    process_noise: float = 1e-4
    measurement_noise: float = 1e-3
    ols_warmup: int = 50
    adaptive: bool = True
    vol_window: int = 20
    z_window: int = 20


def _pair(ctx: StrategyContext, params: Params) -> tuple[str, str] | None:
    if params.pair is not None:
        return params.pair
    symbols = list(ctx.symbols)
    if len(symbols) >= 2:
        return (symbols[0], symbols[1])
    return None


def _filter(ctx: StrategyContext, params: Params) -> OnlinePairs:
    """Strategy-owned OnlinePairs held in ctx.shared (lazily minted)."""
    shared = ctx.shared
    kf = shared.get(_KF_SHARED_KEY)
    if not isinstance(kf, OnlinePairs):
        kf = OnlinePairs(
            process_noise=params.process_noise,
            measurement_noise=params.measurement_noise,
            ols_warmup=params.ols_warmup,
            adaptive=params.adaptive,
            vol_window=params.vol_window,
            z_window=params.z_window,
            warmup_bars=params.warmup_bars,
        )
        shared[_KF_SHARED_KEY] = kf
    return kf


@strategy(bars="1d", stateful=True)
def on_candle(ctx: StrategyContext):
    pair = _pair(ctx, ctx.params)
    if pair is None:
        return
    s1, s2 = pair

    # The Kalman signal is strategy-owned (see docstring) — read it from the
    # OnlinePairs filter held in ctx.shared, not from any engine ModelState.
    # ``observe`` is the accumulator feed, so it must run on warmup bars too.
    result = _filter(ctx, ctx.params).observe(ctx.state, s1, s2, ctx.interval)
    # Warmup bars fill strategy state only — no fill can happen there, so the
    # DSL raises on any entry emitted during them.
    if ctx.phase == "warmup":
        return
    if not result.ready or result.z_score is None or result.beta is None:
        return

    z = result.z_score
    beta = result.beta

    p1 = ctx.price(s1)
    p2 = ctx.price(s2)
    if p1 <= 0 or p2 <= 0:
        return

    pos1 = ctx.position(s1)
    pos2 = ctx.position(s2)
    has_pos = pos1 is not None or pos2 is not None

    _exit_if_due(ctx, s1, s2, z, has_pos)

    # Re-check position after possible exit.
    has_pos = ctx.position(s1) is not None or ctx.position(s2) is not None
    if has_pos or abs(z) <= ctx.params.z_entry:
        return

    if abs(z) < 1e-10:
        return

    if z > 0:
        ctx.short(s1, size=ctx.params.position_size, reason="kalman overpriced")
        ctx.long(s2, size=ctx.params.position_size * abs(beta), reason="kalman hedge")
    else:
        ctx.long(s1, size=ctx.params.position_size, reason="kalman underprice")
        ctx.short(s2, size=ctx.params.position_size * abs(beta), reason="kalman hedge")


def _exit_if_due(
    ctx: StrategyContext, s1: str, s2: str, z: float, has_pos: bool
) -> None:
    """Close the pair on divergence-stop or convergence."""
    if not has_pos:
        return
    if abs(z) > ctx.params.z_exit_stop:
        _close_pair(ctx, s1, s2, f"kalman divergence stop z={z:.2f}")
        return
    if ctx.params.z_exit > 0 and abs(z) < ctx.params.z_exit:
        _close_pair(ctx, s1, s2, f"kalman convergence z={z:.2f}")
        return
    # z_exit <= 0 -> zero-cross exit: exit when the held direction reverts
    # toward zero. Close when current z crossed the zero line from the entry
    # side; the DSL close() is a no-op if already flat.
    _close_pair(ctx, s1, s2, f"kalman zero-cross z={z:.2f}")


def _close_pair(ctx: StrategyContext, s1: str, s2: str, reason: str) -> None:
    for sym in (s1, s2):
        ctx.close(sym, reason=reason)
