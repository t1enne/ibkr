"""Equal-weight buy & hold — a portfolio-fitness (pf) tester.

Allocates capital **equal-weight across every configured symbol once**, then
holds to the end. There is no market-timing, no exit logic and no rebalancing
beyond the initial allocation — exactly what you want from a capacity / fitness
baseline for a portfolio engine (``pf_*`` family). It answers: "what's the
buy & hold, equal-weight portfolio return, and how do per-symbol drawdowns &
contributions behave?"

Weighting: each symbol gets ``1 / len(symbols)`` of *initial* capital. The DSL
``size`` is a 0..1 fraction of initial capital converted to an absolute share
count (``size * initial_capital / close``), so ``long(sym, size=1/n)`` yields a
rigorously equal *dollar* allocation per leg — independent of each symbol's
price level.

Statefulness: the one-shot allocation must fire on the **first** ``on_candle``
dispatch and never again, but each position only exists (fills) from the next
bar's open. We track the ``done`` flag in the framework-owned ``ctx.shared``
per-run holder (``@strategy(stateful=True)``) so split/sweep windows each get a
fresh holder — no module-scope bleed, thread-safe across workers.

By default this is a pure buy & hold. Optionally set ``rebalance_days`` (> 0)
to periodically re-slice overweight legs back toward ``1/n`` — useful for
comparing "buy & hold" against "equal-weight *rebalanced*" as a volatility /
drift control in pf tests. Keep it off for a strict buy & hold.

This is a pf-tester strategy: it is intentionally dumb and deliberately ignores
SL/TP/risk logic. It is not a tradeable signal generator.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "pf_equal_weight"


@dataclass(frozen=True)
class Params(StrategyParams):
    # Periodic re-slicing back to equal weight, in bars. 0 = pure buy & hold.
    rebalance_days: int = 0


@strategy(bars="1d", stateful=True)
def on_candle(ctx: StrategyContext):
    """Open equal-weight longs on the first bar; optional periodic rebalance.

    ``ctx.symbols`` is the full cross-section (the engine dispatches this
    ``on_candle`` on the last symbol's bar, so every symbol is in the store).
    ``ctx.shared`` gates the one-shot allocation; drift re-alignment is tracked
    by a bar counter when ``rebalance_days`` > 0.
    """
    # Warmup bars fill strategy state only — no fill can happen there, so the
    # DSL raises on any entry emitted during them. Accumulators above/below
    # still receive warmup bars because the engine feeds the store regardless.
    if ctx.phase == "warmup":
        return
    n = len(ctx.symbols)
    size = round(1 / n, 2)

    has_traded = len(ctx.state.portfolio.trades) > 0
    if not has_traded:
        # First dispatch: establish the equal-weight book across every symbol.
        for sym in ctx.symbols:
            if ctx.quantity(sym) == 0:
                ctx.long(sym, size=size, reason="equal-weight allocation")
        ctx.shared["bar"] = 0
        if ctx.params.rebalance_days <= 0:
            return

    # Optional periodic drift rebalance (buy & hold leaves this off).
    bar = ctx.shared.get("bar", 0) + 1
    ctx.shared["bar"] = bar

    # must rebalance?
    if ctx.params.rebalance_days <= 0 or bar % ctx.params.rebalance_days != 0:
        return

    equity = ctx.state.portfolio.equity_curve[-1].equity

    target = size * equity

    for sym in ctx.symbols:
        price = ctx.price(sym)
        if price <= 0:
            continue
        value = ctx.quantity(sym) * price  # signed; long book => positive
        delta_value = target - value
        if delta_value > 1e-6:
            # Underweight: top up to the equal-weight share count.
            new_shares = delta_value / price
            if new_shares > 0:
                ctx.long(
                    sym,
                    size=new_shares * price / ctx.state.portfolio.initial_capital,
                    reason="size up",
                )
        elif delta_value < -1e-6:
            # Overweight: shed shares back toward equal weight (netting reduce).
            shed = -delta_value / price
            cur = ctx.quantity(sym)
            if cur > 0:
                ctx.partial_close(sym, qty=min(1.0, shed / cur), reason="size down")
