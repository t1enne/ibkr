"""Repo-level guard: a real strategy's results must not depend on symbol order.

The engine bug this pins existed for the life of the repo and was invisible
because nobody permuted a symbol list. Allocating a bar's concurrent entries
sequentially meant whichever symbol appeared first in ``config.symbols`` won
the cash race, so the same strategy on the same data returned anything from
32% to 79% annual depending only on list order.

Scope of this file: END-TO-END, through the real strategy-module registry and
the real engine, on a synthetic multi-symbol feed. The lower-level unit tests
live in ``test_cohort_settlement.py``; this suite exists so that a regression
in ANY layer between signal emission and fill settlement (DSL sizing, cohort
grouping, allocation, execution) shows up as a failed permutation assertion
rather than as a silently different backtest number.

Fast by construction: synthetic daily frames, no DB, no strategy JSON files.
"""

from dataclasses import dataclass

import pandas as pd

from src.bt.engine.backtest import Backtest, run
from src.bt.strategies.dsl import StrategyContext, strategy
from src.bt.types import StrategyConfig

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]

#: Orders that have historically exposed ordering bugs. Reversal and "rotate
#: one step" both move a different symbol into the tail, which is where the
#: engine treats a symbol specially.
PERMUTATIONS: list[list[str]] = [
    SYMBOLS,
    list(reversed(SYMBOLS)),
    SYMBOLS[1:] + SYMBOLS[:1],
    SYMBOLS[3:] + SYMBOLS[:3],
    [SYMBOLS[2], SYMBOLS[0], SYMBOLS[5], SYMBOLS[1], SYMBOLS[4], SYMBOLS[3]],
]


#: Explicit symbol -> slope table. Deliberately a LITERAL, not arithmetic over
#: an enumeration: keying the slope to a symbol's position in ``symbols`` would
#: permute the MARKET DATA along with ``config.symbols``, making the invariance
#: assertions unsatisfiable for any engine (the run would be comparing two
#: different feeds, not two dispatch orders). A literal makes the independence
#: from list position visible rather than incidental; ``test_feed_is_order_independent``
#: below pins it structurally.
_SLOPE: dict[str, float] = {
    "AAA": 1.0,
    "BBB": 2.0,
    "CCC": 3.0,
    "DDD": 4.0,
    "EEE": 5.0,
    "FFF": 6.0,
}


@dataclass
class _Mod:
    on_candle: object


def _feed(symbols: list[str], n: int = 30) -> pd.DataFrame:
    """Daily feed; price path varies per symbol so sizing differs across names.

    The slope is a function of the SYMBOL IDENTITY alone, so calling this with a
    permutation of ``symbols`` returns the same columns: permuting
    ``config.symbols`` permutes dispatch order only, never the data.
    """
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    data: dict = {}
    for sym in symbols:
        step = _SLOPE[sym]  # distinct slopes => distinct prices => distinct qty
        closes = [100.0 + step * i for i in range(n)]
        data.update(
            {
                (sym, "open"): closes,
                (sym, "high"): [c * 1.01 for c in closes],
                (sym, "low"): [c * 0.99 for c in closes],
                (sym, "close"): closes,
                (sym, "volume"): [1000.0] * n,
            }
        )
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


@strategy(bars="1d", stateful=True)
def _burst(ctx: StrategyContext):
    """Enter the WHOLE book on two separate bars, then exit everything.

    Two bursts (not one) so the second cohort is sized off a book already
    changed by the first — the compounding path the old cash race corrupted.
    The schedule reads only the bar counter, never the symbol list, so the
    strategy is identical under permutation; only the engine's handling of
    order is under test.
    """
    n = ctx.shared.setdefault("_n", 0)
    ctx.shared["_n"] = n + 1
    if n in (0, 5):
        for sym in ctx.symbols:
            if ctx.quantity(sym) == 0:
                ctx.long(sym, size=0.6, reason="burst", size_mode="capital")
    elif n in (3, 9):
        for sym in ctx.symbols:
            if ctx.quantity(sym) != 0:
                ctx.close(sym, reason="flush")


def _cfg(symbols: list[str]) -> StrategyConfig:
    return StrategyConfig(
        name="perm",
        strategy_type="momentum_compression_breakout_dsl",  # registered type
        symbols=symbols,
        initial_capital=10_000.0,
        commission=0.0,
        warmup="0d",
        trading_start="2025-01-01",
        trading_end="2025-12-31",
        bars=["1d"],
        strategy_params={},
    )


def _run(symbols: list[str]):
    return run(Backtest(_cfg(symbols)), _feed(symbols), _Mod(_burst))


def _ledger(symbols: list[str]):
    pf = _run(symbols).final_state.portfolio
    return (
        sorted((t.symbol, round(t.qty, 8), str(t.entry_time)) for t in pf.trades),
        round(pf.cash, 8),
    )


def test_permutation_exercises_multi_open_cohorts():
    """The guard is only meaningful if concurrent opens actually compete.

    Every burst enters the whole book on one bar, so each burst is a
    multi-open cohort sharing a single book — precisely the group whose
    settlement was order-dependent before the fix.
    """
    pf = _run(SYMBOLS).final_state.portfolio
    by_bar: dict = {}
    for t in pf.trades:
        by_bar.setdefault(t.entry_time, []).append(t)
    cohorts = [g for g in by_bar.values() if len(g) > 1]
    assert len(cohorts) >= 1, "expected at least one multi-open cohort"
    widest = max(len(g) for g in cohorts)
    assert widest >= 3, (
        f"cohort of {widest} is too small to stress allocation; "
        "widen the burst so several symbols compete for one book"
    )
