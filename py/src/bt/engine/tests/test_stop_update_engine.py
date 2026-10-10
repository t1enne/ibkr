"""Engine tests for the stop_update drain — arming levels without fills.

Pins two real regressions:

1. Timing: a ``stop_update`` armed on bar t produces ZERO fills/trades on bar t
   and fires as an INTRABAR risk event on bar t+1 AT THE TRIGGER (gap-adjusted)
   — not on bar t (no intra-bar look-ahead), not at bar t+1's open. A drain
   that applied the level before the arming bar's own risk check would stop the
   position on bar t; an arming that never reached the book would leak the
   whole next bar.
2. Plumbing: the action must NEVER reach ``exchange.execute_signal`` — a missed
   drain would silently funnel zero-qty fills into the cohort path, which the
   trades-level assertions alone cannot see (a zero-qty fill is a silent no-op
   at the portfolio layer).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from src.bt.engine.backtest import Backtest, run
from src.bt.exchange.sim import SimExchange
from src.bt.state import ActionType, Candle, ExecutionParams, FillEvent, TradeSignal
from src.bt.strategies.dsl import StrategyContext, strategy
from src.bt.types import StrategyConfig


@dataclass
class _FixtureMod:
    """Module-shaped holder the engine consumes (``.on_candle`` = adapter)."""

    on_candle: object


@strategy(bars="1d", stateful=True)
def _arm_only_once(ctx: StrategyContext) -> None:
    """Enter long once, then arm ONE absolute stop on the very next bar.

    ``sl=0.05`` on the entry is the dsl convention (fraction of the entry
    price at SIGNAL time): the entry signal is generated on bar1 at close 100
    and carries an absolute 95.0 stop, but fills at bar2's open (111). The
    arming bar computes an absolute 105.0 level off its own close; the bar
    AFTER that must be the one that stops the position, even though the arming
    bar's own low (96) traded below the NEW level (105) but above the OLD one
    (95) — the level must not be live on the bar that computed it.
    """
    if ctx.phase == "warmup":
        return
    sym = ctx.candle.symbol
    if ctx.quantity(sym) == 0:
        ctx.long(sym, size=0.9, sl=0.05, reason="entry")
        return
    if ctx.shared.get("armed"):
        return
    ctx.shared["armed"] = True
    ctx.set_stops(sym, sl=ctx.price(sym) - 5.0, reason="arm next bar")


def _cfg() -> StrategyConfig:
    # Zero friction so the risk fill lands EXACTLY at the level — the close
    # price then distinguishes "filled at the trigger" from "filled at the
    # next open" with no tolerance.
    # ``strategy_type`` must name a registered module — the engine resolves
    # params against the registry even when the run's strategy is this file's
    # local adapter (same pattern as test_warmup_phase).
    return StrategyConfig(
        name="t",
        strategy_type="vwatr_div_dsl",
        symbols=["AAPL"],
        initial_capital=10000.0,
        commission=0.0,
        warmup="0d",
        trading_start="2024-01-01",
        trading_end="2024-12-31",
        bars=["1d"],
        strategy_params={},
        benchmark_symbols=[],
        spread_bps=0.0,
        slippage_bps=0.0,
    )


def _data() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    data = {
        ("AAPL", "open"): [100.0, 111.0, 110.0],
        ("AAPL", "high"): [101.0, 111.0, 118.0],
        ("AAPL", "low"): [99.0, 96.0, 104.0],  # bar2 trips the NEW level, bar3 the OLD
        ("AAPL", "close"): [100.0, 110.0, 118.0],
        ("AAPL", "volume"): [1000.0, 1000.0, 1000.0],
    }
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_stop_update_armed_on_bar_t_fires_intrabar_on_bar_t_plus_1() -> None:
    """The armed level fires on the NEXT bar at the trigger, gap-adjusted.

    Timeline: bar1 opens 100.0 (sl=95.0 on entry); bar2 closes 110.0 and arms
    105.0 — its own low (96.0) is below the new level but above the old one, so
    a same-bar arming (look-ahead) would stop on bar2 at min(105, open 111);
    bar3 lows to 104.0, so the risk check must fire at the worse-of trigger:
    min(105, open 110) = 105.0 — NOT the next open (110). The run records ONE
    round-trip trade (a close updates the open Trade record in place) whose
    exit carries bar3's stamp.
    """
    results = run(Backtest(_cfg()), _data(), _FixtureMod(_arm_only_once))
    trades = results.pf.trades
    assert len(trades) == 1
    trade = trades[0]
    assert trade.entry_time == pd.Timestamp("2024-01-01")
    assert trade.entry_price == pytest.approx(111.0)  # next-open fill at bar2
    assert trade.close_reason == "sl"
    assert trade.exit_time == pd.Timestamp("2024-01-03")  # not bar2, not bar4
    assert trade.exit_price == pytest.approx(105.0)  # the level, not bar3 open


def test_stop_update_never_reaches_execute_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``execute_signal`` must never see a ``stop_update`` action.

    Regression: if the drain were missed, the update would be priced as a
    zero-qty fill and silently no-op at the portfolio layer — invisible to
    every trade-level assertion. The spy fails the test the moment the action
    slips through, and also proves the spy itself ran (it saw the entry fill).
    """

    class _SpyExchange(SimExchange):
        def __init__(self) -> None:
            super().__init__()
            self.seen_actions: list[ActionType] = []

        def execute_signal(
            self,
            signal: TradeSignal,
            tick: Candle,
            params: ExecutionParams,
        ) -> FillEvent:
            self.seen_actions.append(signal.action)
            return super().execute_signal(signal, tick, params)

    spy = _SpyExchange()
    monkeypatch.setattr("src.bt.exchange.default_exchange", lambda: spy)
    run(Backtest(_cfg()), _data(), _FixtureMod(_arm_only_once))

    assert ActionType.stop_update not in spy.seen_actions
    assert ActionType.long in spy.seen_actions  # spy really exercised the path
