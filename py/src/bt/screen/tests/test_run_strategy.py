"""Tests for the strategy-as-screen posture projection ("new-computation").

Covers the screen driver's own logic — replaying a symbol's captured signal
feed into an (action, score, signals) row. Signal *capture* itself (the engine
observer firing once per fresh emission, pre-_finalize) is covered by the
engine tests; feeding a stubbed engine is not re-tested here (no new signal).
"""

from typing import cast

import pandas as pd

from src.bt.screen.run_strategy import (
    Posture,
    ScreenRow,
    ScreenRun,
    SignalCollector,
    _filter_recent,
    _project,
    _resolve_posture,
)
from src.bt.state import (
    ActionType,
    BacktestState,
    TradeSignal,
)
from src.bt.types import StrategyConfig

TS: pd.Timestamp = cast(pd.Timestamp, pd.Timestamp("2025-06-10"))
TS_PRIOR: pd.Timestamp = cast(pd.Timestamp, pd.Timestamp("2025-06-09"))


def _sig(
    action: ActionType,
    sym: str = "AAPL",
    ts: pd.Timestamp = TS,
    *,
    qty: float = 0.0,
    price: float = 100.0,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    position_id: str | None = None,
    tag: str = "",
) -> TradeSignal:
    return TradeSignal(
        action=action,
        symbol=sym,
        timestamp=ts,
        price=price,
        qty=qty,
        reason=f"{action.value} {sym}",
        stop_loss=stop_loss,
        take_profit=take_profit,
        position_id=position_id,
        tag=tag,
    )


def test_fresh_open_on_newest_bar_scores_open():
    # long fired on the newest scored bar -> fresh action, 1.0
    rp = _resolve_posture((_sig(ActionType.long),), Posture(), TS)
    assert rp.action == "long"
    assert rp.score == Posture().base_score_open
    assert rp.reasons == ("long AAPL",)
    assert rp.sig_ts == TS  # fresh: signal on newest bar
    assert rp.signal is not None and rp.signal.action == ActionType.long


def test_prior_intent_is_retained_not_fresh():
    # long fired on an earlier bar, nothing newer -> held setup, 0.8
    rp = _resolve_posture((_sig(ActionType.long, ts=TS_PRIOR),), Posture(), TS)
    assert rp.action == "long"
    assert rp.score == Posture().base_score_held
    assert rp.sig_ts == TS_PRIOR  # signal is older than newest data -> stale setup


def test_per_symbol_freshness_not_shared_feed_max():
    # AAPL's own last bar is TS (decided fresh -> 1.0); NVDA's bar ends later.
    # A shared feed-max (NVDA's TS_LATER) must not demote AAPL's same-bar
    # trigger to "held" across the divergent per-symbol calendars.
    TS_LATER = cast(pd.Timestamp, pd.Timestamp("2025-06-11"))
    collector = SignalCollector(("AAPL", "NVDA"))
    collector.on_signal(_sig(ActionType.long, "AAPL", ts=TS))  # own fresh
    rows = _project(
        collector, ("AAPL", "NVDA"), Posture(), {"AAPL": TS, "NVDA": TS_LATER}
    )
    by_sym = {r.symbol: r for r in rows}
    assert by_sym["AAPL"].score == Posture().base_score_open  # 1.0, not 0.8
    assert by_sym["AAPL"].ts == TS
    assert by_sym["NVDA"].ts == TS_LATER  # row stamped with its own last bar


def _row(sym: str, sig_ts: pd.Timestamp | None, ts: pd.Timestamp = TS) -> ScreenRow:
    return ScreenRow(
        symbol=sym,
        action="long" if sig_ts is not None else "flat",
        score=1.0 if sig_ts is not None else 0.0,
        signals=(),
        ts=ts,
        sig_ts=sig_ts,
    )


def test_filter_recent_drops_stale_and_flat():
    old = cast(pd.Timestamp, pd.Timestamp("2025-05-01"))  # 40d before TS
    rows = (_row("AAPL", TS), _row("MSFT", old), _row("NVDA", None))
    assert [r.symbol for r in _filter_recent(rows, 5)] == ["AAPL"]
    assert _filter_recent(rows, None) == rows


def test_filter_recent_keeps_close_rows():
    # A close is actionable intent, not an inert flat row — it must survive the
    # staleness window and reach a live consumer.
    close_row = ScreenRow(
        symbol="AAPL", action="close", score=1.0, signals=(), ts=TS, sig_ts=TS
    )
    assert _filter_recent((close_row,), 5) == (close_row,)


def _config() -> StrategyConfig:
    return StrategyConfig(
        name="t",
        strategy_type="trend",
        symbols=["AAPL", "MSFT"],
        initial_capital=100_000.0,
        commission=0.0,
        warmup="0d",
        trading_start="2025-01-01",
        trading_end="2025-06-10",
        bars=["1d"],
        strategy_params={},
    )


def _run(rows: tuple[ScreenRow, ...]) -> ScreenRun:
    # render_screen_json never touches state; cast a stand-in to keep the test
    # off the DB/engine path (the driver's engine wiring is tested elsewhere).
    return ScreenRun(rows=rows, state=cast(BacktestState, None), config=_config())
