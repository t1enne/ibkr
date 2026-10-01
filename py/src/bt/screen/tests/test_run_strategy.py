"""Tests for the strategy-as-screen posture projection ("new-computation").

Covers the screen driver's own logic — replaying a symbol's captured signal
feed into an (action, score, signals) row. Signal *capture* itself (the engine
observer firing once per fresh emission, pre-_finalize) is covered by the
engine tests; feeding a stubbed engine is not re-tested here (no new signal).
"""

from typing import cast

import pandas as pd
import pytest

from src.bt.screen.run_strategy import (
    Posture,
    ScreenRow,
    ScreenRun,
    SignalCollector,
    _filter_recent,
    _project,
    _resolve_latest,
    _resolve_posture,
    _side_of,
    render_screen_json,
)
from src.bt.state import ActionType, BacktestState, TradeSignal
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


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (ActionType.long, "long"),
        (ActionType.short, "short"),
        (ActionType.close, "close"),  # explicit exit directive
        (ActionType.rebalance, None),  # keeps the incumbent side
    ],
)
def test_side_of(action, expected):
    assert _side_of(_sig(action)) == expected


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


def test_close_after_open_surfaces_close_signal():
    # The exit must survive projection as an actionable ``close`` row, not be
    # collapsed to inert ``flat`` — a live consumer needs the exit directive.
    feed = (_sig(ActionType.long, ts=TS_PRIOR), _sig(ActionType.close, ts=TS))
    rp = _resolve_posture(feed, Posture(), TS)
    assert rp.action == "close"
    assert rp.score == Posture().base_score_open  # fresh on the newest bar
    assert rp.sig_ts == TS
    assert rp.signal is not None and rp.signal.action == ActionType.close


def test_close_with_no_prior_position_is_still_close():
    # A close is emitted by the strategy; the screen does not infer book state.
    rp = _resolve_posture((_sig(ActionType.close),), Posture(), TS)
    assert rp.action == "close"


def test_rebalance_keeps_incumbent_side():
    feed = (_sig(ActionType.long, ts=TS_PRIOR), _sig(ActionType.rebalance, ts=TS))
    rp = _resolve_posture(feed, Posture(), TS)
    assert rp.action == "long"  # a rebalance is not a reposition
    assert rp.sig_ts == TS_PRIOR


def test_latest_wins_when_reopened_after_close():
    feed = (
        _sig(ActionType.long, ts=TS_PRIOR),
        _sig(ActionType.close, ts=TS_PRIOR),
        _sig(ActionType.short, ts=TS),  # newest sets the side
    )
    rp = _resolve_posture(feed, Posture(), TS)
    assert rp.action == "short"
    assert rp.score == Posture().base_score_open
    assert rp.sig_ts == TS  # reopened on newest bar


def test_no_signal_is_flat():
    rp = _resolve_posture((), Posture(), TS)
    assert rp.action == "flat"
    assert rp.score == 0.0
    assert rp.signal is None


def test_project_ranks_actions_before_flat_and_honors_include_flat():
    newest = TS
    collector = SignalCollector(("AAPL", "MSFT", "NVDA"))
    # AAPL: fresh short (1.0); MSFT: held long (0.8); NVDA: no signal -> flat.
    collector.on_signal(_sig(ActionType.short, "AAPL", newest))
    collector.on_signal(_sig(ActionType.long, "MSFT", TS_PRIOR))

    rows = _project(collector, ("AAPL", "MSFT", "NVDA"), Posture(), newest)
    # ranked: fresh open first, held second, flat last
    ordered = [(r.symbol, r.action, r.score) for r in rows]
    assert ordered[0] == ("AAPL", "short", 1.0)
    assert ordered[1] == ("MSFT", "long", 0.8)
    assert ordered[2] == ("NVDA", "flat", 0.0)

    pruned = _project(
        collector, ("AAPL", "MSFT", "NVDA"), Posture(include_flat=False), newest
    )
    assert [r.symbol for r in pruned] == ["AAPL", "MSFT"]


def test_posture_defaults_are_exported_constants():
    p = Posture()
    assert p.base_score_open == 1.0
    assert p.base_score_held == 0.8
    assert p.include_flat is True


def test_resolve_latest_scalar_passthrough():
    assert _resolve_latest(TS, "AAPL") == TS


def test_resolve_latest_uses_symbols_own_bar():
    # Two symbols end on different days (stale listing vs live) after re-anchor.
    latest = {"AAPL": TS, "MSFT": TS_PRIOR}
    assert _resolve_latest(latest, "AAPL") == TS
    assert _resolve_latest(latest, "MSFT") == TS_PRIOR


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


def test_project_scalar_latest_still_supported():
    collector = SignalCollector(("AAPL",))
    collector.on_signal(_sig(ActionType.short, "AAPL", ts=TS))
    rows = _project(collector, ("AAPL",), Posture(), TS)
    assert rows[0].score == Posture().base_score_open


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


def test_project_carries_executable_signal_fields():
    collector = SignalCollector(("AAPL",))
    collector.on_signal(
        _sig(
            ActionType.long,
            "AAPL",
            qty=12.0,
            price=101.5,
            stop_loss=95.0,
            take_profit=120.0,
            position_id="lot-1",
            tag="trend",
        )
    )
    (row,) = _project(collector, ("AAPL",), Posture(), TS)
    assert row.price == 101.5
    assert row.qty == 12.0
    assert row.stop_loss == 95.0
    assert row.take_profit == 120.0
    assert row.position_id == "lot-1"
    assert row.tag == "trend"


def test_flat_row_has_inert_executable_fields():
    collector = SignalCollector(("AAPL",))
    (row,) = _project(collector, ("AAPL",), Posture(), TS)
    assert row.action == "flat"
    assert row.price == 0.0
    assert row.qty == 0.0
    assert row.stop_loss is None and row.take_profit is None
    assert row.position_id is None and row.tag == ""


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


def test_render_screen_json_emits_only_actionable_rows():
    rows = (
        _row("AAPL", TS),  # long, actionable
        ScreenRow(
            symbol="MSFT", action="close", score=1.0, signals=(), ts=TS, sig_ts=TS
        ),
        ScreenRow(symbol="NVDA", action="flat", score=0.0, signals=(), ts=TS),
    )
    payload = render_screen_json(_run(rows), strategy="strats/wip/t.json")
    assert payload["command"] == "screen"
    assert payload["strategy_type"] == "trend"
    assert payload["bars"] == "1d"
    # flat excluded — a live consumer must not read it as "flatten"
    assert [s["symbol"] for s in payload["signals"]] == ["AAPL", "MSFT"]
    assert [s["action"] for s in payload["signals"]] == ["long", "close"]


def test_render_screen_json_serializes_executable_fields():
    collector = SignalCollector(("AAPL",))
    collector.on_signal(
        _sig(
            ActionType.long,
            "AAPL",
            qty=12.0,
            price=101.5,
            stop_loss=95.0,
            take_profit=120.0,
            position_id="lot-1",
            tag="trend",
        )
    )
    rows = _project(collector, ("AAPL",), Posture(), TS)
    payload = render_screen_json(_run(rows), strategy="s.json")
    (sig,) = payload["signals"]
    assert sig["action"] == "long"
    assert sig["symbol"] == "AAPL"
    assert sig["qty"] == 12.0 and sig["price"] == 101.5
    assert sig["stop_loss"] == 95.0 and sig["take_profit"] == 120.0
    assert sig["position_id"] == "lot-1" and sig["tag"] == "trend"
    assert sig["signal_ts"] == str(TS) and sig["data_ts"] == str(TS)
    assert sig["reasons"] == ["long AAPL"]
