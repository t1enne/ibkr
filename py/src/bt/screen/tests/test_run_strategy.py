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
    SignalCollector,
    _project,
    _resolve_posture,
    _side_of,
)
from src.bt.state import ActionType, TradeSignal

TS: pd.Timestamp = cast(pd.Timestamp, pd.Timestamp("2025-06-10"))
TS_PRIOR: pd.Timestamp = cast(pd.Timestamp, pd.Timestamp("2025-06-09"))


def _sig(action: ActionType, sym: str = "AAPL", ts: pd.Timestamp = TS) -> TradeSignal:
    return TradeSignal(
        action=action,
        symbol=sym,
        timestamp=ts,
        price=100.0,
        reason=f"{action.value} {sym}",
    )


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (ActionType.long, "long"),
        (ActionType.short, "short"),
        (ActionType.close, "flat"),
        (ActionType.rebalance, None),  # keeps the incumbent side
    ],
)
def test_side_of(action, expected):
    assert _side_of(_sig(action)) == expected


def test_fresh_open_on_newest_bar_scores_open():
    # long fired on the newest scored bar -> fresh open, 1.0
    action, score, reasons = _resolve_posture((_sig(ActionType.long),), Posture(), TS)
    assert action == "long"
    assert score == Posture().base_score_open
    assert reasons == ("long AAPL",)


def test_prior_intent_is_retained_not_fresh():
    # long fired on an earlier bar, nothing newer -> held setup, 0.8
    action, score, _ = _resolve_posture(
        (_sig(ActionType.long, ts=TS_PRIOR),), Posture(), TS
    )
    assert action == "long"
    assert score == Posture().base_score_held


def test_close_after_open_reverts_to_flat():
    feed = (_sig(ActionType.long, ts=TS_PRIOR), _sig(ActionType.close, ts=TS))
    action, score, _ = _resolve_posture(feed, Posture(), TS)
    assert action == "flat"
    assert score == 0.0


def test_rebalance_keeps_incumbent_side():
    feed = (_sig(ActionType.long, ts=TS_PRIOR), _sig(ActionType.rebalance, ts=TS))
    action, score, _ = _resolve_posture(feed, Posture(), TS)
    assert action == "long"  # a rebalance is not a reposition


def test_latest_wins_when_reopened_after_close():
    feed = (
        _sig(ActionType.long, ts=TS_PRIOR),
        _sig(ActionType.close, ts=TS_PRIOR),
        _sig(ActionType.short, ts=TS),  # newest sets the side
    )
    action, score, _ = _resolve_posture(feed, Posture(), TS)
    assert action == "short"
    assert score == Posture().base_score_open


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
