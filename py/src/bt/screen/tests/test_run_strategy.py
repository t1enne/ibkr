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
    _resolve_latest,
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
    action, score, reasons, sig_ts = _resolve_posture(
        (_sig(ActionType.long),), Posture(), TS
    )
    assert action == "long"
    assert score == Posture().base_score_open
    assert reasons == ("long AAPL",)
    assert sig_ts == TS  # fresh: signal on newest bar


def test_prior_intent_is_retained_not_fresh():
    # long fired on an earlier bar, nothing newer -> held setup, 0.8
    action, score, _, sig_ts = _resolve_posture(
        (_sig(ActionType.long, ts=TS_PRIOR),), Posture(), TS
    )
    assert action == "long"
    assert score == Posture().base_score_held
    assert sig_ts == TS_PRIOR  # signal is older than newest data -> stale setup


def test_close_after_open_reverts_to_flat():
    feed = (_sig(ActionType.long, ts=TS_PRIOR), _sig(ActionType.close, ts=TS))
    action, score, _, sig_ts = _resolve_posture(feed, Posture(), TS)
    assert action == "flat"
    assert score == 0.0
    assert sig_ts == TS


def test_rebalance_keeps_incumbent_side():
    feed = (_sig(ActionType.long, ts=TS_PRIOR), _sig(ActionType.rebalance, ts=TS))
    action, score, _, sig_ts = _resolve_posture(feed, Posture(), TS)
    assert action == "long"  # a rebalance is not a reposition
    assert sig_ts == TS_PRIOR


def test_latest_wins_when_reopened_after_close():
    feed = (
        _sig(ActionType.long, ts=TS_PRIOR),
        _sig(ActionType.close, ts=TS_PRIOR),
        _sig(ActionType.short, ts=TS),  # newest sets the side
    )
    action, score, _, sig_ts = _resolve_posture(feed, Posture(), TS)
    assert action == "short"
    assert score == Posture().base_score_open
    assert sig_ts == TS  # reopened on newest bar


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
