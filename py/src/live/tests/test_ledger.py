"""Tests for the SQLite strategy→lot ledger (record/mark-close/prune)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from src.bt.state import ActionType, Position
from src.data.db import get_connection
from src.live.ledger import (
    PositionRecord,
    PositionStatus,
    SqliteLedger,
    config_hash,
    from_position,
)

TS = cast("pd.Timestamp", pd.Timestamp("2024-06-03"))
OLD = cast("pd.Timestamp", pd.Timestamp("2020-01-01"))


@pytest.fixture
def ledger(tmp_path: Path) -> SqliteLedger:
    return SqliteLedger(tmp_path / "ledger.sqlite")


def rec(
    pid: str,
    *,
    strategy_id: str = "S1",
    symbol: str = "AAPL",
    status: PositionStatus = "open",
    opened_at: pd.Timestamp = TS,
) -> PositionRecord:
    return PositionRecord(
        strategy_id=strategy_id,
        position_id=pid,
        symbol=symbol,
        side="long",
        qty=10.0,
        entry_price=100.0,
        entry_time=TS,
        stop_loss=None,
        take_profit=None,
        tag="",
        status=status,
        opened_at=opened_at,
    )


def test_config_hash_order_insensitive_over_keys() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_config_hash_differs_on_value_change() -> None:
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_config_hash_differs_on_nesting_change() -> None:
    assert config_hash({"a": {"x": 1}}) != config_hash({"a": {"x": 2}})


def test_ensure_strategy_idempotent(ledger: SqliteLedger, tmp_path: Path) -> None:
    ledger.ensure_strategy("S1", "momentum", "paper")
    ledger.ensure_strategy("S1", "momentum", "paper")
    with get_connection(tmp_path / "ledger.sqlite") as con:
        (count,) = con.execute("SELECT COUNT(*) FROM live_strategy").fetchone()
    assert count == 1


def test_record_open_then_open_positions(ledger: SqliteLedger) -> None:
    ledger.record_open(rec("L1"))
    (got,) = ledger.open_positions("S1")
    assert got.position_id == "L1"
    assert got.status == "open"
    assert got.qty == 10.0
    assert got.opened_at == TS


def test_mark_closed_retains_row_but_excludes_from_open(
    ledger: SqliteLedger, tmp_path: Path
) -> None:
    ledger.record_open(rec("L1"))
    ledger.mark_closed("S1", "L1", TS)

    assert ledger.open_positions("S1") == ()

    with get_connection(tmp_path / "ledger.sqlite") as con:
        row = con.execute(
            "SELECT status, closed_at FROM live_position "
            "WHERE strategy_id='S1' AND position_id='L1'"
        ).fetchone()
    assert row is not None
    status, closed_at = row
    assert status == "closed"
    assert closed_at is not None


def test_mark_closed_unknown_id_is_noop(ledger: SqliteLedger) -> None:
    ledger.record_open(rec("L1"))
    ledger.mark_closed("S1", "NOPE", TS)  # never raises
    assert [r.position_id for r in ledger.open_positions("S1")] == ["L1"]


def test_prune_closed_deletes_only_old_closed(
    ledger: SqliteLedger, tmp_path: Path
) -> None:
    ledger.record_open(rec("OLD", opened_at=OLD))
    ledger.record_open(rec("NEW", opened_at=TS))
    ledger.record_open(rec("OPEN", opened_at=OLD))
    ledger.mark_closed("S1", "OLD", OLD)
    ledger.mark_closed("S1", "NEW", TS)

    deleted = ledger.prune_closed(TS)  # cutoff between OLD and NEW
    assert deleted == 1

    assert _all_ids(tmp_path) == {"NEW", "OPEN"}  # open row survives


def _all_ids(tmp_path: Path) -> set[str]:
    with get_connection(tmp_path / "ledger.sqlite") as con:
        rows = con.execute("SELECT position_id FROM live_position").fetchall()
    return {r[0] for r in rows}


def test_unknown_strategy_is_empty(ledger: SqliteLedger) -> None:
    ledger.record_open(rec("L1"))
    assert ledger.open_positions("OTHER") == ()


def test_two_strategies_do_not_leak(ledger: SqliteLedger) -> None:
    ledger.record_open(rec("A", strategy_id="S1", symbol="AAPL"))
    ledger.record_open(rec("B", strategy_id="S2", symbol="MSFT"))
    assert [r.position_id for r in ledger.open_positions("S1")] == ["A"]
    assert [r.position_id for r in ledger.open_positions("S2")] == ["B"]


def test_touch_cycle_sets_last_cycle_at(ledger: SqliteLedger, tmp_path: Path) -> None:
    ledger.ensure_strategy("S1", "momentum", "paper")
    ledger.touch_cycle("S1", TS)
    with get_connection(tmp_path / "ledger.sqlite") as con:
        (last,) = con.execute(
            "SELECT last_cycle_at FROM live_strategy WHERE strategy_id='S1'"
        ).fetchone()
    assert last == int(TS.timestamp() * 1000)


def test_from_position_lifts_position_fields() -> None:
    pos = Position(
        symbol="MSFT",
        qty=3.0,
        entry_price=50.0,
        entry_time=TS,
        stop_loss=45.0,
        take_profit=60.0,
        last_price=50.0,
        type=ActionType.short,
        position_id="L9",
        tag="t1",
    )
    got = from_position(rec("ignored"), pos)
    assert got.position_id == "L9"
    assert got.symbol == "MSFT"
    assert got.side == "short"
    assert got.qty == 3.0
    assert got.stop_loss == 45.0
    assert got.take_profit == 60.0
    assert got.tag == "t1"
