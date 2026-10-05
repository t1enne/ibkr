"""Determinism tests for the shared order-ref scheme."""

from __future__ import annotations

import pandas as pd

from src.exec.refs import order_ref
from src.utils import parse_timestamp


def ts(value: str) -> pd.Timestamp:
    """``pd.Timestamp`` narrowed past its ``NaTType`` union (repo convention)."""
    return parse_timestamp(value)


TS = ts("2025-03-04 09:30")


def test_order_ref_format() -> None:
    assert order_ref("momentum_x", TS, 7) == "momentum-20250304T0930-007"


def test_order_ref_is_deterministic() -> None:
    assert order_ref("abcdefghij", TS, 0) == order_ref("abcdefghij", TS, 0)


def test_order_ref_truncates_strategy_id_to_eight() -> None:
    ref = order_ref("abcdefghij", TS, 0)
    assert ref.split("-")[0] == "abcdefgh"


def test_order_ref_sequence_is_disjoint() -> None:
    refs = {order_ref("strat", TS, seq) for seq in range(100)}
    assert len(refs) == 100


def test_order_ref_ignores_seconds_and_below() -> None:
    assert order_ref("strat", TS, 1) == order_ref("strat", ts("2025-03-04 09:30:59"), 1)
