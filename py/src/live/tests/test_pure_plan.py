"""Pure tests for the write planner (``plan_result_writes`` / ``next_attempt``).

No sqlite: every assertion is about the VALUES a cycle's results imply — which
fill rows and the cash signs on them. ``src/live/tests/test_ledger.py`` keeps
only the db-level guarantees (atomicity, idempotence, prune).

Methodology: build results through the real ``OrderResult``/``OrderIntent``
shapes (never a mock), and assert on the ``ExecutionRecord`` fields the money
math rides on.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pandas as pd

from src.bt.state import ActionType, FillEvent
from src.live.identity import IntentKey, IntentRecord, IntentState
from src.live.pure import OrderResult, intent_to_signal
from src.live.pure_plan import (
    FillWrite,
    next_attempt,
    plan_result_writes,
)
from src.live.types import OrderIntent

TS = cast("pd.Timestamp", pd.Timestamp("2024-06-03T15:00:00Z"))


def _intent(
    action: ActionType = ActionType.long,
    *,
    symbol: str = "AAPL",
    pid: str | None = None,
) -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        action=action,
        qty=10.0,
        ref_price=100.0,
        reason="test",
        position_id=pid,
    )


def _fill(
    *,
    qty: float = 10.0,
    price: float = 100.0,
    commission: float = 1.0,
    ts: pd.Timestamp = TS,
    position_side: ActionType | None = None,
) -> FillEvent:
    signal = intent_to_signal(_intent(), ts)
    if position_side is not None:
        signal = replace(signal, position_side=position_side)
    return FillEvent(
        signal=signal,
        filled_qty=qty,
        executed_price=price,
        commission=commission,
        slippage=0.0,
        timestamp=ts,
    )


def _open(
    pid: str | None = "L1",
    *,
    ok: bool = True,
    fill: FillEvent | None = None,
    intent: OrderIntent | None = None,
) -> OrderResult:
    return OrderResult(
        intent=intent if intent is not None else _intent(),
        fill=_fill() if fill is None else fill,
        ok=ok,
        position_id=pid,
    )


def _close(
    pid: str | None = "L1",
    *,
    ok: bool = True,
    fill: FillEvent | None = None,
    intent: OrderIntent | None = None,
) -> OrderResult:
    return OrderResult(
        intent=intent if intent is not None else _intent(ActionType.close, pid=pid),
        fill=fill,
        ok=ok,
        position_id=None,
    )


# --- nothing-to-write paths -------------------------------------------------


def test_failed_results_write_nothing() -> None:
    assert plan_result_writes("S1", (_open(ok=False),), TS) == ()
    assert plan_result_writes("S1", (_close(ok=False),), TS) == ()


def test_a_fillless_result_writes_nothing() -> None:
    result = OrderResult(intent=_intent(), fill=None, ok=True, position_id="L1")
    assert plan_result_writes("S1", (result,), TS) == ()


def test_unnamed_results_write_nothing() -> None:
    assert plan_result_writes("S1", (_open(None),), TS) == ()
    assert plan_result_writes("S1", (_close(None),), TS) == ()


# --- opens ------------------------------------------------------------------


def test_an_open_emits_its_entry_fill() -> None:
    (write,) = plan_result_writes("S1", (_open("L1"),), TS)
    assert isinstance(write, FillWrite)
    record = write.record
    assert (record.execution_id, record.side, record.position_id, record.symbol) == (
        "L1:open",
        "BUY",
        "L1",
        "AAPL",
    )
    assert record.qty == 10.0 and record.price == 100.0
    assert record.cash_delta == -(10.0 * 100.0 + 1.0)  # a long entry DEBITS
    assert record.conid is None


def test_a_short_open_credits_cash() -> None:
    short = _intent(ActionType.short)
    (write,) = plan_result_writes("S1", (_open("L1", intent=short),), TS)
    assert write.record.side == "SELL"
    assert write.record.cash_delta == 10.0 * 100.0 - 1.0  # a short entry CREDITS


# --- closes -----------------------------------------------------------------


def test_a_long_close_emits_the_mirror_exit_fill() -> None:
    (write,) = plan_result_writes(
        "S1",
        (_close("L1", fill=_fill(price=110.0, position_side=ActionType.long)),),
        TS,
    )
    assert write.record.execution_id == "L1:close"
    assert write.record.side == "SELL"
    assert write.record.cash_delta == 10.0 * 110.0 - 1.0  # a long exit CREDITS


def test_a_short_close_emits_the_mirror_exit_fill() -> None:
    (write,) = plan_result_writes(
        "S1",
        (_close("L1", fill=_fill(price=90.0, position_side=ActionType.short)),),
        TS,
    )
    assert write.record.execution_id == "L1:close"
    assert write.record.side == "BUY"
    assert write.record.cash_delta == -(10.0 * 90.0 + 1.0)  # cover DEBITS


def test_all_writes_carry_the_scope() -> None:
    writes = plan_result_writes("S1", (_open("L1"), _close("L1")), TS)
    assert writes and {w.record.scope for w in writes} == {"S1"}


# --- next_attempt -----------------------------------------------------------


def _record(attempt: int) -> IntentRecord:
    return IntentRecord(
        key=IntentKey(
            scope="S1", symbol="AAPL", action=ActionType.long, position_id=None
        ),
        state=IntentState.WORKING,
        attempt=attempt,
        order_ref="r",
        order_id=None,
        decision_ts=None,
    )


def test_next_attempt_starts_at_zero_and_never_reuses_one() -> None:
    assert next_attempt(None) == 0
    assert next_attempt(_record(0)) == 1
    assert next_attempt(_record(7)) == 8
