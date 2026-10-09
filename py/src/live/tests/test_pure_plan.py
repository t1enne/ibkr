"""Pure tests for the write planner (``plan_result_writes`` / ``next_attempt``).

No sqlite: every assertion is about the VALUES a cycle's results imply — which
lot rows, which fills, and the cash signs on them. ``src/live/tests/test_ledger.py``
keeps only the db-level guarantees (atomicity, idempotence, prune).

Methodology: build results through the real ``OrderResult``/``OrderIntent``
shapes (never a mock), plan against in-memory lots, and assert on the write
types plus the fill fields the money math rides on.
"""

from __future__ import annotations

from typing import cast

import pandas as pd

from src.bt.state import ActionType, FillEvent
from src.live.identity import IntentKey, IntentRecord, IntentState
from src.live.pure import OrderResult, intent_to_signal
from src.live.pure_plan import (
    FillWrite,
    LotClose,
    LotOpen,
    next_attempt,
    plan_result_writes,
)
from src.live.types import OrderIntent, SimLot

TS = cast("pd.Timestamp", pd.Timestamp("2024-06-03T15:00:00Z"))
LATER = cast("pd.Timestamp", pd.Timestamp("2024-06-04T15:00:00Z"))


def _intent(
    action: ActionType = ActionType.long,
    *,
    symbol: str = "AAPL",
    pid: str | None = None,
    stop_loss: float | None = None,
    tag: str = "",
) -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        action=action,
        qty=10.0,
        ref_price=100.0,
        reason="test",
        position_id=pid,
        stop_loss=stop_loss,
        tag=tag,
    )


def _fill(
    *,
    qty: float = 10.0,
    price: float = 100.0,
    commission: float = 1.0,
    ts: pd.Timestamp = TS,
) -> FillEvent:
    return FillEvent(
        signal=intent_to_signal(_intent(), ts),
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


def _lot(
    pid: str = "L1",
    *,
    side: str = "long",
    qty: float = 10.0,
    entry: float = 100.0,
    opened_at: pd.Timestamp | None = TS,
    exit_price: float | None = None,
    closed_at: pd.Timestamp | None = None,
) -> SimLot:
    return SimLot(
        position_id=pid,
        symbol="AAPL",
        side=side,
        qty=qty,
        entry_price=entry,
        opened_at=opened_at,
        exit_price=exit_price,
        closed_at=closed_at,
    )


def test_all_writes_carry_the_scope() -> None:
    # The fill rows are keyed by (scope, execution_id); a wrong scope would split
    # one book across two, so the fold stamps the caller's scope on every write.
    writes = plan_result_writes("S1", (_open("L1"), _close("L1")), (), TS)
    fills = [w.record for w in writes if isinstance(w, FillWrite)]
    assert fills and {r.scope for r in fills} == {"S1"}


# --- nothing-to-write paths -------------------------------------------------


def test_failed_results_write_nothing() -> None:
    # A rejected / failed result is a VALUE the book must ignore: no lot, no fill.
    assert plan_result_writes("S1", (_open(ok=False),), (), TS) == ()
    assert plan_result_writes("S1", (_close(ok=False),), (_lot(),), TS) == ()


def test_unnamed_results_write_nothing() -> None:
    # A lot the broker never named can never be targeted by a close: not owned;
    # an unnamed close addresses no lot at all.
    assert plan_result_writes("S1", (_open(None),), (), TS) == ()
    assert plan_result_writes("S1", (_close(None),), (_lot(),), TS) == ()


def test_close_of_an_unknown_lot_writes_nothing() -> None:
    # The stored book has no such lot, so the close addresses no row.
    assert plan_result_writes("S1", (_close("GHOST"),), (_lot("L1"),), TS) == ()


# --- opens ------------------------------------------------------------------


def test_an_open_upserts_its_lot_and_its_entry_fill() -> None:
    (open_write, fill) = plan_result_writes("S1", (_open("L1"),), (), TS)
    assert isinstance(open_write, LotOpen)
    lot = open_write.lot
    assert (lot.position_id, lot.symbol, lot.side, lot.qty, lot.entry_price) == (
        "L1",
        "AAPL",
        "long",
        10.0,
        100.0,
    )
    assert isinstance(fill, FillWrite)
    assert (fill.record.execution_id, fill.record.side) == ("L1:open", "BUY")
    assert fill.record.cash_delta == -(10.0 * 100.0 + 1.0)  # a long entry DEBITS


def test_a_short_open_credits_cash_and_the_exit_is_the_mirror() -> None:
    short = _intent(ActionType.short)
    (open_write, entry) = plan_result_writes(
        "scope", (_open("L1", intent=short),), (), TS
    )
    assert isinstance(open_write, LotOpen)
    assert isinstance(entry, FillWrite)
    assert entry.record.side == "SELL"
    assert entry.record.cash_delta == 10.0 * 100.0 - 1.0  # a short entry CREDITS

    # A close re-derives the lot's legs from its POST-close state, so the exit is
    # the last write and the mirror of the entry that is already stored.
    closed = plan_result_writes(
        "scope",
        (_close("L1", fill=_fill(price=90.0)),),
        (_lot("L1", side="short"),),
        LATER,
    )
    assert isinstance(closed[-1], FillWrite)
    assert closed[-1].record.side == "BUY"
    assert closed[-1].record.cash_delta == -(10.0 * 90.0 + 1.0)  # cover DEBITS


def test_an_ownership_only_lot_implies_no_fill() -> None:
    # No size or entry to book — inventing one would put a phantom fill in history.
    bare = SimLot(position_id="BARE", closed_at=None)
    writes = plan_result_writes("S1", (_close("BARE", fill=_fill()),), (bare,), TS)
    assert writes == (LotClose("BARE", TS, exit_price=100.0, commission=1.0),)


def test_a_close_with_no_reported_exit_of_a_bare_lot_writes_only_the_close() -> None:
    # An ownership-only lot has no size or entry, so the close stamps the row and
    # writes NO fill rather than inventing a zero-qty leg.
    (close_write,) = plan_result_writes(
        "S1", (_close("BARE", fill=None),), (SimLot(position_id="BARE"),), LATER
    )
    assert close_write == LotClose("BARE", LATER, exit_price=None, commission=None)


def test_an_open_result_with_no_fill_records_ownership_without_a_fill() -> None:
    # An OK open that reported no fill still owns its lot (ownership is never lost
    # to missing detail) but must NOT mint a fill: there is no size or price to
    # book, and a phantom zero-qty leg would enter the scope's history.
    result = OrderResult(intent=_intent(), fill=None, ok=True, position_id="L1")
    (opened,) = plan_result_writes("S1", (result,), (), TS)
    assert isinstance(opened, LotOpen)
    assert opened.lot.position_id == "L1" and opened.lot.qty is None
    assert not opened.lot.has_detail


# --- closes -----------------------------------------------------------------


def test_a_close_stamps_the_exit_it_reported() -> None:
    writes = plan_result_writes(
        "S1",
        (_close("L1", fill=_fill(price=110.0, commission=2.0)),),
        (_lot("L1"),),
        LATER,
    )
    assert writes[0] == LotClose("L1", LATER, exit_price=110.0, commission=2.0)
    exit_fill = cast("FillWrite", writes[-1])
    assert exit_fill.record.execution_id == "L1:close"
    assert exit_fill.record.cash_delta == 10.0 * 110.0 - 2.0  # a long exit CREDITS


def test_a_close_never_erases_a_recorded_exit() -> None:
    # A close that does not say what it exited at leaves the exit COLUMNS alone:
    # the write carries no exit, so the UPDATE cannot null the stored one.
    already = _lot("L1", exit_price=120.0, closed_at=TS)
    writes = plan_result_writes("S1", (_close("L1", fill=None),), (already,), LATER)
    assert writes[0] == LotClose("L1", LATER, exit_price=None, commission=None)


# --- one batch, many results ------------------------------------------------


def test_open_then_close_in_one_batch_writes_an_open_and_an_exit() -> None:
    """The fold sees the open's lot, so the same cycle's close can target it.

    The exit price comes from the CLOSE's fill (there was no lot to inherit one
    from), and the exit fill is derived from the lot's POST-close state.
    """
    writes = plan_result_writes(
        "S1",
        (_open("L1"), _close("L1", fill=_fill(price=130.0))),
        (),
        TS,
    )
    kinds = [type(w).__name__ for w in writes]
    # The close re-derives the lot's legs from its POST-close state, so the entry
    # fill appears twice (an IGNORE insert, exactly as the read-after-write path
    # produced it). The last write is the exit fill.
    assert kinds == ["LotOpen", "FillWrite", "LotClose", "FillWrite", "FillWrite"]
    close_write, exit_fill = cast("LotClose", writes[2]), cast("FillWrite", writes[-1])
    assert close_write == LotClose("L1", TS, exit_price=130.0, commission=1.0)
    assert exit_fill.record.execution_id == "L1:close"
    assert exit_fill.record.cash_delta == 10.0 * 130.0 - 1.0


def test_two_closes_in_one_batch_stamp_the_later_watermark() -> None:
    # The unknown-id no-op cannot swallow a KNOWN lot's second close.
    writes = plan_result_writes(
        "S1",
        (_close("L1", fill=_fill(price=110.0)), _close("L1", fill=_fill(price=120.0))),
        (_lot("L1"),),
        LATER,
    )
    closes = [w for w in writes if isinstance(w, LotClose)]
    # The second close re-reports the same commission, so only its exit_price
    # moved it — a field the close does not change is never re-written.
    assert closes == [
        LotClose("L1", LATER, exit_price=110.0, commission=1.0),
        LotClose("L1", LATER, exit_price=120.0, commission=None),
    ]
    exits = [w for w in writes if isinstance(w, FillWrite)]
    assert [w.record.execution_id for w in exits] == [
        "L1:open",
        "L1:close",
        "L1:open",
        "L1:close",
    ]


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
