"""Tests for the live cycle engine: freshness gate, run_cycle, build_report."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from src.bt.state import ActionType, FillEvent, PortfolioState, Position
from src.data.db import get_connection
from src.live.broker import OrderResult, intent_to_signal
from src.live.engine import (
    CycleReport,
    PortfolioFetchError,
    StaleDataError,
    assert_data_fresh,
    build_report,
    run_cycle,
)
from src.live.ledger import PositionRecord, SqliteLedger
from src.live.result import Err, Ok
from src.live.types import (
    FeedError,
    LiveConfig,
    LiveSignal,
    OrderIntent,
    PortfolioSnapshot,
    SignalAction,
)

TS = cast("pd.Timestamp", pd.Timestamp("2024-06-03"))
OLD = cast("pd.Timestamp", pd.Timestamp("2020-01-01"))

FetchResult = Ok[PortfolioSnapshot, FeedError] | Err[PortfolioSnapshot, FeedError]
PlaceResult = Ok[OrderResult, FeedError] | Err[OrderResult, FeedError]

CFG = LiveConfig(
    strategy_type="momentum",
    symbols=("AAPL",),
    initial_capital=100_000.0,
    strategy_params={},
    bars=("1d",),
    warmup="1y",
)


def book(*lots: Position) -> PortfolioState:
    grouped: dict[str, list[Position]] = {}
    for pos in lots:
        grouped.setdefault(pos.symbol, []).append(pos)
    return PortfolioState(
        cash=100_000.0,
        positions={sym: tuple(v) for sym, v in grouped.items()},
        trades=(),
        equity_curve=(),
        initial_capital=100_000.0,
    )


def lot(pid: str) -> Position:
    return Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=TS,
        stop_loss=None,
        take_profit=None,
        last_price=100.0,
        type=ActionType.long,
        position_id=pid,
    )


def signal(action: SignalAction, qty: float = 0.0) -> LiveSignal:
    return LiveSignal(
        symbol="AAPL",
        action=action,
        score=1.0,
        reasons=(),
        signal_ts=TS,
        price=100.0,
        qty=qty,
    )


class FakeSource:
    owns_book = True

    def __init__(self, portfolio: PortfolioState) -> None:
        self._portfolio = portfolio

    async def fetch(self) -> FetchResult:
        return Ok(PortfolioSnapshot(self._portfolio, TS))


class FakeBroker:
    def __init__(self, reject: bool = False, open_pid: str | None = "L1") -> None:
        self.seeded: PortfolioState | None = None
        self.placed: list[OrderIntent] = []
        self._reject = reject
        self._open_pid = open_pid

    def seed(self, portfolio: PortfolioState) -> None:
        self.seeded = portfolio

    async def place(self, intent: OrderIntent) -> PlaceResult:
        self.placed.append(intent)
        if self._reject:
            return Ok(OrderResult(intent=intent, fill=None, ok=False, message="no"))
        fill = FillEvent(
            signal=intent_to_signal(intent, TS),
            filled_qty=intent.qty,
            executed_price=intent.ref_price,
            commission=0.0,
            slippage=0.0,
            timestamp=TS,
        )
        pid = (
            intent.position_id if intent.action is ActionType.close else self._open_pid
        )
        return Ok(OrderResult(intent=intent, fill=fill, ok=True, position_id=pid))

    async def place_cohort(
        self, intents: tuple[OrderIntent, ...]
    ) -> (
        Ok[tuple[OrderResult, ...], FeedError] | Err[tuple[OrderResult, ...], FeedError]
    ):
        results: list[OrderResult] = []
        for intent in intents:
            placed = await self.place(intent)
            assert isinstance(placed, Ok)
            results.append(cast("OrderResult", placed.value))
        return Ok(tuple(results))

    async def close(self) -> Ok[None, FeedError]:
        return Ok(None)


def make_candle_db(path: Path, ticker: str | None, ts: pd.Timestamp | None) -> None:
    """A minimal candle table with one row (``None`` -> an empty universe DB)."""
    con = get_connection(path)
    con.execute("CREATE TABLE candle (ticker TEXT, timestamp INTEGER)")
    if ticker is not None and ts is not None:
        con.execute(
            "INSERT INTO candle VALUES (?,?)", (ticker, int(ts.timestamp() * 1000))
        )
    con.commit()
    con.close()


def ledger_rows(path: Path) -> list[tuple[object, ...]]:
    """Every ``live_position`` row (id, status, closed_at), open AND closed."""
    con = get_connection(path)
    try:
        return con.execute(
            "SELECT position_id, status, closed_at FROM live_position"
        ).fetchall()
    finally:
        con.close()


def cycle_ts(path: Path, strategy_id: str) -> object:
    """The strategy row's ``last_cycle_at`` (None if never touched)."""
    con = get_connection(path)
    try:
        row = con.execute(
            "SELECT last_cycle_at FROM live_strategy WHERE strategy_id=?",
            (strategy_id,),
        ).fetchone()
    finally:
        con.close()
    return row[0] if row else None


def _source_fn(acts: tuple[tuple[SignalAction, float], ...]):
    def _f(path: str, max_age: int | None = None) -> tuple[LiveSignal, ...]:
        return tuple(signal(a, q) for a, q in acts)

    return _f


LONG_10: tuple[tuple[SignalAction, float], ...] = (("long", 10.0),)
CLOSE: tuple[tuple[SignalAction, float], ...] = (("close", 0.0),)


# --- freshness gate ---------------------------------------------------------


def test_assert_data_fresh_passes_fresh(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    assert_data_fresh(("AAPL",), 5, TS, db)  # age 0, no raise


def test_assert_data_fresh_raises_stale(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", OLD)
    with pytest.raises(StaleDataError, match="is .* old"):
        assert_data_fresh(("AAPL",), 5, TS, db)


def test_assert_data_fresh_raises_empty_universe_db(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    make_candle_db(db, None, None)
    with pytest.raises(StaleDataError, match="no data for universe"):
        assert_data_fresh(("AAPL",), 5, TS, db)


def test_assert_data_fresh_disabled_with_zero(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    make_candle_db(db, None, None)
    assert_data_fresh(("AAPL",), 0, TS, db)  # gate off, no raise


def test_assert_data_fresh_uppercases_symbol(tmp_path: Path) -> None:
    # candle.ticker is UPPERCASE; a lowercase config symbol must still match
    # (else a fresh feed would report a bogus "no data for universe").
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    assert_data_fresh(("aapl",), 5, TS, db)


def test_assert_data_fresh_accepts_tz_aware_now(tmp_path: Path) -> None:
    # now may be tz-aware UTC; the age is still 0 against a fresh ms-epoch bar.
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    now_utc = TS.tz_localize("UTC")
    assert_data_fresh(("AAPL",), 5, now_utc, db)


# --- run_cycle --------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_cycle_opens_and_records(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger = SqliteLedger(tmp_path / "l.sqlite")
    source = FakeSource(book())
    broker = FakeBroker()

    report = await run_cycle(
        CFG,
        source=source,
        broker=broker,
        ledger=ledger,
        strategy_id="S1",
        config_path="x.json",
        now=TS,
        db_path=db,
        signal_source=_source_fn(LONG_10),
    )

    assert isinstance(report, CycleReport)
    assert [i.action for i in report.intents] == [ActionType.long]
    assert len(report.results) == len(report.intents) == 1
    assert report.portfolio_before == book()
    assert broker.seeded == book()  # book aligned with the fetched read
    assert [r.position_id for r in ledger.open_positions("S1")] == ["L1"]


@pytest.mark.asyncio
async def test_run_cycle_close_marks_closed(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger = SqliteLedger(tmp_path / "l.sqlite")
    ledger.record_open(_rec("L1"))
    broker = FakeBroker()

    report = await run_cycle(
        CFG,
        source=FakeSource(book(lot("L1"))),
        broker=broker,
        ledger=ledger,
        strategy_id="S1",
        config_path="x.json",
        now=TS,
        db_path=db,
        signal_source=_source_fn(CLOSE),
    )

    assert [i.action for i in report.intents] == [ActionType.close]
    assert ledger.open_positions("S1") == ()
    # Mark-closed, NOT deleted: the row survives with status + closed_at.
    rows = ledger_rows(tmp_path / "l.sqlite")
    assert len(rows) == 1
    pid, status, closed_at = rows[0]
    assert pid == "L1"
    assert status == "closed"
    assert closed_at is not None


@pytest.mark.asyncio
async def test_unnamed_open_records_nothing(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger = SqliteLedger(tmp_path / "l.sqlite")

    report = await run_cycle(
        CFG,
        source=FakeSource(book()),
        broker=FakeBroker(open_pid=None),  # ok=True but no broker lot id
        ledger=ledger,
        strategy_id="S1",
        config_path="x.json",
        now=TS,
        db_path=db,
        signal_source=_source_fn(LONG_10),
    )

    assert len(report.results) == 1 and report.results[0].ok
    assert ledger_rows(tmp_path / "l.sqlite") == []  # unclosable lot NOT recorded
    assert ledger.open_positions("S1") == ()


@pytest.mark.asyncio
async def test_rejected_close_records_nothing(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger = SqliteLedger(tmp_path / "l.sqlite")
    ledger.record_open(_rec("L1"))

    await run_cycle(
        CFG,
        source=FakeSource(book(lot("L1"))),
        broker=FakeBroker(reject=True),
        ledger=ledger,
        strategy_id="S1",
        config_path="x.json",
        now=TS,
        db_path=db,
        signal_source=_source_fn(CLOSE),
    )

    assert [r.position_id for r in ledger.open_positions("S1")] == ["L1"]


@pytest.mark.asyncio
async def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger_path = tmp_path / "l.sqlite"
    ledger = SqliteLedger(ledger_path)
    ledger.ensure_strategy("S1", "s", "paper")
    broker = FakeBroker()

    report = await run_cycle(
        CFG,
        source=FakeSource(book()),
        broker=broker,
        ledger=ledger,
        strategy_id="S1",
        config_path="x.json",
        now=TS,
        db_path=db,
        dry_run=True,
        signal_source=_source_fn(LONG_10),
    )

    assert [i.action for i in report.intents] == [ActionType.long]  # still computes
    assert report.results == ()
    assert broker.placed == []  # nothing placed
    assert ledger_rows(ledger_path) == []  # nothing recorded
    assert cycle_ts(ledger_path, "S1") is None  # didn't even touch the cycle


@pytest.mark.asyncio
async def test_foreign_lot_survives_cycle_untouched(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger_path = tmp_path / "l.sqlite"
    ledger = SqliteLedger(ledger_path)
    ledger.record_open(_rec("L1"))  # ours; the book's OTHER lot is foreign
    foreign = Position(
        symbol="AAPL",
        qty=7.0,
        entry_price=100.0,
        entry_time=TS,
        stop_loss=None,
        take_profit=None,
        last_price=100.0,
        type=ActionType.long,
        position_id="OTHER",
    )
    broker = FakeBroker()

    report = await run_cycle(
        CFG,
        source=FakeSource(book(foreign)),
        broker=broker,
        ledger=ledger,
        strategy_id="S1",
        config_path="x.json",
        now=TS,
        db_path=db,
        signal_source=_source_fn(CLOSE),
    )

    # Close is ownership-scoped: the foreign lot is neither closed nor adopted.
    assert report.intents == ()
    assert broker.placed == []
    assert report.portfolio_before.positions["AAPL"] == (foreign,)
    assert [r.position_id for r in ledger.open_positions("S1")] == ["L1"]


@pytest.mark.asyncio
async def test_changed_config_rescopes_ownership(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger = SqliteLedger(tmp_path / "l.sqlite")
    ledger.record_open(_rec("L1"))  # lot opened under config revision A ("S1")
    broker = FakeBroker()

    # Revision B ("S2") sees the same book, but L1 is foreign to it: no close.
    report = await run_cycle(
        CFG,
        source=FakeSource(book(lot("L1"))),
        broker=broker,
        ledger=ledger,
        strategy_id="S2",
        config_path="x.json",
        now=TS,
        db_path=db,
        signal_source=_source_fn(CLOSE),
    )

    assert report.intents == ()
    assert broker.placed == []
    assert [r.position_id for r in ledger.open_positions("S1")] == ["L1"]


@pytest.mark.asyncio
async def test_self_owned_source_closes_replayed_lot_with_empty_ledger(
    tmp_path: Path,
) -> None:
    """S2: an ibkr-style book is already ours — closes must not need the ledger.

    The IBKR execution replay excludes foreign orders, so its book carries only
    our lots (``owns_book = False``). With an EMPTY ledger (a dry run writes
    nothing) a close signal must still yield a close intent for the replayed lot.
    """
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger = SqliteLedger(tmp_path / "l.sqlite")  # no rows recorded

    class SelfOwnedSource(FakeSource):
        owns_book = False

    report = await run_cycle(
        CFG,
        source=SelfOwnedSource(book(lot("97932"))),
        broker=FakeBroker(),
        ledger=ledger,
        strategy_id="S1",
        config_path="x.json",
        now=TS,
        db_path=db,
        signal_source=_source_fn(CLOSE),
    )

    assert len(report.intents) == 1
    assert report.intents[0].action is ActionType.close
    assert report.intents[0].position_id == "97932"


@pytest.mark.asyncio
async def test_run_cycle_stale_data_raises(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", OLD)
    with pytest.raises(StaleDataError):
        await run_cycle(
            CFG,
            source=FakeSource(book()),
            broker=FakeBroker(),
            ledger=SqliteLedger(tmp_path / "l.sqlite"),
            strategy_id="S1",
            config_path="x.json",
            now=TS,
            db_path=db,
            signal_source=_source_fn(LONG_10),
        )


@pytest.mark.asyncio
async def test_run_cycle_fetch_error_raises(tmp_path: Path) -> None:
    class DeadSource:
        owns_book = True

        async def fetch(self) -> FetchResult:
            return Err(FeedError(kind="transport", message="down"))

    with pytest.raises(PortfolioFetchError):
        await run_cycle(
            CFG,
            source=DeadSource(),
            broker=FakeBroker(),
            ledger=SqliteLedger(tmp_path / "l.sqlite"),
            strategy_id="S1",
            config_path="x.json",
            now=TS,
            signal_source=_source_fn(()),
        )


def test_build_report_is_pure() -> None:
    args = (book(), (signal("long", 10.0),), (), ())
    first = build_report(*args, as_of=TS)
    second = build_report(*args, as_of=TS)
    assert first == second
    assert first.as_of == TS
    assert first.portfolio_before == book()


def _rec(pid: str) -> PositionRecord:
    return PositionRecord(
        strategy_id="S1",
        position_id=pid,
        symbol="AAPL",
        side="long",
        qty=10.0,
        entry_price=100.0,
        entry_time=TS,
        stop_loss=None,
        take_profit=None,
        tag="",
        status="open",
        opened_at=TS,
    )
