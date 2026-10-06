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
from src.live.ledger import SqliteLedger
from src.live.lease import CycleInProgressError
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
    owns_book = False

    def __init__(self, portfolio: PortfolioState) -> None:
        self._portfolio = portfolio

    async def fetch(self) -> FetchResult:
        return Ok(PortfolioSnapshot(self._portfolio, TS))


class FakeBroker:
    def __init__(self, reject: bool = False, open_pid: str | None = "L1") -> None:
        self.seeded: PortfolioState | None = None
        self.placed: list[OrderIntent] = []
        self.resynced = 0
        self.events: list[str] = []
        self._reject = reject
        self._open_pid = open_pid

    def seed(self, portfolio: PortfolioState) -> None:
        self.seeded = portfolio

    async def resync(self) -> Ok[tuple[OrderResult, ...], FeedError]:
        self.resynced += 1
        self.events.append("resync")
        return Ok(())

    async def place(self, intent: OrderIntent) -> PlaceResult:
        self.events.append("place")
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


class ErrCohortBroker(FakeBroker):
    """A broker whose cohort placement fails at the cohort level (a port-level Err)."""

    async def place_cohort(
        self, intents: tuple[OrderIntent, ...]
    ) -> Err[tuple[OrderResult, ...], FeedError]:
        return Err(FeedError(kind="transport", message="cohort refused"))


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


def book_rows(path: Path) -> list[tuple[object, ...]]:
    """Every ``live_position`` row for the scope (empty if the table is unwritten)."""
    con = get_connection(path)
    try:
        return con.execute("SELECT conid, closed_at FROM live_position").fetchall()
    except Exception:
        return []
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
    assert_data_fresh(("AAPL",), 5, TS, db)


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
    assert_data_fresh(("AAPL",), 0, TS, db)


def test_assert_data_fresh_uppercases_symbol(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    assert_data_fresh(("aapl",), 5, TS, db)


def test_assert_data_fresh_accepts_tz_aware_now(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    assert_data_fresh(("AAPL",), 5, TS.tz_localize("UTC"), db)


# --- run_cycle --------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_cycle_places_open_and_records_nothing_in_the_book(
    tmp_path: Path,
) -> None:
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger_path = tmp_path / "l.sqlite"
    ledger = SqliteLedger(ledger_path)
    ledger.ensure_strategy("S1", "aapl", "momentum", "paper")

    report = await run_cycle(
        CFG,
        source=FakeSource(book()),
        broker=FakeBroker(),
        ledger=ledger,
        strategy_id="S1",
        scope="S1",
        config_path="x.json",
        now=TS,
        db_path=db,
        signal_source=_source_fn(LONG_10),
    )

    assert isinstance(report, CycleReport)
    assert [i.action for i in report.intents] == [ActionType.long]
    assert len(report.results) == len(report.intents) == 1
    assert report.portfolio_before == book()
    # The book advances from the broker's executions, NOT the placement result.
    assert book_rows(ledger_path) == []
    assert cycle_ts(ledger_path, "S1") is not None  # cycle touched


@pytest.mark.asyncio
async def test_run_cycle_close_yields_a_close_intent(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger = SqliteLedger(tmp_path / "l.sqlite")

    report = await run_cycle(
        CFG,
        source=FakeSource(book(lot("L1"))),
        broker=FakeBroker(),
        ledger=ledger,
        strategy_id="S1",
        scope="S1",
        config_path="x.json",
        now=TS,
        db_path=db,
        signal_source=_source_fn(CLOSE),
    )

    assert [i.action for i in report.intents] == [ActionType.close]
    assert report.intents[0].position_id == "L1"


class SimSource(FakeSource):
    """A mock-fixture source: its book may hold lots we never opened."""

    owns_book = True


@pytest.mark.asyncio
async def test_sim_source_does_not_close_a_foreign_fixture_lot(tmp_path: Path) -> None:
    # Ownership scoping (sim path): a lot the strategy never opened is not ours.
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger = SqliteLedger(tmp_path / "l.sqlite")  # no lots recorded

    report = await run_cycle(
        CFG,
        source=SimSource(book(lot("L1"))),
        broker=FakeBroker(),
        ledger=ledger,
        strategy_id="S1",
        scope="S1",
        config_path="x.json",
        now=TS,
        db_path=db,
        signal_source=_source_fn(CLOSE),
    )

    assert report.intents == ()


@pytest.mark.asyncio
async def test_sim_source_closes_only_its_ledger_owned_lot(tmp_path: Path) -> None:
    # The strategy opened L1 (recorded); OTHER is a foreign fixture lot. A bare
    # close targets only the lot we own.
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger = SqliteLedger(tmp_path / "l.sqlite")
    ledger.record_sim_open("S1", "L1")

    report = await run_cycle(
        CFG,
        source=SimSource(book(lot("L1"), lot("OTHER"))),
        broker=FakeBroker(),
        ledger=ledger,
        strategy_id="S1",
        scope="S1",
        config_path="x.json",
        now=TS,
        db_path=db,
        signal_source=_source_fn(CLOSE),
    )

    assert [i.position_id for i in report.intents] == ["L1"]


@pytest.mark.asyncio
async def test_sim_cycle_records_an_opened_lot_as_owned(tmp_path: Path) -> None:
    # A confirmed sim open is recorded as owned so a later cycle may close it.
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger = SqliteLedger(tmp_path / "l.sqlite")

    await run_cycle(
        CFG,
        source=SimSource(book()),
        broker=FakeBroker(open_pid="AAPL_1"),
        ledger=ledger,
        strategy_id="S1",
        scope="S1",
        config_path="x.json",
        now=TS,
        db_path=db,
        signal_source=_source_fn(LONG_10),
    )

    assert ledger.sim_open_ids("S1") == frozenset({"AAPL_1"})


@pytest.mark.asyncio
async def test_sim_ownership_survives_a_config_hash_change(tmp_path: Path) -> None:
    # A parameter edit changes the config-hash strategy_id, but ownership is the
    # stable scope: the earlier lot stays closable (never HOLD-forever).
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger = SqliteLedger(tmp_path / "l.sqlite")
    ledger.record_sim_open("momentum", "L1")

    report = await run_cycle(
        CFG,
        source=SimSource(book(lot("L1"))),
        broker=FakeBroker(),
        ledger=ledger,
        strategy_id="hash-after-edit",
        scope="momentum",
        config_path="x.json",
        now=TS,
        db_path=db,
        signal_source=_source_fn(CLOSE),
    )

    assert [i.position_id for i in report.intents] == ["L1"]


@pytest.mark.asyncio
async def test_run_cycle_refuses_when_a_cycle_lease_is_held(tmp_path: Path) -> None:
    # A cron overlap / racing human run must refuse to start, not both place.
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger = SqliteLedger(tmp_path / "l.sqlite")

    with ledger.cycle_lease():
        with pytest.raises(CycleInProgressError):
            await run_cycle(
                CFG,
                source=FakeSource(book()),
                broker=FakeBroker(),
                ledger=ledger,
                strategy_id="S1",
                scope="S1",
                config_path="x.json",
                now=TS,
                db_path=db,
                signal_source=_source_fn(LONG_10),
            )


@pytest.mark.asyncio
async def test_dry_run_takes_no_lease(tmp_path: Path) -> None:
    # A read-only run must never be blocked by a live cycle holding the lease.
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger = SqliteLedger(tmp_path / "l.sqlite")

    with ledger.cycle_lease():
        report = await run_cycle(
            CFG,
            source=FakeSource(book()),
            broker=FakeBroker(),
            ledger=ledger,
            strategy_id="S1",
            scope="S1",
            config_path="x.json",
            now=TS,
            db_path=db,
            dry_run=True,
            signal_source=_source_fn(LONG_10),
        )

    assert [i.action for i in report.intents] == [ActionType.long]


@pytest.mark.asyncio
async def test_sim_unnamed_open_records_nothing(tmp_path: Path) -> None:
    # A lot the broker did not name can never be targeted by a close: not owned.
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger = SqliteLedger(tmp_path / "l.sqlite")

    report = await run_cycle(
        CFG,
        source=SimSource(book()),
        broker=FakeBroker(open_pid=None),
        ledger=ledger,
        strategy_id="S1",
        scope="S1",
        config_path="x.json",
        now=TS,
        db_path=db,
        signal_source=_source_fn(LONG_10),
    )

    assert report.results[0].ok
    assert ledger.sim_open_ids("S1") == frozenset()


@pytest.mark.asyncio
async def test_sim_rejected_close_records_nothing(tmp_path: Path) -> None:
    # A rejected close records NOTHING: the lot stays owned for a later cycle.
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger = SqliteLedger(tmp_path / "l.sqlite")
    ledger.record_sim_open("S1", "L1")

    await run_cycle(
        CFG,
        source=SimSource(book(lot("L1"))),
        broker=FakeBroker(reject=True),
        ledger=ledger,
        strategy_id="S1",
        scope="S1",
        config_path="x.json",
        now=TS,
        db_path=db,
        signal_source=_source_fn(CLOSE),
    )

    assert ledger.sim_open_ids("S1") == frozenset({"L1"})


@pytest.mark.asyncio
async def test_self_owned_source_closes_replayed_lot_with_empty_ledger(
    tmp_path: Path,
) -> None:
    """An ibkr-style book is already ours: closes need no ledger (plan rev 4.1 §3)."""
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger = SqliteLedger(tmp_path / "l.sqlite")  # a dry run writes nothing

    report = await run_cycle(
        CFG,
        source=FakeSource(book(lot("97932"))),  # owns_book = False
        broker=FakeBroker(),
        ledger=ledger,
        strategy_id="S1",
        scope="S1",
        config_path="x.json",
        now=TS,
        db_path=db,
        signal_source=_source_fn(CLOSE),
    )

    assert [i.action for i in report.intents] == [ActionType.close]
    assert report.intents[0].position_id == "97932"


@pytest.mark.asyncio
async def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger_path = tmp_path / "l.sqlite"
    ledger = SqliteLedger(ledger_path)
    broker = FakeBroker()

    report = await run_cycle(
        CFG,
        source=FakeSource(book()),
        broker=broker,
        ledger=ledger,
        strategy_id="S1",
        scope="S1",
        config_path="x.json",
        now=TS,
        db_path=db,
        dry_run=True,
        signal_source=_source_fn(LONG_10),
    )

    assert [i.action for i in report.intents] == [ActionType.long]
    assert report.results == ()
    assert broker.placed == []
    with get_connection(ledger_path) as con:
        tables = {
            r[0]
            for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert tables == set()  # no DDL written on a dry run


@pytest.mark.asyncio
async def test_dry_run_writes_no_peewee_live_tables(tmp_path: Path) -> None:
    """peewee lazy DDL: a dry-run cycle leaves the live models absent from
    ``sqlite_master`` (create_tables never runs on a read-only path)."""
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger_path = tmp_path / "l.sqlite"
    ledger = SqliteLedger(ledger_path)
    broker = FakeBroker()

    report = await run_cycle(
        CFG,
        source=FakeSource(book()),
        broker=broker,
        ledger=ledger,
        strategy_id="S1",
        scope="S1",
        config_path="x.json",
        now=TS,
        db_path=db,
        dry_run=True,
        signal_source=_source_fn(LONG_10),
    )

    assert report.results == () and broker.placed == []
    with get_connection(ledger_path) as con:
        tables = {
            r[0]
            for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    live_tables = {
        "live_strategy",
        "live_position",
        "live_execution",
        "live_cash",
        "live_sim_lot",
    }
    assert tables.isdisjoint(live_tables)  # peewee DDL withheld on a dry run


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
            scope="S1",
            config_path="x.json",
            now=TS,
            db_path=db,
            signal_source=_source_fn(LONG_10),
        )


@pytest.mark.asyncio
async def test_run_cycle_fetch_error_raises(tmp_path: Path) -> None:
    class DeadSource:
        owns_book = False

        async def fetch(self) -> FetchResult:
            return Err(FeedError(kind="transport", message="down"))

    with pytest.raises(PortfolioFetchError):
        await run_cycle(
            CFG,
            source=DeadSource(),
            broker=FakeBroker(),
            ledger=SqliteLedger(tmp_path / "l.sqlite"),
            strategy_id="S1",
            scope="S1",
            config_path="x.json",
            now=TS,
            signal_source=_source_fn(()),
        )


@pytest.mark.asyncio
async def test_run_cycle_cohort_error_is_surfaced_not_silently_empty(
    tmp_path: Path,
) -> None:
    """A cohort-level ``Err`` must reach the report, not read as "0 orders" (M1)."""
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger = SqliteLedger(tmp_path / "l.sqlite")

    report = await run_cycle(
        CFG,
        source=FakeSource(book()),
        broker=ErrCohortBroker(),
        ledger=ledger,
        strategy_id="S1",
        scope="S1",
        config_path="x.json",
        now=TS,
        db_path=db,
        signal_source=_source_fn(LONG_10),
    )

    assert report.results == ()
    assert report.placement_error is not None
    assert report.placement_error.kind == "transport"
    assert "cohort refused" in report.placement_error.message


@pytest.mark.asyncio
async def test_run_cycle_resyncs_before_placing(tmp_path: Path) -> None:
    # Cycle start must reconcile OPEN intents (resync) BEFORE signals/reconcile/
    # placement, so a prior cycle's working order is adopted, never re-minted.
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger = SqliteLedger(tmp_path / "l.sqlite")
    broker = FakeBroker()

    await run_cycle(
        CFG,
        source=FakeSource(book()),
        broker=broker,
        ledger=ledger,
        strategy_id="S1",
        scope="S1",
        config_path="x.json",
        now=TS,
        db_path=db,
        signal_source=_source_fn(LONG_10),
    )

    assert broker.resynced == 1
    assert broker.events[0] == "resync"  # before any placement
    assert "place" in broker.events


@pytest.mark.asyncio
async def test_dry_run_never_resyncs(tmp_path: Path) -> None:
    # resync persists state, which a read-only run must not do.
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger = SqliteLedger(tmp_path / "l.sqlite")
    broker = FakeBroker()

    await run_cycle(
        CFG,
        source=FakeSource(book()),
        broker=broker,
        ledger=ledger,
        strategy_id="S1",
        scope="S1",
        config_path="x.json",
        now=TS,
        db_path=db,
        dry_run=True,
        signal_source=_source_fn(LONG_10),
    )

    assert broker.resynced == 0


def test_build_report_is_pure() -> None:
    args = (book(), (signal("long", 10.0),), (), ())
    first = build_report(*args, as_of=TS)
    second = build_report(*args, as_of=TS)
    assert first == second
    assert first.as_of == TS
    assert first.portfolio_before == book()


def test_build_report_carries_a_placement_error() -> None:
    error = FeedError(kind="transport", message="cohort refused")
    report = build_report(book(), (), (), (), as_of=TS, placement_error=error)
    assert report.placement_error == error
    assert report.results == ()


pytestmark = pytest.mark.db
