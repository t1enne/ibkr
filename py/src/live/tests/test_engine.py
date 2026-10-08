"""Tests for the live cycle engine: freshness gate, run_cycle."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import pytest


from src.bt.state import ActionType, FillEvent, PortfolioState, Position
from src.data.db import get_connection
from src.live.pure import OrderResult, intent_to_signal
from src.live.engine import (
    run_cycle,
)
from src.live.adapters.sim.adapter import SimAdapter, build_sim_adapter
from src.live.ledger import SimLot, SqliteLedger
from src.live.lease import CycleInProgressError
from src.live.result import Err, Ok, Result
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


class FakeAdapter:
    """A ``LiveAdapter`` double: scripted book, recorded placements, no real edge.

    ``owns_book=False`` models the IBKR replay (the book is already only our
    lots); ``True`` models the sim account book (which may hold lots we never
    opened, so closes are ledger-scoped).
    """

    def __init__(
        self,
        book: PortfolioState,
        *,
        owns_book: bool = False,
        reject: bool = False,
        open_pid: str | None = "L1",
        scope: str = "S1",
    ) -> None:
        self._book = book
        self.owns_book = owns_book
        self.scope = scope
        self.seeded: PortfolioState | None = None
        self.placed: list[OrderIntent] = []
        self.resynced = 0
        self.events: list[str] = []
        self._reject = reject
        self._open_pid = open_pid

    async def read_book(self) -> FetchResult:
        return Ok(PortfolioSnapshot(self._book, TS))

    async def resync(self) -> Result[tuple[OrderResult, ...], FeedError]:
        self.resynced += 1
        self.events.append("resync")
        return Ok(())

    async def place(self, book: PortfolioState, intent: OrderIntent) -> PlaceResult:
        self.events.append("place")
        self.seeded = book
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
        self, book: PortfolioState, intents: tuple[OrderIntent, ...]
    ) -> (
        Ok[tuple[OrderResult, ...], FeedError] | Err[tuple[OrderResult, ...], FeedError]
    ):
        results: list[OrderResult] = []
        for intent in intents:
            placed = await self.place(book, intent)
            assert isinstance(placed, Ok)
            results.append(cast("OrderResult", placed.value))
        return Ok(tuple(results))

    async def close(self) -> Ok[None, FeedError]:
        return Ok(None)


class ErrResyncAdapter(FakeAdapter):
    """An adapter whose cycle-start resync returns an ``Err``."""

    async def resync(
        self,
    ) -> Err[tuple[OrderResult, ...], FeedError]:
        self.resynced += 1
        return Err(FeedError(kind="transport", message="open_orders failed"))


class ErrCohortAdapter(FakeAdapter):
    """An adapter whose cohort placement fails at the cohort level (a port-level Err)."""

    async def place_cohort(
        self, book: PortfolioState, intents: tuple[OrderIntent, ...]
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
        return con.execute(
            "SELECT position_id, closed_at FROM live_position"
        ).fetchall()
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


# --- run_cycle --------------------------------------------------------------


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
        adapter=FakeAdapter(book(lot("L1")), owns_book=True),
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

    with ledger.cycle_lease("S1"):
        with pytest.raises(CycleInProgressError):
            await run_cycle(
                CFG,
                adapter=FakeAdapter(book()),
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

    with ledger.cycle_lease("S1"):
        report = await run_cycle(
            CFG,
            adapter=FakeAdapter(book()),
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
        adapter=FakeAdapter(book(), owns_book=True, open_pid=None),
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
        adapter=FakeAdapter(book(lot("L1")), owns_book=True, reject=True),
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
async def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger_path = tmp_path / "l.sqlite"
    ledger = SqliteLedger(ledger_path)
    adapter = FakeAdapter(book())

    report = await run_cycle(
        CFG,
        adapter=adapter,
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
    assert adapter.placed == []
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
    adapter = FakeAdapter(book())

    report = await run_cycle(
        CFG,
        adapter=adapter,
        ledger=ledger,
        strategy_id="S1",
        scope="S1",
        config_path="x.json",
        now=TS,
        db_path=db,
        dry_run=True,
        signal_source=_source_fn(LONG_10),
    )

    assert report.results == () and adapter.placed == []
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
        "live_order_intent",
    }
    assert tables.isdisjoint(live_tables)  # peewee DDL withheld on a dry run


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
        adapter=ErrCohortAdapter(book()),
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
async def test_dry_run_never_resyncs(tmp_path: Path) -> None:
    # resync persists state, which a read-only run must not do.
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger = SqliteLedger(tmp_path / "l.sqlite")
    adapter = FakeAdapter(book())

    await run_cycle(
        CFG,
        adapter=adapter,
        ledger=ledger,
        strategy_id="S1",
        scope="S1",
        config_path="x.json",
        now=TS,
        db_path=db,
        dry_run=True,
        signal_source=_source_fn(LONG_10),
    )

    assert adapter.resynced == 0


# --- the real sim adapter through the cycle (the seam, end to end) ----------


def _sim_adapter(
    ledger: SqliteLedger, scope: str, *, dry_run: bool = False
) -> SimAdapter:
    return build_sim_adapter(CFG, scope, ledger, dry_run, lambda _m: None)


@pytest.mark.asyncio
async def test_a_sim_close_settles_and_leaves_the_book_flat(tmp_path: Path) -> None:
    """A close through the adapter+ledger round trip empties the book."""
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger = SqliteLedger(tmp_path / "l.sqlite")
    scope = "sim_momentum_1a2b3c4d"

    await run_cycle(
        CFG,
        _sim_adapter(ledger, scope),
        ledger=ledger,
        strategy_id="S1",
        scope=scope,
        config_path="x.json",
        now=TS,
        db_path=db,
        signal_source=_source_fn(LONG_10),
    )
    assert ledger.sim_open_ids(scope) != frozenset()

    closed = await run_cycle(
        CFG,
        _sim_adapter(ledger, scope),
        ledger=ledger,
        strategy_id="S1",
        scope=scope,
        config_path="x.json",
        now=TS,
        db_path=db,
        signal_source=_source_fn(CLOSE),
    )

    assert [i.action for i in closed.intents] == [ActionType.close]
    assert [r.ok for r in closed.results] == [True]
    assert ledger.sim_open_ids(scope) == frozenset()


@pytest.mark.asyncio
async def test_a_second_cycle_on_the_same_scope_is_refused(tmp_path: Path) -> None:
    """The lease is PER SCOPE: one scope refuses a concurrent cycle."""
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger = SqliteLedger(tmp_path / "l.sqlite")

    with ledger.cycle_lease("sim_a_1"):
        with pytest.raises(CycleInProgressError):
            await run_cycle(
                CFG,
                _sim_adapter(ledger, "sim_a_1"),
                ledger=ledger,
                strategy_id="S1",
                scope="sim_a_1",
                config_path="x.json",
                now=TS,
                db_path=db,
                signal_source=_source_fn(LONG_10),
            )


@pytest.mark.asyncio
async def test_a_hand_edited_account_lot_is_a_divergence_and_unsafe(
    tmp_path: Path,
) -> None:
    """The sim divergence path: our fill fold vs the account rows disagree.

    A human editing the human-editable surface (here: opening a lot the fills do
    not explain) must surface as a report ``Divergence`` and make the cycle
    unsafe. Silently adopting the edit would re-size onto a book we cannot
    explain.
    """
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger = SqliteLedger(tmp_path / "l.sqlite")
    scope = "sim_momentum_1a2b3c4d"
    # An account lot with NO fill behind it (the edit a human makes).
    ledger.record_sim_lot(
        scope,
        SimLot(
            position_id="HAND_1",
            symbol="AAPL",
            side="long",
            qty=5.0,
            entry_price=100.0,
            opened_at=TS,
        ),
    )

    report = await run_cycle(
        CFG,
        _sim_adapter(ledger, scope),
        ledger=ledger,
        strategy_id="S1",
        scope=scope,
        config_path="x.json",
        now=TS,
        db_path=db,
        signal_source=_source_fn(CLOSE),
    )

    assert [d.position_id for d in report.divergences] == ["HAND_1"]
    assert report.divergences[0].kind == "missing_ours"
    assert report.is_unsafe()


@pytest.mark.asyncio
async def test_agreeing_books_report_no_divergence(tmp_path: Path) -> None:
    """A cycle whose fills explain its account rows is clean."""
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "AAPL", TS)
    ledger = SqliteLedger(tmp_path / "l.sqlite")
    scope = "sim_momentum_1a2b3c4d"

    first = await run_cycle(
        CFG,
        _sim_adapter(ledger, scope),
        ledger=ledger,
        strategy_id="S1",
        scope=scope,
        config_path="x.json",
        now=TS,
        db_path=db,
        signal_source=_source_fn(LONG_10),
    )
    second = await run_cycle(
        CFG,
        _sim_adapter(ledger, scope),
        ledger=ledger,
        strategy_id="S1",
        scope=scope,
        config_path="x.json",
        now=TS,
        db_path=db,
        signal_source=_source_fn(LONG_10),
    )

    assert first.divergences == ()
    assert second.divergences == ()
    assert not second.is_unsafe()


pytestmark = pytest.mark.db
