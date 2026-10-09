"""Critical-path tests for the JSON sim broker store.

The store is the operator-editable account book; the sqlite ``live_execution``
fold is our own record. These tests pin the JSON shape and the write-back /
divergence behaviour a broken store would get wrong:

- (a) the file is never created on a read, round-trips on a write, reads empty on
  malformed JSON, and leaves no ``.tmp`` behind;
- (b) a placed cycle writes a position + trade and moves cash; a close reduces the
  row to zero;

- (c) a hand-edit yields a non-empty divergence and an unsafe cycle, unedited is
  clean;
- (d) a ``--dry-run`` writes no JSON.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from src.data.db import get_connection
from src.live.adapters.ibkr.mapping import num, opt_str
from src.live.adapters.sim.adapter import build_sim_adapter
from src.live.adapters.sim.store import SimBook, SimBookStore
from src.live.engine import run_cycle
from src.live.pf import read_sim_broker
from src.live.tests.ledgers import live_ledger
from src.live.types import LiveConfig, LiveSignal, SignalAction

TS = cast("pd.Timestamp", pd.Timestamp("2024-06-03"))

CFG = LiveConfig(
    strategy_type="momentum",
    symbols=("AAPL",),
    initial_capital=100_000.0,
    strategy_params={},
    bars=("1d",),
    warmup="1y",
)

_NO_LOG: Callable[[str], None] = lambda _m: None  # noqa: E731


def _make_candle_db(path: Path, ticker: str | None, ts: pd.Timestamp | None) -> None:
    con = get_connection(path)
    con.execute("CREATE TABLE candle (ticker TEXT, timestamp INTEGER)")
    if ticker is not None and ts is not None:
        con.execute(
            "INSERT INTO candle VALUES (?,?)", (ticker, int(ts.timestamp() * 1000))
        )
    con.commit()
    con.close()


def _source_fn(acts: tuple[tuple[SignalAction, float], ...]):
    def _f(path: str, max_age: int | None = None) -> tuple[LiveSignal, ...]:
        return tuple(
            LiveSignal(
                symbol="AAPL",
                action=a,
                score=1.0,
                reasons=(),
                signal_ts=TS,
                price=100.0,
                qty=q,
            )
            for a, q in acts
        )

    return _f


LONG_10: tuple[tuple[SignalAction, float], ...] = (("long", 10.0),)
SHORT_10: tuple[tuple[SignalAction, float], ...] = (("short", 10.0),)
CLOSE: tuple[tuple[SignalAction, float], ...] = (("close", 0.0),)


def _adapter(ledger, scope: str, store: SimBookStore, *, dry_run: bool = False):
    return build_sim_adapter(CFG, scope, ledger, dry_run, _NO_LOG, store=store)


def _run(ledger, scope: str, store: SimBookStore, acts, *, dry_run: bool = False):
    return run_cycle(
        CFG,
        _adapter(ledger, scope, store, dry_run=dry_run),
        ledger=ledger,
        strategy_id="S1",
        scope=scope,
        config_path="x.json",
        now=TS,
        db_path=None,
        signal_source=_source_fn(acts),
        dry_run=dry_run,
        max_age_days=0,
    )


# --- (a) read/write/malformed/no-tmp ----------------------------------------


def test_absent_file_reads_empty_and_creates_nothing(tmp_path: Path) -> None:
    path = tmp_path / "sim.json"
    store = SimBookStore(path)
    book = store.read("scope")
    assert book.account == "" and book.positions == ()
    assert not path.exists()  # read NEVER creates the file


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    store = SimBookStore(tmp_path / "sim.json")
    store.write(
        "s1",
        SimBook(
            account="SIM",
            summary={
                "netliquidation": {"amount": 1000.0, "currency": "USD"},
                "totalcashvalue": {"amount": 900.0, "currency": "USD"},
            },
            positions=(
                {
                    "conid": 1,
                    "position_id": "P1",
                    "contractDesc": "AAPL",
                    "position": 10.0,
                    "avgCost": 100.0,
                    "mktPrice": 110.0,
                    "currency": "USD",
                },
            ),
        ),
    )
    book = store.read("s1")
    assert book.account == "SIM"
    assert book.positions and book.positions[0]["position_id"] == "P1"
    # The file is keyed by scope.
    raw = json.loads((tmp_path / "sim.json").read_text())
    assert "s1" in raw["scopes"]


def test_malformed_json_reads_empty(tmp_path: Path) -> None:
    path = tmp_path / "sim.json"
    path.write_text("{ not json")
    assert SimBookStore(path).read("s1").positions == ()


def test_write_leaves_no_tmp_file(tmp_path: Path) -> None:
    path = tmp_path / "sim.json"
    store = SimBookStore(path)
    store.write("s1", SimBook(account="SIM"))
    assert path.exists()
    assert not path.with_suffix(".json.tmp").exists()


# --- (b) a placed cycle writes the JSON book -------------------------------


@pytest.mark.asyncio
async def test_a_cycle_writes_position_trade_and_cash(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    _make_candle_db(db, "AAPL", TS)
    ledger = live_ledger(tmp_path / "l.sqlite")
    scope = "sim_x_1"
    store = SimBookStore(tmp_path / "sim.json")

    await _run(ledger, scope, store, LONG_10)

    book = store.read(scope)
    assert len(book.positions) == 1
    (row,) = book.positions
    assert row["position_id"] != ""
    assert row["contractDesc"] == "AAPL"
    assert num(row["position"]) > 0.0
    # A trade row was appended, and cash moved (debited) from the summary.
    assert book.trades and opt_str(book.trades[0]["execution_id"]).endswith(":open")
    total_cash = cast("dict", book.summary["totalcashvalue"])["amount"]
    assert float(total_cash) < 100_000.0


@pytest.mark.asyncio
async def test_a_close_zeroes_the_position_row(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    _make_candle_db(db, "AAPL", TS)
    ledger = live_ledger(tmp_path / "l.sqlite")
    scope = "sim_x_1"
    store = SimBookStore(tmp_path / "sim.json")

    await _run(ledger, scope, store, LONG_10)
    await _run(ledger, scope, store, CLOSE)

    book = store.read(scope)
    assert book.positions and num(book.positions[0]["position"]) == 0.0


@pytest.mark.asyncio
async def test_a_short_cycle_writes_a_short_row_and_does_not_diverge(
    tmp_path: Path,
) -> None:
    """A short open must read back as short, or an unedited book falsely diverges."""
    db = tmp_path / "c.sqlite"
    _make_candle_db(db, "AAPL", TS)
    ledger = live_ledger(tmp_path / "l.sqlite")
    scope = "sim_x_1"
    store = SimBookStore(tmp_path / "sim.json")

    await _run(ledger, scope, store, SHORT_10)
    book = store.read(scope)
    (row,) = book.positions
    assert opt_str(row["side"]) == "short"
    assert num(row["position"]) < 0.0

    # Our fill fold also holds a short, so the unedited pair agrees.
    again = await _run(ledger, scope, store, SHORT_10)
    assert again.divergences == ()


def test_two_lots_of_one_symbol_stay_distinct(tmp_path: Path) -> None:
    """Sim lots are per-lot: a shared synthetic conid must not collapse their ids."""
    store = SimBookStore(tmp_path / "sim.json")
    row = {
        "conid": 1,
        "contractDesc": "AAPL",
        "avgCost": 100.0,
        "mktPrice": 100.0,
    }
    store.write(
        "s1",
        SimBook(
            account="SIM",
            positions=(
                {**row, "position_id": "P1", "position": 10.0},
                {**row, "position_id": "P2", "position": 5.0},
            ),
        ),
    )
    broker = read_sim_broker(store, "s1", frozenset({"P1", "P2"}))
    assert {lot.id for lot in broker.positions} == {"P1", "P2"}


# --- (c) hand-edit -> divergence + unsafe -----------------------------------


@pytest.mark.asyncio
async def test_a_hand_edit_diverges_and_is_unsafe(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    _make_candle_db(db, "AAPL", TS)
    ledger = live_ledger(tmp_path / "l.sqlite")
    scope = "sim_x_1"
    store = SimBookStore(tmp_path / "sim.json")

    first = await _run(ledger, scope, store, LONG_10)
    assert first.divergences == ()  # unedited is clean

    # Hand-edit: bump the position qty the fills do not explain.
    book = store.read(scope)
    positions = [dict(r) for r in book.positions]
    positions[0]["position"] = num(positions[0]["position"]) + 5.0
    store.write(scope, SimBook(account="SIM", positions=tuple(positions)))

    edited = await _run(ledger, scope, store, LONG_10)
    assert edited.divergences != ()
    assert edited.is_unsafe()


# --- (d) dry run writes no JSON ---------------------------------------------


@pytest.mark.asyncio
async def test_dry_run_writes_no_json(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    _make_candle_db(db, "AAPL", TS)
    ledger = live_ledger(tmp_path / "l.sqlite")
    scope = "sim_x_1"
    store = SimBookStore(tmp_path / "sim.json")

    await _run(ledger, scope, store, LONG_10, dry_run=True)

    assert not (tmp_path / "sim.json").exists()
