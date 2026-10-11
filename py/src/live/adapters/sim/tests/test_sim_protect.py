"""Step 4: the sim BROKER's bracket side — pair rests, fires, replaces (offline).

The sim store IS the broker: ``place_cohort`` writes the OCA pair into its
working-orders surface, ``resync`` fires a resting stop the lot's mark has
breached (reduce, signed cash, sibling cancel, trade), and ``apply_bracket``
REPLACEs both legs on a tighten — never one. ``--dry-run`` places nothing and
writes no JSON.

REGRESSION tests: each pins behaviour a broke broker would get wrong (a fired
stop that moves cash the wrong way, a sibling left resting after the stop fires,
a replace that cancels one leg, a dry run that mints a pair are all money bugs).
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from src.bt.state import ActionType
from src.data.db import get_connection
from src.exec.refs import scope_tag
from src.exec.types import OrderType
from src.live.pipeline import run_cycle
from src.live.adapters.sim.adapter import (
    SimAdapter,
    build_account_book,
    build_sim_adapter,
)
from src.live.adapters.sim.protect import apply_bracket, resting_rows
from src.live.adapters.sim.store import SimBook, SimBookStore, synthetic_conid
from src.live.identity import OrderOutcome
from src.live.ledger import SqliteLedger
from src.live.pure import OrderResult
from src.live.result import Err, Ok
from src.live.tests.ledgers import live_ledger
from src.live.types import LiveConfig, LiveSignal, OrderIntent

TS = cast("pd.Timestamp", pd.Timestamp("2024-06-03T14:30:00Z"))
SCOPE = "sim_t_protect"
TAG = scope_tag(SCOPE)
CONID = synthetic_conid("AAPL")

_NO_LOG: Callable[[str], None] = lambda _m: None  # noqa: E731


def _config() -> LiveConfig:
    return LiveConfig(
        strategy_type="m",
        symbols=("AAPL",),
        initial_capital=100_000.0,
        strategy_params={},
        bars=("1d",),
        warmup="1y",
    )


def _child(
    *,
    kind: str,
    level: float,
    position_side: ActionType = ActionType.short,
    qty: float = 10.0,
) -> OrderIntent:
    return OrderIntent(
        symbol="AAPL",
        action=ActionType.close,
        qty=qty,
        ref_price=level,
        reason=f"protect AAPL {kind}",
        position_id="L1",
        stop_loss=level if kind == "stop" else None,
        take_profit=level if kind == "take_profit" else None,
        decision_ts=TS,
        order_type=OrderType.STP if kind == "stop" else OrderType.LMT,
        protective=True,
        position_side=position_side,
    )


def _pair(*, sl: float = 92.0, tp: float = 108.0) -> tuple[OrderIntent, OrderIntent]:
    return (_child(kind="stop", level=sl), _child(kind="take_profit", level=tp))


def _long_lot_book(mark: float = 100.0) -> SimBook:
    """A funded sim book holding one long lot (the bracket's parent, resting)."""
    return SimBook(
        account="SIM",
        summary={"totalcashvalue": {"amount": 99_000.0, "currency": "USD"}},
        positions=(
            {
                "conid": CONID,
                "position_id": "L1",
                "contractDesc": "AAPL",
                "position": 10.0,
                "side": "long",
                "avgCost": 100.0,
                "mktPrice": mark,
                "currency": "USD",
            },
        ),
    )


def _adapter(ledger: SqliteLedger, store: SimBookStore) -> SimAdapter:
    return build_sim_adapter(_config(), SCOPE, ledger, False, _NO_LOG, store=store)


def _open_long_with_levels() -> OrderIntent:
    """The entry reconcile mints for a level-carrying long (spec Step 3 / D4)."""
    return OrderIntent(
        symbol="AAPL",
        action=ActionType.long,
        qty=10.0,
        ref_price=100.0,
        reason="open long (flat->long)",
        stop_loss=92.0,
        take_profit=108.0,
        decision_ts=TS,
    )


# --- pair rests / read back / refs ------------------------------------------


@pytest.mark.asyncio
async def test_a_confirmed_entry_rests_a_pair_in_the_working_surface() -> None:
    ledger = live_ledger(Path(tempfile.mkdtemp()) / "l.sqlite")
    store = SimBookStore(Path(tempfile.mkdtemp()) / "sim.json")
    adapter = _adapter(ledger, store)

    placed = await adapter.place_cohort(
        build_account_book(store, ledger, SCOPE, _config()),
        (_open_long_with_levels(),),
    )
    assert isinstance(placed, Ok)
    results = cast("tuple[OrderResult, ...]", placed.value)
    # The entry fill PLUS two protective children (both PLACED, never dotting).
    assert len(results) == 3
    children = [r for r in results if r.intent.protective]
    assert len(children) == 2
    assert all(r.ok and r.outcome is OrderOutcome.PLACED for r in children)

    book = store.read(SCOPE)
    pair = resting_rows(book, scope_tag=TAG, conid=CONID)
    assert pair is not None
    stop, take = pair
    # The broker's working surface is the single source of truth for LEVELS.
    assert stop.level == 92.0 and take.level == 108.0
    assert stop.status.upper() == "SUBMITTED" and take.status.upper() == "SUBMITTED"
    assert stop.side == take.side == "SELL"
    # Refs are scope-tagged and anchored on the LOT's own fill time (stable).
    assert stop.order_ref.startswith(f"{TAG}-")
    assert take.order_ref != stop.order_ref


@pytest.mark.asyncio
async def test_a_second_cycle_does_not_duplicate_a_resting_pair() -> None:
    """The working-orders read IS the re-mint guard: a resting pair is re-read,
    never re-sent (spec §4.2: never place a bracket for a lot that rests one)."""
    ledger = live_ledger(Path(tempfile.mkdtemp()) / "l.sqlite")
    store = SimBookStore(Path(tempfile.mkdtemp()) / "sim.json")
    adapter = _adapter(ledger, store)

    for _ in range(2):
        placed = await adapter.place_cohort(
            build_account_book(store, ledger, SCOPE, _config()),
            (_open_long_with_levels(),),
        )
        assert isinstance(placed, Ok)
        _ = cast("tuple[OrderResult, ...]", placed.value)
        # A repeat entry does NOT top up the lot (partial entry never topped),
        # but the resting pair stays put — replacing same-level legs is a
        # no-op-shaped REPLACE, never a duplicate POST.
        book = store.read(SCOPE)
        pair = resting_rows(book, scope_tag=TAG, conid=CONID)
        assert pair is not None
        assert len(book.orders) == 2


# --- fire: sibling cancel + signed cash + reduce -----------------------------


def _pair_book(mark: float) -> SimBook:
    """The broker's view after a pair rests: parent lot + both legs."""
    book = _long_lot_book(mark=mark)
    results, resting = apply_bracket(
        book, _pair(), scope_tag=TAG, conid=CONID, anchor=TS, log=_NO_LOG
    )
    assert len(results) == 2
    return resting


def _summary_cash(book: SimBook) -> float:
    return _num(cast("dict[str, object]", book.summary["totalcashvalue"]), "amount")


def _num(row: dict[str, object], key: str) -> float:
    """A JSON field narrowed to float (test helper over the broker's rows)."""
    return float(cast("float", row[key]))


@pytest.mark.asyncio
async def test_a_breached_long_stop_fires_reduces_credits_and_cancels_sibling() -> None:
    ledger = live_ledger(Path(tempfile.mkdtemp()) / "l.sqlite")
    store = SimBookStore(Path(tempfile.mkdtemp()) / "sim.json")
    adapter = _adapter(ledger, store)
    resting = _pair_book(mark=88.0)  # below the 92.0 stop: breached
    store.write(SCOPE, resting)

    resynced = await adapter.resync()
    assert isinstance(resynced, Ok)
    results = tuple(
        r for r in cast("tuple[OrderResult, ...]", resynced.value) if r.fill is not None
    )
    assert len(results) == 1
    fired = results[0]
    assert fired.fill is not None
    assert (
        fired.intent.position_side is ActionType.long
    )  # covers a long: BUY side is SELL

    book = store.read(SCOPE)
    # The lot is reduced to flat; a SELL CREDITs cash (the ONE signed rule).
    (lot,) = book.positions
    assert _num(dict(lot), "position") == 0.0
    assert _summary_cash(book) > 99_000.0
    # The OCA sibling is gone; the fired leg stays, marked Cancelled.
    assert len(book.orders) == 1
    (fired_row,) = book.orders
    assert fired_row["status"] == "Cancelled"
    # The fired stop's trade row landed in the broker's own trade list (a long
    # lot's stop is a SELL: side "S").
    assert book.trades
    (trade,) = book.trades
    assert trade["side"] == "S" and _num(dict(trade), "size") == 10.0


@pytest.mark.asyncio
async def test_a_breached_short_stop_fires_and_debits_cash() -> None:
    """A short's cover is a BUY: the shared rule DEBITs (long credits, short debits)."""
    ledger = live_ledger(Path(tempfile.mkdtemp()) / "l.sqlite")
    store = SimBookStore(Path(tempfile.mkdtemp()) / "sim.json")
    book = SimBook(
        account="SIM",
        summary={"totalcashvalue": {"amount": 99_000.0, "currency": "USD"}},
        positions=(
            {
                "conid": CONID,
                "position_id": "S1",
                "contractDesc": "AAPL",
                "position": -10.0,
                "side": "short",
                "avgCost": 100.0,
                "mktPrice": 130.0,  # above the 128.0 stop: breached
                "currency": "USD",
            },
        ),
    )
    children = (
        _child(kind="stop", level=128.0, position_side=ActionType.long),
        _child(kind="take_profit", level=80.0, position_side=ActionType.long),
    )
    results, resting = apply_bracket(
        book, children, scope_tag=TAG, conid=CONID, anchor=TS, log=_NO_LOG
    )
    assert len(results) == 2
    store.write(SCOPE, resting)

    resynced = await _adapter(ledger, store).resync()
    assert isinstance(resynced, Ok)
    fired = [
        r for r in cast("tuple[OrderResult, ...]", resynced.value) if r.fill is not None
    ]
    assert len(fired) == 1
    short = store.read(SCOPE)
    assert _num(dict(short.positions[0]), "position") == 0.0
    assert _summary_cash(short) < 99_000.0  # BUY debits
    assert len(short.orders) == 1  # only the fired (Cancelled) stop remains


# --- REPLACE on tighten ------------------------------------------------------


def test_replace_cancels_both_legs_then_rests_the_new_pair() -> None:
    book = _long_lot_book()
    _, resting = apply_bracket(
        book, _pair(), scope_tag=TAG, conid=CONID, anchor=TS, log=_NO_LOG
    )
    old = resting_rows(resting, scope_tag=TAG, conid=CONID)
    assert old is not None
    old_refs = {leg.order_ref for leg in old}

    tightened = _pair(sl=94.0, tp=106.0)
    results, replaced = apply_bracket(
        resting, tightened, scope_tag=TAG, conid=CONID, anchor=TS, log=_NO_LOG
    )
    assert len(results) == 2
    new = resting_rows(replaced, scope_tag=TAG, conid=CONID)
    assert new is not None
    new_stop, new_take = new
    # Levels moved, old refs are GONE (both legs cancelled first — never one).
    assert new_stop.level == 94.0 and new_take.level == 106.0
    assert all(leg.order_ref not in old_refs for leg in new)
    assert len(replaced.orders) == 2  # never a leftover half-bracket


# --- a fired stop advances OUR durable fold (no repeat DIVERGENCE) -----------


def _make_candle_db(path: Path, ts: pd.Timestamp) -> None:
    con = get_connection(path)
    con.execute("CREATE TABLE candle (ticker TEXT, timestamp INTEGER)")
    con.execute("INSERT INTO candle VALUES (?,?)", ("AAPL", int(ts.timestamp() * 1000)))
    con.commit()
    con.close()


def _long_source(
    config_path: str, max_age_days: int | None = None
) -> tuple[LiveSignal, ...]:
    return (
        LiveSignal(
            symbol="AAPL",
            action="long",
            score=1.0,
            reasons=(),
            signal_ts=TS,
            price=100.0,
            qty=10.0,
            stop_loss=92.0,
            take_profit=108.0,
        ),
    )


@pytest.mark.asyncio
async def test_a_fired_stop_advances_the_durable_fold_without_diverging(
    tmp_path: Path,
) -> None:
    """Cycle 1 opens the lot with the pair resting; cycle 2's resync fires the
    stop; the fired fill must land in ``live_execution`` so cycle 3 reads our
    fold closed and does NOT report a divergence against the flat account."""
    _make_candle_db(tmp_path / "c.sqlite", TS)
    ledger = live_ledger(tmp_path / "l.sqlite")
    store = SimBookStore(tmp_path / "sim.json")
    adapter = _adapter(ledger, store)

    first = await run_cycle(
        _config(),
        adapter,
        ledger=ledger,
        strategy_id="S1",
        scope=SCOPE,
        config_path="x.json",
        now=TS,
        db_path=tmp_path / "c.sqlite",
        signal_source=_long_source,
        max_age_days=0,
    )
    assert first.divergences == ()
    assert resting_rows(store.read(SCOPE), scope_tag=TAG, conid=CONID) is not None

    # The lot's mark drops below the resting stop (a mark edit, not a qty edit —
    # the divergence oracle compares signed qty only, so this is broker-side
    # price, not a human book edit).
    book = store.read(SCOPE)
    positions = [dict(r) for r in book.positions]
    positions[0]["mktPrice"] = 88.0
    store.write(
        SCOPE,
        SimBook(
            account=book.account,
            summary=book.summary,
            positions=tuple(positions),
            orders=book.orders,
            trades=book.trades,
        ),
    )

    fired = await run_cycle(
        _config(),
        _adapter(ledger, store),
        ledger=ledger,
        strategy_id="S1",
        scope=SCOPE,
        config_path="x.json",
        now=TS,
        db_path=tmp_path / "c.sqlite",
        signal_source=_long_source,
        max_age_days=0,
    )
    # Resync fired the stop; the fold now holds the close, so the cycle is clean.
    assert any(r.fill is not None for r in fired.results)
    assert fired.divergences == ()
    assert not fired.is_unsafe()
    assert ledger.executions_of(SCOPE)
    # The SELL credited cash: after the cycle-1 entry debit (~98 999), the fired
    # stop's credit lifts the fold back above it.
    assert ledger.cash_of(SCOPE, 100_000.0) > 98_999.0

    # Cycle 3: our fold and the account book both read flat — still clean.
    third = await run_cycle(
        _config(),
        _adapter(ledger, store),
        ledger=ledger,
        strategy_id="S1",
        scope=SCOPE,
        config_path="x.json",
        now=TS,
        db_path=tmp_path / "c.sqlite",
        signal_source=_long_source,
        max_age_days=0,
    )
    assert third.divergences == ()


# --- dry-run: nothing placed, nothing written --------------------------------


@pytest.mark.asyncio
async def test_dry_run_places_no_pair_and_writes_no_json(tmp_path: Path) -> None:
    ledger = live_ledger(tmp_path / "l.sqlite")
    store = SimBookStore(tmp_path / "sim.json")
    adapter = build_sim_adapter(_config(), SCOPE, ledger, True, _NO_LOG, store=store)

    placed = await adapter.place_cohort(
        build_account_book(store, ledger, SCOPE, _config()),
        (_open_long_with_levels(),),
    )
    assert isinstance(
        placed, Err
    )  # a dry-run broker refuses to place (defence in depth)

    book = store.read(SCOPE)
    assert book.orders == ()
    assert not (tmp_path / "sim.json").exists()  # and writes no JSON at all
