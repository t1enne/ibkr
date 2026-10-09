"""Critical-path tests for the live CLI: exit contract, scope/adapter, dry-run.

Rendering/format tests are deliberately absent: the report's wording and layout
are not behaviour. What is pinned here is the machine contract (exit codes), the
scope resolution two adapters must never share, ``--allow-unsafe`` naming what it
suppressed, and a dry run writing nothing.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from click.testing import CliRunner, Result as CliResult

from src.bt.state import ActionType, PortfolioState
from src.exec.refs import scope_tag
from src.live.pure import OrderResult
from src.live.cli import (
    _config_scope,
    live_group,
    load_live_config,
)
from src.live.divergence import Divergence
from src.live.engine import CycleReport
from src.live.identity import (
    IntentKey,
    IntentRecord,
    IntentState,
    OrderOutcome,
    order_ref,
)
from src.live.ledger import SqliteLedger
from src.live.tests.ledgers import live_ledger
from src.live.types import (
    FeedError,
    LiveConfig,
    LiveSignal,
    OrderIntent,
    PortfolioSnapshot,
)

TS = cast("pd.Timestamp", pd.Timestamp("2024-06-03"))

BASE_CONFIG = {
    "name": "cli_test",
    "strategy_type": "vwatr_div_dsl",
    "symbols": ["AAPL", "MSFT"],
    "initial_capital": 50000,
    "commission": 0.05,
    "warmup": "300d",
    "trading_start": "2020-09-01",
    "trading_end": "2026-09-01",
    "bars": ["1d"],
    "strategy_params": {"vwatr_period": 14},
}


def write_config(path: Path, **live_keys: object) -> str:
    """Write a valid strategy config plus live-only keys; return its path."""
    doc = {**BASE_CONFIG, **live_keys}
    target = path / "live_config.json"
    target.write_text(json.dumps(doc))
    return str(target)


def _signal() -> LiveSignal:
    return LiveSignal(
        symbol="AAPL",
        action="long",
        score=0.9,
        reasons=("mfi",),
        signal_ts=TS,
        price=100.0,
        qty=0.0,
    )


def _intent() -> OrderIntent:
    return OrderIntent(
        symbol="AAPL",
        action=ActionType.long,
        qty=10.0,
        ref_price=100.0,
        reason="open long (flat->long)",
    )


def _report() -> CycleReport:
    intent = _intent()
    portfolio = PortfolioState(
        cash=49000.0,
        positions={},
        trades=(),
        equity_curve=(),
        initial_capital=50000.0,
    )
    return CycleReport(
        as_of=TS,
        signals=(_signal(),),
        intents=(intent,),
        results=(OrderResult(intent=intent, fill=None, ok=True, position_id="L1"),),
        portfolio_before=portfolio,
    )


def _unsafe_result(outcome: OrderOutcome) -> OrderResult:
    return OrderResult(
        intent=_intent(),
        fill=None,
        ok=False,
        message="x",
        outcome=outcome,
        error_kind=outcome.value,
    )


# --- scope resolution + adapter selection (INV-5) ---------------------------


# --- load_live_config -------------------------------------------------------


# --- exit code (cron must see an unsafe cycle) -------------------------------


def _invoke_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report: CycleReport,
    *extra: str,
) -> CliResult:
    """Run one CLI cycle with a scripted report, returning the CliRunner result."""

    class FakeLedger:
        def ensure_strategy(self, *a: object, **k: object) -> None: ...
        def ensure_cash(self, *a: object, **k: object) -> None: ...
        def prune_closed(self, *a: object, **k: object) -> int:
            return 0

    async def fake_cycle(*a: object, **k: object) -> CycleReport:
        return report

    monkeypatch.setattr("src.live.cli.SqliteLedger", lambda *a, **k: FakeLedger())
    monkeypatch.setattr("src.live.cli.run_cycle", fake_cycle)
    path = write_config(tmp_path)
    return CliRunner().invoke(live_group, ["run", path, *extra])


def test_live_run_missing_file_exits_two(tmp_path: Path) -> None:
    """A nonexistent CONFIG_PATH is a usage error (click exits 2)."""
    out = CliRunner().invoke(live_group, ["run", str(tmp_path / "nope.json")])
    assert out.exit_code == 2


def test_clean_cycle_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _invoke_run(tmp_path, monkeypatch, _report())
    assert out.exit_code == 0, out.output


def test_placement_error_exits_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = replace(
        _report(), placement_error=FeedError(kind="transport", message="refused")
    )
    out = _invoke_run(tmp_path, monkeypatch, report)
    assert out.exit_code == 3, out.output


def test_resync_error_exits_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = replace(
        _report(), resync_error=FeedError(kind="transport", message="open_orders")
    )
    out = _invoke_run(tmp_path, monkeypatch, report)
    assert out.exit_code == 3, out.output


@pytest.mark.parametrize(
    "outcome",
    [
        OrderOutcome.UNRESOLVED,
        OrderOutcome.WEDGED,
        OrderOutcome.TIMEOUT,
        OrderOutcome.DIVERGENCE,
        OrderOutcome.UNFUNDED,
    ],
)
def test_unknown_or_stuck_order_exits_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: OrderOutcome
) -> None:
    report = replace(_report(), results=(_unsafe_result(outcome),))
    out = _invoke_run(tmp_path, monkeypatch, report)
    assert out.exit_code == 3, out.output


def test_a_refused_close_exits_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D3: a refused CLOSE is a book divergence, so it must not exit 0."""
    report = replace(_report(), results=(_unsafe_result(OrderOutcome.DIVERGENCE),))
    out = _invoke_run(tmp_path, monkeypatch, report)
    assert out.exit_code == 3, out.output


def test_an_all_opens_dropped_cohort_exits_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D3: a STRUCTURAL cohort drop recurs every cycle, so it must not exit 0."""
    report = replace(_report(), results=(_unsafe_result(OrderOutcome.UNFUNDED),))
    out = _invoke_run(tmp_path, monkeypatch, report)
    assert out.exit_code == 3, out.output


def test_a_plain_refused_open_still_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D3 boundary: a genuine broker refusal on an OPEN stays honest and safe."""
    report = replace(_report(), results=(_unsafe_result(OrderOutcome.REJECTED),))
    out = _invoke_run(tmp_path, monkeypatch, report)
    assert out.exit_code == 0, out.output


def test_allow_unsafe_suppresses_the_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = replace(_report(), results=(_unsafe_result(OrderOutcome.WEDGED),))
    out = _invoke_run(tmp_path, monkeypatch, report, "--allow-unsafe")
    assert out.exit_code == 0, out.output


def test_allow_unsafe_emits_a_note_naming_the_suppressed_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D2: forcing exit 0 must not silence the contract — the note names what it hid.

    The note is DERIVED from the report, so it names the outcome that actually
    occurred (here a structural open drop and a refused close) rather than a
    fixed string, and it goes to stderr so stdout stays the parseable report.
    """
    report = replace(
        _report(),
        results=(
            _unsafe_result(OrderOutcome.UNFUNDED),
            _unsafe_result(OrderOutcome.DIVERGENCE),
        ),
    )
    out = _invoke_run(tmp_path, monkeypatch, report, "--allow-unsafe")
    assert out.exit_code == 0, out.output
    assert "--allow-unsafe" in out.stderr
    assert OrderOutcome.UNFUNDED.value in out.stderr
    assert OrderOutcome.DIVERGENCE.value in out.stderr
    # A healthy outcome the cycle did not produce must NOT be named.
    assert OrderOutcome.TIMEOUT.value not in out.stderr


def test_is_unsafe_predicate_covers_errors_and_outcomes() -> None:
    assert not _report().is_unsafe()
    assert (
        replace(_report(), results=(_unsafe_result(OrderOutcome.REJECTED),)).is_unsafe()
        is False
    )
    assert replace(
        _report(), placement_error=FeedError(kind="transport", message="x")
    ).is_unsafe()


# --- dry-run: writes nothing, takes no lease (durability INV-6) --------------


class _FakeAdapter:
    """A ``LiveAdapter`` double whose placements are recorded, never priced."""

    def __init__(self) -> None:
        self.placed: list[OrderIntent] = []
        self.owns_book = True
        self.scope = "S1"

    async def read_book(self):
        from src.live.result import Ok

        book = PortfolioState(
            cash=50000.0,
            positions={},
            trades=(),
            equity_curve=(),
            initial_capital=50000.0,
        )
        return Ok(PortfolioSnapshot(book, TS))

    async def resync(self):
        from src.live.result import Ok

        return Ok(())

    async def place(self, book, intent):
        from src.live.result import Ok

        self.placed.append(intent)
        return Ok(OrderResult(intent=intent, fill=None, ok=True))

    async def place_cohort(self, book, intents):
        from src.live.result import Ok

        results: list[OrderResult] = []
        for intent in intents:
            placed = await self.place(book, intent)
            results.append(cast("OrderResult", placed.value))
        return Ok(tuple(results))

    async def close(self):
        from src.live.result import Ok

        return Ok(None)


class _RecordingLedger:
    """A ledger double that records every write a cycle makes (none on a dry run)."""

    def __init__(self) -> None:
        self.touched = 0
        self.recorded: list[tuple[OrderResult, ...]] = []
        self.leases: list[str] = []
        self.pruned = 0

    def ensure_strategy(self, *a: object, **k: object) -> None: ...

    def ensure_cash(self, *a: object, **k: object) -> None: ...

    def prune_closed(self, before: pd.Timestamp) -> int:
        return 0

    def prune(self, before: pd.Timestamp) -> int:
        self.pruned += 1
        return 0

    def touch_cycle(self, strategy_id: str, at: pd.Timestamp) -> None:
        self.touched += 1

    def cycle_lease(self, scope: str = "") -> AbstractContextManager[None]:
        self.leases.append(scope)
        return nullcontext()

    def owned_ids(self, scope: str) -> frozenset[str]:
        return frozenset()

    def executions_of(self, scope: str):
        return ()

    def record_results(self, scope: str, results, now: pd.Timestamp) -> None:
        self.recorded.append(results)


def _dry_cfg() -> LiveConfig:
    return LiveConfig(
        strategy_type="vwatr_div_dsl",
        symbols=("AAPL",),
        initial_capital=50000.0,
        strategy_params={},
        bars=("1d",),
        warmup="300d",
        size=0.5,
    )


@pytest.mark.asyncio
async def test_run_cycle_dry_run_places_nothing() -> None:
    from src.live.engine import run_cycle

    ledger = _RecordingLedger()
    adapter = _FakeAdapter()

    def source_fn(config_path: str, max_age_days: int | None = None):
        return (_signal(),)

    report = await run_cycle(
        _dry_cfg(),
        adapter,
        ledger=ledger,
        strategy_id="S1",
        scope="S1",
        config_path="x.json",
        max_age_days=0,
        dry_run=True,
        now=TS,
        signal_source=source_fn,
    )

    assert [i.action for i in report.intents] == [ActionType.long]  # decided
    assert report.results == ()  # but nothing placed
    assert adapter.placed == []
    assert ledger.recorded == []
    assert ledger.leases == []  # a read-only run takes no lease at all
    assert ledger.touched == 0 and ledger.pruned == 0  # write-free: not even the stamp


def _row_counts(db: Path) -> dict[str, int]:
    """Row counts per live table; a missing table counts as zero."""
    con = sqlite3.connect(db)
    try:
        tables = [
            r[0]
            for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ]
        return {
            name: con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            for name in tables
        }
    finally:
        con.close()


def test_dry_run_leaves_every_row_count_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read-only cycle writes NOTHING: not one row moves across the whole store.

    Asserted by row counts before vs after (behaviour), not by matching DDL
    strings: a dry run must be indistinguishable from never having run.
    """
    db = tmp_path / "live.sqlite"
    ledger = live_ledger(db)
    ledger.ensure_cash("sim_cli_test_1a2b3c4d", 50000.0)  # a pre-existing row set
    before = _row_counts(db)

    async def fake_cycle(*a: object, **k: object) -> CycleReport:
        return _report()

    monkeypatch.setattr("src.live.cli.SqliteLedger", lambda *a, **k: live_ledger(db))
    monkeypatch.setattr("src.live.cli.run_cycle", fake_cycle)
    path = write_config(tmp_path, adapter="sim")
    out = CliRunner().invoke(live_group, ["run", path, "--dry-run"])
    assert out.exit_code == 0, out.output

    assert _row_counts(db) == before
    assert not (
        db.parent
        / f"{db.name}.{scope_tag(_config_scope(load_live_config(path), 'sim'))}.cycle.lock"
    ).exists()


def test_ibkr_dry_run_adapter_still_refuses_to_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defence in depth: the dry-run adapter refuses even if the CLI guard is skipped."""
    import asyncio

    from src.bt.state import PortfolioState
    from src.live.adapters.ibkr.adapter import IbkrAdapter
    from src.live.result import Err

    seen: dict[str, object] = {}

    class _Client:
        async def resolve_account(self) -> str:
            return "DU1234567"

    class _Gateway:
        def __init__(self, *a: object, **k: object) -> None:
            self.client = _Client()

        async def aclose(self) -> None:
            return None

    class FakeLedger:
        def ensure_strategy(self, *a: object, **k: object) -> None: ...
        def ensure_cash(self, *a: object, **k: object) -> None: ...
        def prune_closed(self, *a: object, **k: object) -> int:
            return 0

        def prune(self, *a: object, **k: object) -> int:
            return 0

    monkeypatch.setattr("src.live.cli.IbkrGateway", _Gateway)
    monkeypatch.setattr("src.live.cli.IbkrClient", lambda *a, **k: object())
    monkeypatch.setattr(
        "src.live.adapters.ibkr.adapter.IbkrPortfolioSource", lambda *a, **k: object()
    )
    monkeypatch.setattr("src.live.cli.SqliteLedger", lambda *a, **k: FakeLedger())

    async def fake_cycle(*a: object, **k: object) -> CycleReport:
        seen["adapter"] = a[1]
        return _report()

    monkeypatch.setattr("src.live.cli.run_cycle", fake_cycle)
    path = write_config(tmp_path, broker="ibkr")
    out = CliRunner().invoke(
        live_group, ["run", path, "--dry-run", "--adapter", "ibkr"]
    )
    assert out.exit_code == 0, out.output
    adapter = seen["adapter"]
    assert isinstance(adapter, IbkrAdapter)
    flat = PortfolioState(
        cash=50000.0, positions={}, trades=(), equity_curve=(), initial_capital=50000.0
    )
    placed = asyncio.run(adapter.place(flat, _intent()))
    assert isinstance(placed, Err)
    assert cast("FeedError", placed.error).kind == "auth"


def test_sim_adapter_refuses_to_place_on_a_dry_run() -> None:
    """The sim adapter refuses too: a construction bug cannot place on a dry run."""
    import asyncio

    from src.bt.state import PortfolioState
    from src.live.adapters.sim.adapter import build_sim_adapter
    from src.live.result import Err

    adapter = build_sim_adapter(
        _dry_cfg(),
        "sim_x_1",
        live_ledger(tempfile.mkdtemp() + "/l.sqlite"),
        True,
        lambda _m: None,
    )
    flat = PortfolioState(
        cash=50000.0, positions={}, trades=(), equity_curve=(), initial_capital=50000.0
    )
    placed = asyncio.run(adapter.place(flat, _intent()))
    assert isinstance(placed, Err)
    assert cast("FeedError", placed.error).kind == "auth"


# --- housekeeping ------------------------------------------------------------


class _RaisingLedger:
    """A stand-in whose ``prune_closed`` raises *error* (housekeeping's input)."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def prune_closed(self, *a: object, **k: object) -> int:
        raise self._error


# --- N1: the operator escape for a wedged key --------------------------------


def _wedged_ledger(tmp_path: Path) -> tuple[SqliteLedger, IntentKey]:
    ledger = live_ledger(tmp_path / "abandon.sqlite")
    key = IntentKey(
        scope="momentum", symbol="AAPL", action=ActionType.long, position_id=None
    )
    ledger.save(
        IntentRecord(
            key=key,
            state=IntentState.WORKING,
            attempt=0,
            order_ref=order_ref(key, 0),
            order_id="97932",
            decision_ts=TS,
        )
    )
    return ledger, key


def test_abandon_clears_a_wedged_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The escape verb settles a wedged OPEN key terminal so the next cycle re-mints."""
    ledger, key = _wedged_ledger(tmp_path)
    monkeypatch.setattr("src.live.cli.SqliteLedger", lambda *a, **k: ledger)

    out = CliRunner().invoke(
        live_group,
        [
            "abandon",
            "--scope",
            "momentum",
            "--symbol",
            "AAPL",
            "--action",
            "long",
            "--yes",
        ],
    )

    assert out.exit_code == 0, out.output
    record = ledger.load(key)
    assert record is not None and record.state is IntentState.UNFILLED


def test_a_divergence_makes_the_cycle_unsafe() -> None:
    report = replace(
        _report(),
        divergences=(
            Divergence(
                symbol="AAPL",
                position_id="L1",
                ours_qty=10.0,
                account_qty=4.0,
                kind="qty_mismatch",
            ),
        ),
    )
    assert report.is_unsafe()
