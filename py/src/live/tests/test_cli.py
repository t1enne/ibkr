"""Tests for the live CLI: config load, report rendering, arg validation.

The CLI path runs the real screen, so ``live_run`` is never invoked end to end;
we exercise the pure helpers plus click's own argument validation.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
import click
from click.testing import CliRunner

from src.bt import load_strategy
from src.bt.state import ActionType, ExecutionParams, PortfolioState, Position
from src.live.broker import OrderResult
from src.live.cli import (
    _STRATEGY_FIELDS,
    _strategy_config,
    _write_strategy_config,
    live_group,
    load_live_config,
    render_report,
)
from src.live.engine import CycleReport, run_cycle
from src.live.result import Ok, Result
from src.live.types import (
    FeedError,
    LiveConfig,
    LiveSignal,
    OrderIntent,
    PortfolioSnapshot,
    cost_provenance,
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
        positions={
            "AAPL": (
                Position(
                    symbol="AAPL",
                    qty=10.0,
                    entry_price=100.0,
                    entry_time=TS,
                    stop_loss=None,
                    take_profit=None,
                    last_price=100.0,
                    type=ActionType.long,
                    position_id="L1",
                ),
            )
        },
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


# --- help / arg validation --------------------------------------------------


def test_live_help_lists_flags() -> None:
    out = CliRunner().invoke(live_group, ["--help"])
    assert out.exit_code == 0


def test_live_run_help_lists_options() -> None:
    out = CliRunner().invoke(live_group, ["run", "--help"])
    assert out.exit_code == 0
    assert "--dry-run" in out.output
    assert "--max-age" in out.output
    assert "--format" in out.output
    assert "--no-gateway" in out.output


def test_live_run_missing_file_fails(tmp_path: Path) -> None:
    out = CliRunner().invoke(live_group, ["run", str(tmp_path / "nope.json")])
    assert out.exit_code != 0


def test_live_run_empty_portfolio_path_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeLedger:
        def ensure_strategy(self, *args: object, **kwargs: object) -> None: ...
        def ensure_cash(self, *args: object, **kwargs: object) -> None: ...

    monkeypatch.setattr("src.live.cli.SqliteLedger", lambda *a, **k: FakeLedger())
    path = write_config(tmp_path, portfolio_path="", mode="paper")
    out = CliRunner().invoke(live_group, ["run", path])
    assert out.exit_code != 0
    assert "portfolio_path" in out.output


# --- load_live_config -------------------------------------------------------


def test_load_live_config_maps_every_field(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        portfolio_path="pf.json",
        mode="live",
        size_mode="cash",
        size=0.25,
        max_symbol_allocation=0.5,
    )
    cfg = load_live_config(path)
    assert isinstance(cfg, LiveConfig)
    assert cfg.strategy_type == "vwatr_div_dsl"
    assert cfg.symbols == ("AAPL", "MSFT")
    assert cfg.initial_capital == 50000
    assert cfg.strategy_params == {"vwatr_period": 14}
    assert cfg.bars == ("1d",)
    assert cfg.warmup == "300d"
    assert cfg.size_mode == "cash"
    assert cfg.size == 0.25
    assert cfg.max_symbol_allocation == 0.5
    assert cfg.commission == 0.05
    assert cfg.portfolio_path == "pf.json"
    assert cfg.mode == "live"


def test_load_live_config_nested_sizing(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        sizing={"sizing_mode": "cash", "size": 0.25, "max_symbol_allocation": 0.4},
    )
    cfg = load_live_config(path)
    assert (cfg.size_mode, cfg.size, cfg.max_symbol_allocation) == ("cash", 0.25, 0.4)


def test_load_live_config_flat_overrides_nested(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        sizing={"sizing_mode": "cash", "size": 0.1, "max_symbol_allocation": 0.9},
        size_mode="equity",
        size=0.5,
    )
    cfg = load_live_config(path)
    assert cfg.size_mode == "equity"  # flat wins
    assert cfg.size == 0.5  # flat wins
    assert cfg.max_symbol_allocation == 0.9  # no flat key -> nested kept


def test_load_live_config_defaults(tmp_path: Path) -> None:
    cfg = load_live_config(write_config(tmp_path))
    assert cfg.mode == "paper"
    assert cfg.portfolio_path == ""
    assert (cfg.size_mode, cfg.size, cfg.max_symbol_allocation) == ("equity", 0.0, 1.0)


def test_load_live_config_bad_mode_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mode must be"):
        load_live_config(write_config(tmp_path, mode="bogus"))


def test_load_live_config_bad_size_mode_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="size_mode must be"):
        load_live_config(write_config(tmp_path, size_mode="bogus"))


def test_write_strategy_config_is_load_strategy_able(tmp_path: Path) -> None:
    path = write_config(tmp_path, portfolio_path="pf.json", mode="live", size=0.25)
    raw = json.loads(Path(path).read_text())
    strategy = _strategy_config(path, raw)
    with tempfile.TemporaryDirectory() as tmp:
        normalized = _write_strategy_config(strategy, tmp)
        # Reloadable through the strict canonical loader the screen bridge uses.
        reloaded = load_strategy(normalized)
        assert reloaded.name == strategy.name
        assert reloaded.strategy_type == strategy.strategy_type
        doc = json.loads(Path(normalized).read_text())
        # Strategy-only: no live-only keys leak through.
        assert set(doc) <= _STRATEGY_FIELDS
        assert "portfolio_path" not in doc
        assert "mode" not in doc
        assert "size" not in doc


def test_live_run_passes_commission_to_execution_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_exec(**kwargs: object) -> ExecutionParams:
        captured.update(kwargs)
        return ExecutionParams()

    class FakeLedger:
        def ensure_strategy(self, *a: object, **k: object) -> None: ...
        def ensure_cash(self, *a: object, **k: object) -> None: ...
        def prune_closed(self, *a: object, **k: object) -> int:
            return 0

    async def fake_cycle(*a: object, **k: object) -> CycleReport:
        return _report()

    monkeypatch.setattr("src.live.cli.create_execution_params", fake_exec)
    monkeypatch.setattr("src.live.cli.SqliteLedger", lambda *a, **k: FakeLedger())
    monkeypatch.setattr("src.live.cli.MockPortfolioSource", lambda p: object())
    monkeypatch.setattr("src.live.cli.run_cycle", fake_cycle)

    path = write_config(tmp_path, portfolio_path="pf.json", commission=0.05)
    out = CliRunner().invoke(live_group, ["run", path])
    assert out.exit_code == 0, out.output
    assert captured.get("fixed_commission") == 0.05


# --- render_report ----------------------------------------------------------


def test_render_report_json_roundtrips() -> None:
    report = _report()
    out = render_report(report, "json")
    assert out == render_report(report, "json")  # deterministic
    doc = json.loads(out)
    assert doc["as_of"] == str(TS)
    assert len(doc["signals"]) == 1
    assert len(doc["intents"]) == 1
    assert doc["signals"][0]["symbol"] == "AAPL"
    assert doc["portfolio_before"]["cash"] == 49000.0
    assert doc["portfolio_before"]["positions"] == {"AAPL": 1}


def test_render_report_text_is_deterministic_and_listed() -> None:
    report = _report()
    out = render_report(report, "text")
    assert out == render_report(report, "text")
    assert "AAPL" in out
    assert "1 intents" in out
    assert "cash: 49000.00" in out


def test_cost_provenance_is_broker_exact_book_only_for_ibkr() -> None:
    assert cost_provenance("ibkr").bookkeeping == "broker_executions"
    assert cost_provenance("ibkr").sizing == "modelled"  # sizing stays modelled
    sim = cost_provenance("sim")
    assert (sim.bookkeeping, sim.sizing) == ("modelled", "modelled")


def test_render_report_states_both_cost_sources() -> None:
    """A mixed IBKR run must name BOTH sources, not one ambiguous tag (plan §7.3)."""
    report = replace(_report(), cost=cost_provenance("ibkr"))
    text = render_report(report, "text")
    assert "costs: bookkeeping=broker_executions sizing=modelled" in text
    doc = json.loads(render_report(report, "json"))
    assert doc["costs"] == {"bookkeeping": "broker_executions", "sizing": "modelled"}


# --- dry-run (engine flag, exercised through the CLI's engine call) ---------


class _RecordingLedger:
    def __init__(self) -> None:
        self.touched = 0
        self.opens: list[object] = []
        self.closed: list[str] = []

    def ensure_strategy(self, *a: object, **k: object) -> None: ...

    def ensure_cash(self, *a: object, **k: object) -> None: ...

    def prune_closed(self, before: pd.Timestamp) -> int:
        return 0

    def touch_cycle(self, strategy_id: str, at: pd.Timestamp) -> None:
        self.touched += 1

    def sim_open_ids(self, strategy_id: str) -> frozenset[str]:
        return frozenset(cast("str", p) for p in self.opens)

    def record_sim_open(self, strategy_id: str, position_id: str) -> None:
        self.opens.append(position_id)

    def mark_sim_closed(
        self, strategy_id: str, position_id: str, closed_at: pd.Timestamp
    ) -> None:
        self.closed.append(position_id)


class _FakeBroker:
    def __init__(self) -> None:
        self.placed: list[OrderIntent] = []

    def seed(self, portfolio: PortfolioState) -> None: ...

    async def place(self, intent: OrderIntent) -> Result[OrderResult, FeedError]:
        self.placed.append(intent)
        return Ok(OrderResult(intent=intent, fill=None, ok=True))

    async def place_cohort(
        self, intents: tuple[OrderIntent, ...]
    ) -> Result[tuple[OrderResult, ...], FeedError]:
        results: list[OrderResult] = []
        for intent in intents:
            placed = await self.place(intent)
            assert isinstance(placed, Ok)
            results.append(cast("OrderResult", placed.value))
        return Ok(tuple(results))

    async def close(self) -> Result[None, FeedError]:
        return Ok(None)


class _FakeSource:
    owns_book = True

    async def fetch(self) -> Result[PortfolioSnapshot, FeedError]:
        book = PortfolioState(
            cash=50000.0,
            positions={},
            trades=(),
            equity_curve=(),
            initial_capital=50000.0,
        )
        return Ok(PortfolioSnapshot(book, TS))


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
    ledger = _RecordingLedger()
    broker = _FakeBroker()

    def source_fn(
        config_path: str, max_age_days: int | None = None
    ) -> tuple[LiveSignal, ...]:
        return (_signal(),)

    report = await run_cycle(
        _dry_cfg(),
        source=_FakeSource(),
        broker=broker,
        ledger=ledger,
        strategy_id="S1",
        config_path="x.json",
        max_age_days=0,
        dry_run=True,
        now=TS,
        signal_source=source_fn,
    )

    assert [i.action for i in report.intents] == [ActionType.long]  # decided
    assert report.results == ()  # but nothing placed
    assert broker.placed == []
    assert ledger.opens == [] and ledger.closed == []
    assert ledger.touched == 0  # write-free: not even the cycle stamp


# --- adapter resolution (phase 2) -------------------------------------------


def _cfg(
    *, broker: str = "sim", mode: str = "paper", strategy_params: dict | None = None
) -> LiveConfig:
    return LiveConfig(
        strategy_type="vwatr_div_dsl",
        symbols=("AAPL",),
        initial_capital=50000.0,
        strategy_params=strategy_params or {},
        bars=("1d",),
        warmup="300d",
        broker=cast("Any", broker),
        mode=cast("Any", mode),
    )


def test_resolve_adapter_cli_flag_wins_over_config() -> None:
    from src.live.cli import resolve_adapter

    assert resolve_adapter("sim", {"broker": "ibkr"}, _cfg(broker="ibkr")) == "sim"
    assert resolve_adapter("ibkr", {}, _cfg()) == "ibkr"


def test_resolve_adapter_falls_back_to_config_broker() -> None:
    from src.live.cli import resolve_adapter

    assert resolve_adapter(None, {"broker": "ibkr"}, _cfg(broker="ibkr")) == "ibkr"
    assert resolve_adapter(None, {}, _cfg()) == "sim"


def test_resolve_adapter_reads_broker_from_strategy_params() -> None:
    from src.live.cli import resolve_adapter

    cfg = _cfg(broker="ibkr", strategy_params={"broker": "ibkr"})
    assert resolve_adapter(None, {}, cfg) == "ibkr"


def test_resolve_adapter_live_mode_must_name_its_adapter() -> None:
    from src.live.cli import resolve_adapter

    with pytest.raises(click.UsageError, match="must name its adapter"):
        resolve_adapter(None, {}, _cfg(mode="live"))


def test_ibkr_non_dry_run_builds_the_placing_broker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 3: an ibkr cycle without --dry-run reaches the real routing edge."""
    from src.live.adapters.ibkr.broker import IbkrBroker

    seen: dict[str, object] = {}

    monkeypatch.setattr("src.live.cli.IbkrGateway", _FakeGateway)
    monkeypatch.setattr("src.live.cli.IbkrClient", lambda *a, **k: object())
    monkeypatch.setattr("src.live.cli.IbkrPortfolioSource", lambda *a, **k: object())
    monkeypatch.setattr("src.live.cli.SqliteLedger", lambda *a, **k: FakeLedger())

    async def fake_cycle(*a: object, **k: object) -> CycleReport:
        seen["broker"] = k.get("broker")
        seen["dry_run"] = k.get("dry_run")
        return _report()

    monkeypatch.setattr("src.live.cli.run_cycle", fake_cycle)
    path = write_config(tmp_path, broker="ibkr", mode="paper")
    out = CliRunner().invoke(live_group, ["run", path, "--adapter", "ibkr"])
    assert out.exit_code == 0, out.output
    assert isinstance(seen["broker"], IbkrBroker)
    assert seen["dry_run"] is False


def test_live_run_labels_ibkr_cost_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI hands the report the broker-exact book provenance for an IBKR run."""
    seen: dict[str, object] = {}

    monkeypatch.setattr("src.live.cli.IbkrGateway", _FakeGateway)
    monkeypatch.setattr("src.live.cli.IbkrClient", lambda *a, **k: object())
    monkeypatch.setattr("src.live.cli.IbkrPortfolioSource", lambda *a, **k: object())
    monkeypatch.setattr("src.live.cli.SqliteLedger", lambda *a, **k: FakeLedger())

    async def fake_cycle(*a: object, **k: object) -> CycleReport:
        seen["cost"] = k.get("cost")
        return _report()

    monkeypatch.setattr("src.live.cli.run_cycle", fake_cycle)
    path = write_config(tmp_path, broker="ibkr", mode="paper")
    out = CliRunner().invoke(live_group, ["run", path, "--adapter", "ibkr"])
    assert out.exit_code == 0, out.output
    assert seen["cost"] == cost_provenance("ibkr")


def test_ibkr_dry_run_broker_still_refuses_to_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defence in depth: the dry-run broker refuses even if the CLI guard is skipped."""
    import asyncio

    from src.live.adapters.ibkr.broker import IbkrBroker
    from src.live.result import Err

    seen: dict[str, object] = {}

    monkeypatch.setattr("src.live.cli.IbkrGateway", _FakeGateway)
    monkeypatch.setattr("src.live.cli.IbkrClient", lambda *a, **k: object())
    monkeypatch.setattr("src.live.cli.IbkrPortfolioSource", lambda *a, **k: object())
    monkeypatch.setattr("src.live.cli.SqliteLedger", lambda *a, **k: FakeLedger())

    async def fake_cycle(*a: object, **k: object) -> CycleReport:
        seen["broker"] = k.get("broker")
        return _report()

    monkeypatch.setattr("src.live.cli.run_cycle", fake_cycle)
    path = write_config(tmp_path, broker="ibkr", mode="paper")
    out = CliRunner().invoke(
        live_group, ["run", path, "--dry-run", "--adapter", "ibkr"]
    )
    assert out.exit_code == 0, out.output
    broker = seen["broker"]
    assert isinstance(broker, IbkrBroker)
    placed = asyncio.run(broker.place(_intent()))
    assert isinstance(placed, Err)
    assert cast("FeedError", placed.error).kind == "auth"


def test_sim_path_never_builds_a_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: object, **k: object) -> object:
        raise AssertionError("IbkrGateway must not be built for the sim adapter")

    monkeypatch.setattr("src.live.cli.IbkrGateway", boom)
    monkeypatch.setattr("src.live.cli.SqliteLedger", lambda *a, **k: FakeLedger())
    monkeypatch.setattr("src.live.cli.MockPortfolioSource", lambda p: object())

    async def fake_cycle(*a: object, **k: object) -> CycleReport:
        return _report()

    monkeypatch.setattr("src.live.cli.run_cycle", fake_cycle)
    path = write_config(tmp_path, portfolio_path="pf.json")
    out = CliRunner().invoke(live_group, ["run", path, "--dry-run"])
    assert out.exit_code == 0, out.output


class FakeLedger:
    def ensure_strategy(self, *a: object, **k: object) -> None: ...
    def ensure_cash(self, *a: object, **k: object) -> None: ...
    def prune_closed(self, *a: object, **k: object) -> int:
        return 0


class _FakeGateway:
    def __init__(self, *a: object, **k: object) -> None:
        self.ready_calls = 0

        class _Client:
            async def resolve_account(self) -> str:
                return "DU1234567"

        self.client = _Client()

    async def ensure_ready(self, *a: object, **k: object) -> Result[None, FeedError]:
        self.ready_calls += 1
        return Ok(None)

    async def aclose(self) -> None:
        self.closed = True


def test_ibkr_ensures_ready_before_the_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[_FakeGateway] = []

    def make_gateway(*a: object, **k: object) -> _FakeGateway:
        gateway = _FakeGateway()
        made.append(gateway)
        return gateway

    monkeypatch.setattr("src.live.cli.IbkrGateway", make_gateway)
    monkeypatch.setattr("src.live.cli.IbkrClient", lambda *a, **k: object())
    monkeypatch.setattr("src.live.cli.IbkrPortfolioSource", lambda *a, **k: object())
    monkeypatch.setattr("src.live.cli.SqliteLedger", lambda *a, **k: FakeLedger())

    async def fake_cycle(*a: object, **k: object) -> CycleReport:
        assert made and made[0].ready_calls == 1  # ready BEFORE the cycle ran
        return _report()

    monkeypatch.setattr("src.live.cli.run_cycle", fake_cycle)
    path = write_config(tmp_path, broker="ibkr", mode="paper")
    out = CliRunner().invoke(
        live_group, ["run", path, "--dry-run", "--adapter", "ibkr"]
    )
    assert out.exit_code == 0, out.output
    assert made[0].ready_calls == 1


def test_no_gateway_skips_ensure_ready_and_still_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--no-gateway skips the readiness probe (loudly) yet still runs the cycle."""
    made: list[_FakeGateway] = []
    ran: dict[str, bool] = {}

    def make_gateway(*a: object, **k: object) -> _FakeGateway:
        gateway = _FakeGateway()
        made.append(gateway)
        return gateway

    monkeypatch.setattr("src.live.cli.IbkrGateway", make_gateway)
    monkeypatch.setattr("src.live.cli.IbkrClient", lambda *a, **k: object())
    monkeypatch.setattr("src.live.cli.IbkrPortfolioSource", lambda *a, **k: object())
    monkeypatch.setattr("src.live.cli.SqliteLedger", lambda *a, **k: FakeLedger())

    async def fake_cycle(*a: object, **k: object) -> CycleReport:
        ran["cycle"] = True
        return _report()

    monkeypatch.setattr("src.live.cli.run_cycle", fake_cycle)
    path = write_config(tmp_path, broker="ibkr", mode="paper")
    out = CliRunner().invoke(
        live_group,
        ["run", path, "--dry-run", "--adapter", "ibkr", "--no-gateway"],
    )
    assert out.exit_code == 0, out.output
    assert made[0].ready_calls == 0  # probe skipped
    assert ran.get("cycle") is True  # cycle still ran
    assert "readiness check skipped" in out.output  # never silent
