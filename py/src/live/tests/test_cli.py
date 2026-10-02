"""Tests for the live CLI: config load, report rendering, arg validation.

The CLI path runs the real screen, so ``live_run`` is never invoked end to end;
we exercise the pure helpers plus click's own argument validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from click.testing import CliRunner

from src.bt.state import ActionType, PortfolioState, Position
from src.live.broker import OrderResult
from src.live.cli import live_group, load_live_config, render_report
from src.live.engine import CycleReport, run_cycle
from src.live.ledger import PositionRecord
from src.live.result import Ok, Result
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


def test_live_run_missing_file_fails(tmp_path: Path) -> None:
    out = CliRunner().invoke(live_group, ["run", str(tmp_path / "nope.json")])
    assert out.exit_code != 0


def test_live_run_empty_portfolio_path_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeLedger:
        def ensure_strategy(self, *args: object, **kwargs: object) -> None: ...

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


# --- dry-run (engine flag, exercised through the CLI's engine call) ---------


class _RecordingLedger:
    def __init__(self) -> None:
        self.touched = 0
        self.opens: list[PositionRecord] = []
        self.closed: list[str] = []

    def ensure_strategy(self, strategy_id: str, name: str, mode: str) -> None: ...

    def record_open(self, rec: PositionRecord) -> None:
        self.opens.append(rec)

    def mark_closed(
        self, strategy_id: str, position_id: str, closed_at: pd.Timestamp
    ) -> None:
        self.closed.append(position_id)

    def open_positions(self, strategy_id: str) -> tuple[PositionRecord, ...]:
        return ()

    def prune_closed(self, before: pd.Timestamp) -> int:
        return 0

    def touch_cycle(self, strategy_id: str, at: pd.Timestamp) -> None:
        self.touched += 1


class _FakeBroker:
    def __init__(self) -> None:
        self.placed: list[OrderIntent] = []

    def seed(self, portfolio: PortfolioState) -> None: ...

    async def place(self, intent: OrderIntent) -> Result[OrderResult, FeedError]:
        self.placed.append(intent)
        return Ok(OrderResult(intent=intent, fill=None, ok=True))

    async def close(self) -> Result[None, FeedError]:
        return Ok(None)


class _FakeSource:
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
    assert ledger.touched == 1  # cycle still stamped
