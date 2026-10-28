"""Critical-path tests for the bot log and the TARGET surface (plan §6).

One test per eligible behaviour (AGENTS.md § ``src/live`` tests); rendering,
shape and wording assertions are out of scope by that rule.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx
from click.testing import CliRunner

from src.bot.cli import bot_group, load_scope
from src.bot.journal import FileJournal, rotate
from src.bot.logs import follow, read_tail
from src.bot.logschema import Level, LogLine
from src.bot.loop import BotOptions, run_loop
from src.bot.registry import BotRegistry
from src.bot.render import LogStream, render_logs
from src.bot.targets import name_of, resolve_targets
from src.live.adapters.ibkr.tests.conftest import (
    BASE,
    ORDER_ID,
    SUBMIT,
    FakeExposure,
    broker,
    failure,
    open_intent,
)
from src.live.types import LiveConfig

OPEN_ORDERS = f"{BASE}iserver/account/orders"
TRADES = f"{BASE}iserver/account/trades"
POSITIONS = f"{BASE}portfolio/DU452563/positions/0"

BASE_CONFIG = {
    "name": "bot_regress",
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


def write_config(path: Path, *, name: str = "bot_regress") -> str:
    """Write a valid live strategy config; return its path."""
    path.write_text(json.dumps({**BASE_CONFIG, "name": name}))
    return str(path)


def _mini_cfg() -> LiveConfig:
    """A minimal LiveConfig: only the cadence-relevant fields matter here."""
    return LiveConfig(
        strategy_type="vwatr_div_dsl",
        symbols=("AAPL",),
        initial_capital=50_000.0,
        strategy_params={},
        bars=("1d",),
        warmup="300d",
    )


def _stamp(text: str) -> pd.Timestamp:
    """A UTC timestamp, narrowed (``pd.Timestamp`` also admits ``NaT``)."""
    parsed = pd.Timestamp(text)
    assert isinstance(parsed, pd.Timestamp)
    return parsed


class RecordingJournal:
    """A ``Journal`` double that records every emitted line (no file at all)."""

    def __init__(self) -> None:
        self.lines: list[LogLine] = []

    def emit(self, level: Level, event: str, fields: Mapping[str, str]) -> None:
        extra = {"scope": "sim_x", "pid": "1", **fields}
        self.lines.append(LogLine(_stamp("2026-01-01T00:00:00Z"), level, event, extra))

    def events(self) -> list[str]:
        return [line.event for line in self.lines]

    def codes(self, event: str) -> list[str]:
        return [line.fields["code"] for line in self.lines if line.event == event]


@dataclass
class ScriptedCycle:
    """The ``RunCycleFn`` seam: replays scripted codes and counts its calls."""

    codes: Sequence[int]
    calls: int = 0

    def __call__(self, config_path: str, max_age_days: int) -> int:
        self.calls += 1
        index = min(self.calls - 1, len(self.codes) - 1)
        return self.codes[index]


@dataclass
class StopAtSleep:
    """The ``SleepFn`` seam: writes the stop flag at the FIRST sleep point."""

    flag: Path

    def __call__(self, seconds: float) -> None:
        self.flag.write_text("")


class _NeverSpawner:
    """A ``BotSpawner`` that fails the test if anything is actually spawned."""

    def spawn(self, argv: Sequence[str], log_path: Path) -> int:
        raise AssertionError("a dry run must never spawn")


def _no_spawn_spawner() -> _NeverSpawner:
    """The never-spawning spawner stand-in."""
    return _NeverSpawner()


def _registry(tmp_path: Path) -> BotRegistry:
    return BotRegistry(tmp_path / "live.db", tmp_path / "runtime")


@pytest.fixture()
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    """A temp live db + runtime dir, so no command can touch the real ones."""
    monkeypatch.setattr("src.db.path.LIVE_DB_PATH", tmp_path / "live.db")
    monkeypatch.setenv("IBKR_BOT_RUNTIME_DIR", str(tmp_path / "runtime"))
    return CliRunner()


# --- §6 durability: a dry run writes nothing --------------------------------


def test_dry_run_writes_nothing(
    tmp_path: Path, isolated: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dry start/restart writes nothing: no file, above all no stop flag.

    The flag IS the stop request (§4), so writing it in a dry run would really
    stop a live bot.
    """
    registry = _registry(tmp_path)
    monkeypatch.setattr("src.bot.cli._build_registry", lambda: registry)
    monkeypatch.setattr("src.bot.cli._build_spawner", lambda: _no_spawn_spawner())
    config = write_config(tmp_path / "cfg.json")
    _, scope = load_scope(config)
    fd = registry.acquire_lifetime(scope)
    try:
        registry.write_pid_record(
            scope, pid=4242, started_at=pd.Timestamp.now(), config_path=config
        )
        before = {p.name for p in tmp_path.iterdir()}
        started = isolated.invoke(bot_group, ["start", config, "--dry-run"])
        restarted = isolated.invoke(bot_group, ["restart", config, "--dry-run"])
        after = {p.name for p in tmp_path.iterdir()}
    finally:
        os.close(fd)

    for result in (started, restarted):
        assert result.exit_code == 0, result.output
    assert before == after
    assert not registry.stop_flag_path(scope).exists()


# --- §2 exit codes: a log failure never changes the cycle's code ------------


def test_log_write_failure_does_not_change_cycle_code(tmp_path: Path) -> None:
    """An unwritable log still exits 0 and still stamps the pid row with 0.

    The log is a report, the pid row is the authority: otherwise a full disk
    would look like a trading failure.
    """
    scope = "sim_logfail_deadbeef"
    # A path whose parent is a FILE: mkdir/open must fail, deterministically.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    journal = FileJournal(path=blocker / "x.log", scope=scope, pid=1)
    registry = _registry(tmp_path)

    code = run_loop(
        "/cfg.json",
        cfg=_mini_cfg(),
        scope=scope,
        registry=registry,
        options=BotOptions(retries=0),
        run_cycle=ScriptedCycle([0]),
        sync=lambda cfg, *, today: None,
        sleep=StopAtSleep(registry.stop_flag_path(scope)),
        journal=journal,
    )

    assert code == 0
    row = registry.probe(scope)
    assert row is not None and row.exit_code == 0


def test_stop_observed_logged_then_exit_zero(tmp_path: Path) -> None:
    """A stop flag at the sleep point journals ``stop.observed`` and exits 0."""
    scope = "sim_stop_observed_deadbeef"
    journal = RecordingJournal()
    registry = _registry(tmp_path)

    code = run_loop(
        "/cfg.json",
        cfg=_mini_cfg(),
        scope=scope,
        registry=registry,
        options=BotOptions(retries=0),
        run_cycle=ScriptedCycle([0]),
        sync=lambda cfg, *, today: None,
        sleep=StopAtSleep(registry.stop_flag_path(scope)),
        journal=journal,
    )

    assert code == 0
    assert "stop.observed" in journal.events()
    assert journal.codes("bot.exit") == ["0"]


def test_transient_retries_exhaust_logs_terminal_code(tmp_path: Path) -> None:
    """Exhausted retries journal ``retries.exhausted`` and exit the last code."""
    scope = "sim_exhaust_deadbeef"
    journal = RecordingJournal()
    registry = _registry(tmp_path)
    cycle = ScriptedCycle([1])  # always transient

    code = run_loop(
        "/cfg.json",
        cfg=_mini_cfg(),
        scope=scope,
        registry=registry,
        options=BotOptions(retries=1),  # first + 1 retry, then the bound bites
        run_cycle=cycle,
        sync=lambda cfg, *, today: None,
        sleep=lambda seconds: None,  # never writes the flag: run to the bound
        journal=journal,
    )

    assert code == 1
    assert cycle.calls == 2
    assert "retries.exhausted" in journal.events()
    assert journal.codes("bot.exit") == ["1"]
    # The line's code and the row's code are written from the same value.
    assert journal.codes("cycle.exit")[-1] == "1"
    row = registry.probe(scope)
    assert row is not None and row.exit_code == 1


# --- §6 durability: rotation bounds, and a follower survives it -------------


def test_rotation_keeps_n_files_and_follow_reopens(tmp_path: Path) -> None:
    """Rotation keeps ``keep`` generations, and ``follow`` continues on the fresh file."""
    path = tmp_path / "sim_x.log"
    journal = FileJournal(path=path, scope="sim_x", pid=1, max_bytes=80, keep=2)

    for index in range(12):
        journal.emit("INFO", "cycle.exit", {"n": str(index)})

    generations = sorted(p.name for p in tmp_path.iterdir())
    assert generations == ["sim_x.log", "sim_x.log.1", "sim_x.log.2"]
    # The bound is checked BEFORE an append, so the live file may exceed it by
    # at most one line — never by a whole generation's worth.
    assert path.stat().st_size < 80 + 200

    # A follower positioned at the live file's end must land on the REPLACEMENT
    # file, not the renamed generation: the inode is the evidence.
    offset = path.stat().st_size
    before_inode = path.stat().st_ino
    landed: list[int] = []
    marker = "INFO cycle.exit n=after-rotation"
    journal_after = FileJournal(path=path, scope="sim_x", pid=1, max_bytes=80, keep=2)

    def tick() -> str | None:
        if landed:
            return None
        # Force a rotation, then append exactly one recognisable line.
        path.write_bytes(b"x" * 200)
        landed.append(1)
        journal_after.emit("INFO", "cycle.exit", {"n": "after-rotation"})
        return ""

    seen = list(follow(path, offset, tick))

    rotated_inode = path.stat().st_ino
    assert rotated_inode != before_inode, "the rotation must replace the live file"
    # The follower yielded the post-rotation line: it is on the live inode, not
    # replaying the renamed generation it was holding.
    assert any(marker in line for line in seen), seen


def test_rotate_zero_disables_and_renames_instead_of_truncating(
    tmp_path: Path,
) -> None:
    """``max_bytes=0`` never rotates; a real rotation RENAMES the live file."""
    path = tmp_path / "a.log"
    path.write_bytes(b"x" * 500)

    rotate(path, max_bytes=0, keep=2)
    assert path.stat().st_size == 500
    assert not (tmp_path / "a.log.1").exists()

    rotate(path, max_bytes=100, keep=1)
    assert (tmp_path / "a.log.1").stat().st_size == 500
    # A follower then sees the live name appear from zero, rather than replaying
    # a copied-and-truncated file.
    assert not path.exists()


def test_follow_ignores_a_garbled_line_but_keeps_the_stream(tmp_path: Path) -> None:
    """One unparseable-by-timestamp line passes through, never raises.

    ``2026-99-99T…`` matches the line regex but is not a real instant, so one
    bad byte must not take down ``logs`` for the whole file.
    """
    path = tmp_path / "garbled.log"
    path.write_text(
        "2026-99-99T00:00:00Z INFO cycle.exit code=0\n"
        "2026-08-15T09:30:09Z INFO cycle.exit code=1\n"
    )
    tail = read_tail("sim_x", path, None)

    rendered = render_logs(
        [LogStream(tail.scope, tail.path, tail.lines, tail.offset, tail.missing)],
        "json",
    )
    parsed = [json.loads(line) for line in rendered.splitlines()]

    assert parsed[0] == "2026-99-99T00:00:00Z INFO cycle.exit code=0"
    assert parsed[1]["fields"]["code"] == "1"


# --- §2 exit codes: an ambiguous TARGET is usage (2) ------------------------


def test_ambiguous_target_is_a_cli_usage_error(
    tmp_path: Path, isolated: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI maps that ambiguity to exit 2, never to an operational 1."""
    registry = _registry(tmp_path)
    monkeypatch.setattr("src.bot.cli._build_registry", lambda: registry)
    for name in ("bot_alpha_long", "bot_alpha_short"):
        config = write_config(tmp_path / f"{name}.json", name=name)
        _, scope = load_scope(config)
        registry.write_pid_record(
            scope, pid=4242, started_at=pd.Timestamp.now(), config_path=config
        )

    result = isolated.invoke(bot_group, ["stop", "ibkr_bot_alpha"])

    assert result.exit_code == 2, result.output
    assert not list(tmp_path.glob("*.bot.stop"))


# --- §5 identity/scope: registry-row-first, never re-minted -----------------


def test_target_resolves_from_registry_after_config_edit(tmp_path: Path) -> None:
    """An edited config still addresses the scope the live row holds."""
    registry = _registry(tmp_path)
    config = write_config(tmp_path / "cfg.json")
    _, original = load_scope(config)
    registry.write_pid_record(
        original, pid=4242, started_at=pd.Timestamp.now(), config_path=config
    )
    # A strategy-intent edit re-mints the scope — the operator risk AGENTS.md
    # names. A read-only command must NOT follow it to the new scope.
    edited = {**BASE_CONFIG, "symbols": ["AAPL", "MSFT", "NVDA"], "name": "bot_regress"}
    Path(config).write_text(json.dumps(edited))
    minted = load_scope(config)[1]
    assert minted != original

    resolved = resolve_targets(registry, [name_of(config)], mint=lambda p: minted)

    assert resolved[0].scope == original
    assert resolved[0].from_registry is True


def test_scope_is_never_reminted_by_read_only_commands(
    tmp_path: Path, isolated: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ls``/``inspect``/``logs``/``info`` create no identity file at all."""
    registry = _registry(tmp_path)
    monkeypatch.setattr("src.bot.cli._build_registry", lambda: registry)
    config = write_config(tmp_path / "cfg.json")
    _, scope = load_scope(config)
    registry.write_pid_record(
        scope, pid=4242, started_at=pd.Timestamp.now(), config_path=config
    )
    before = {p.name for p in tmp_path.iterdir()}

    for argv in (
        ["ls"],
        ["ls", "-q"],
        ["inspect", name_of(config)],
        ["inspect", name_of(config), "--adapter", "sim"],
        ["info"],
    ):
        result = isolated.invoke(bot_group, argv)
        assert result.exit_code == 0, (argv, result.output)
    # ``logs`` on a bot that never wrote one is the documented exit 1 — reading
    # it still creates nothing, which is what this test is about.
    missing = isolated.invoke(bot_group, ["logs", name_of(config)])
    assert missing.exit_code == 1

    after = {p.name for p in tmp_path.iterdir()}
    assert before == after  # no new pid row, no lock, no log


# --- §5 identity/scope: a re-mint is refused, never half-followed ----------


def test_start_refuses_a_target_whose_config_remints_the_scope(
    tmp_path: Path, isolated: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-minting config over a live row spawns NOTHING, exit 1.

    The parent would claim the row's scope while the child trades another — the
    double-exposure hazard. ``--dry-run`` reports the same refusal.
    """
    registry = _registry(tmp_path)
    monkeypatch.setattr("src.bot.cli._build_registry", lambda: registry)
    monkeypatch.setattr("src.bot.cli._build_spawner", lambda: _no_spawn_spawner())
    config = write_config(tmp_path / "cfg.json")
    _, original = load_scope(config)
    fd = registry.acquire_lifetime(original)
    registry.write_pid_record(
        original, pid=4242, started_at=pd.Timestamp.now(), config_path=config
    )
    Path(config).write_text(
        json.dumps({**BASE_CONFIG, "symbols": ["AAPL", "MSFT", "NVDA"]})
    )
    minted = load_scope(config)[1]
    rows_before = sorted(row.scope for row in registry.list_bots())
    try:
        started = isolated.invoke(bot_group, ["start", config])
        restarted = isolated.invoke(bot_group, ["restart", config, "--wait", "1"])
        dry = isolated.invoke(bot_group, ["start", config, "--dry-run"])
        rows_after = sorted(row.scope for row in registry.list_bots())
    finally:
        os.close(fd)

    for result in (started, restarted, dry):
        assert result.exit_code == 1, result.output
        assert original in result.output and minted in result.output
    # No child under EITHER identity, and no new row: the only row is the old one.
    assert rows_after == rows_before == [original]
    assert not registry.stop_flag_path(minted).exists()


# --- §1/§2: the journal rides on the loop, never before a cycle -------------


@respx.mock
@pytest.mark.asyncio
async def test_order_safety_unchanged_by_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed working-orders read still POSTs nothing, with the journal live.

    The journal is the REAL one on the loop's path (``run_loop`` builds it), and
    a submit would be visible to respx: INV-1 holds byte-for-byte beside it.
    """
    monkeypatch.setenv("IBKR_BOT_RUNTIME_DIR", str(tmp_path / "runtime"))
    submit = respx.post(SUBMIT).mock(
        return_value=httpx.Response(200, json=[{"order_id": ORDER_ID}])
    )
    respx.get(OPEN_ORDERS).mock(return_value=httpx.Response(500, json={}))
    # The execution-window sweep runs BEFORE the working-orders read; the
    # positions route is the account read a bare ``place`` also makes.
    respx.get(TRADES).mock(return_value=httpx.Response(200, json={"trades": []}))
    respx.get(POSITIONS).mock(return_value=httpx.Response(200, json=[]))
    registry = _registry(tmp_path)
    scope = "sim_ordersafety_deadbeef"

    error = failure(await broker(exposure=FakeExposure()).place(open_intent()))
    code = run_loop(
        "/cfg.json",
        cfg=_mini_cfg(),
        scope=scope,
        registry=registry,
        options=BotOptions(retries=0),
        run_cycle=ScriptedCycle([0]),
        sync=lambda cfg, *, today: None,
        sleep=StopAtSleep(registry.stop_flag_path(scope)),
    )

    assert error.kind in {"transport", "auth", "rate_limit"}
    assert submit.call_count == 0  # fail CLOSED: nothing sent
    # The loop's journal really was on the path: it wrote this scope's file.
    assert code == 0
    assert "code=0" in registry.log_path(scope).read_text()


# --- §6 durability: --since/--level filter, and the legacy bypass ----------


def test_render_logs_filters_by_since_and_level(tmp_path: Path) -> None:
    """``--since`` and ``--level`` select on the parsed line only."""
    path = tmp_path / "sim_x.log"
    lines = (
        "2026-08-15T09:00:00Z DEBUG cycle.exit code=0",
        "2026-08-15T09:30:09Z INFO cycle.exit code=0",
        "2026-08-15T10:00:00Z WARN sync.fail err=x",
        "cycle exit 3",  # legacy: unparseable, never filtered
    )
    path.write_text("".join(f"{line}\n" for line in lines))
    stream = LogStream("sim_x", path, lines, path.stat().st_size, False)

    by_level = render_logs([stream], "json", minimum="WARN")
    by_since = render_logs([stream], "json", since=_stamp("2026-08-15T09:45:00Z"))

    levels = [json.loads(line)["level"] for line in by_level.splitlines()[:-1]]
    times = [json.loads(line) for line in by_since.splitlines()[:-1]]
    assert levels == ["WARN"]
    assert [item["event"] for item in times] == ["sync.fail"]
    # The unparseable line bypasses BOTH filters in each pass.
    assert json.loads(by_level.splitlines()[-1]) == "cycle exit 3"
    assert json.loads(by_since.splitlines()[-1]) == "cycle exit 3"
