"""Critical-path tests for ``bot restart`` and ``bot logs``.

Restart must REPLACE a live bot (stop, then spawn a new life) and must never
replace one that ignored the stop. Logs only read. The fake child here is
stop-aware — it consumes its stop flag and releases the lifetime lock, exactly
what a live bot's exit looks like to the registry — so replacement is visible
as a new pid/``started_at``.

aim: pin restart ordering, the never-force-kill rule, target isolation, dry-run
purity, the tail window, the missing-log rule, following, and reader-only purity.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from src.bot import cli as bot_cli
from src.bot.cli import bot_group, load_scope
from src.bot.logs import follow
from src.bot.registry import BotRegistry, runtime_dir
from src.bot.spawn import ready_fd_of, run_config_of
from src.db.path import resolve_live_db_path

BASE_CONFIG = {
    "name": "bot_test",
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

#: How often the fake bot re-reads its stop flag (the loop's sleep point).
_POLL_SECONDS = 0.02


def write_config(path: Path, *, name: str = "bot_test") -> str:
    """Write a valid live strategy config; return its path."""
    path.write_text(json.dumps({**BASE_CONFIG, "name": name}))
    return str(path)


def log_for(scope: str) -> Path:
    """The log path the registry mints for *scope*."""
    return BotRegistry(resolve_live_db_path()).log_path(scope)


def seed_log(scope: str, lines: int) -> Path:
    """Create *scope*'s log holding ``cycle exit 0`` *lines* times."""
    path = log_for(scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join("cycle exit 0\n" for _ in range(lines)))
    return path


def seed_row(scope: str, config_path: str) -> None:
    """Register a pid row (never live) so ``--all`` has a row to target."""
    started_at = pd.Timestamp(0)
    assert isinstance(started_at, pd.Timestamp)  # narrows Timestamp | NaTType
    BotRegistry(resolve_live_db_path()).write_pid_record(
        scope, pid=4242, started_at=started_at, config_path=config_path
    )


def digest_runtime() -> str:
    """A content hash of every file under the runtime dir (names + bytes)."""
    root = runtime_dir()
    if not root.exists():
        return "absent"
    parts: list[str] = []
    for path in sorted(root.rglob("*")):
        parts.append(str(path.relative_to(root)))
        if path.is_file():
            parts.append(hashlib.sha256(path.read_bytes()).hexdigest())
        parts.append(str(path.stat().st_mtime_ns))
    return "|".join(parts)


class StopAwareSpawner:
    """In-process stand-in for a spawned bot that honours ``bot stop``."""

    def __init__(self, registry: BotRegistry) -> None:
        self.registry = registry
        self.calls = 0
        self.held: dict[str, int] = {}
        self._stop = threading.Event()

    def observe_stop(self) -> None:
        """Exit each live fake whose scope saw a stop flag (the loop's poll)."""
        while not self._stop.is_set():
            for scope in list(self.held):
                flag = self.registry.stop_flag_path(scope)
                if not flag.exists():
                    continue
                fd = self.held.pop(scope, None)
                if fd is None:
                    continue
                flag.unlink(missing_ok=True)  # the bot consumed its request
                os.close(fd)  # the lifetime lock drops: the scope is not live
            self._stop.wait(_POLL_SECONDS)

    def spawn(self, argv: Sequence[str], log_path: Path) -> int:
        self.calls += 1
        cfg = run_config_of(argv)
        assert cfg is not None
        _, scope = load_scope(cfg)
        assert scope not in self.held, "a second bot spawned over a live one"
        pid = os.getpid() + self.calls
        self.held[scope] = self.registry.acquire_lifetime(scope)
        self.registry.write_pid_record(
            scope,
            pid=pid,
            started_at=pd.Timestamp.now(),
            config_path=str(Path(cfg).resolve()),
        )
        ready = ready_fd_of(argv)
        if ready is not None:
            os.write(ready, b"1")
        return pid


class Watcher:
    """Run :meth:`StopAwareSpawner.observe_stop` beside a CLI call."""

    def __init__(self, spawner: StopAwareSpawner) -> None:
        self.spawner = spawner
        self.thread = threading.Thread(target=spawner.observe_stop, daemon=True)

    def __enter__(self) -> Watcher:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.spawner._stop.set()
        self.thread.join(timeout=30)
        assert not self.thread.is_alive(), "stop watcher hung"


def drain(spawner: StopAwareSpawner) -> None:
    """Release every fake's lifetime fd as if the bots had exited."""
    for fd in spawner.held.values():
        os.close(fd)
    spawner.held.clear()


@pytest.fixture()
def runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    monkeypatch.setattr("src.db.path.LIVE_DB_PATH", tmp_path / "live.db")
    monkeypatch.setenv("IBKR_BOT_RUNTIME_DIR", str(tmp_path / "runtime"))
    return CliRunner()


@pytest.fixture()
def spawner(monkeypatch: pytest.MonkeyPatch) -> StopAwareSpawner:
    fake = StopAwareSpawner(BotRegistry(resolve_live_db_path()))
    monkeypatch.setattr("src.bot.cli._build_spawner", lambda: fake)
    return fake


# --- restart ----------------------------------------------------------------


def test_restart_live_bot_stops_then_respawns_with_new_identity(
    tmp_path: Path, runner: CliRunner, spawner: StopAwareSpawner
) -> None:
    """Pins stop-then-spawn: a live bot's restart yields a NEW pid/started_at.

    A restart that only ran ``start`` would report "already live" and leave the
    identity unchanged, so the different identity IS the proof of a real stop.
    """
    config = write_config(tmp_path / "cfg.json")
    _, scope = load_scope(config)
    assert runner.invoke(bot_group, ["start", config]).exit_code == 0
    before = spawner.registry.probe(scope)
    assert before is not None and before.state == "live"

    with Watcher(spawner):
        result = runner.invoke(bot_group, ["restart", config, "--wait", "10"])

    assert result.exit_code == 0, result.output
    assert spawner.calls == 2  # the old bot AND its replacement
    after = spawner.registry.probe(scope)
    assert after is not None and after.state == "live"
    assert after.pid != before.pid
    assert after.started_at > before.started_at


def test_restart_still_live_bot_is_never_replaced_and_exits_1(
    tmp_path: Path, runner: CliRunner, spawner: StopAwareSpawner
) -> None:
    """Pins never-force-kill: a bot ignoring the stop is reported, exit 1.

    No watcher runs, so the fake stays live past the wait. Starting anyway
    would be a second writer on a live book.
    """
    config = write_config(tmp_path / "cfg.json")
    _, scope = load_scope(config)
    assert runner.invoke(bot_group, ["start", config]).exit_code == 0
    before = spawner.registry.probe(scope)
    assert before is not None and before.state == "live"

    try:
        result = runner.invoke(bot_group, ["restart", config, "--wait", "0.3"])
        after = spawner.registry.probe(scope)
    finally:
        drain(spawner)

    assert result.exit_code == 1, result.output
    assert "still live" in result.output
    assert spawner.calls == 1  # no replacement spawned over a live bot
    assert after is not None and after.state == "live" and after.pid == before.pid


def test_restart_dead_bot_is_plain_start(
    tmp_path: Path, runner: CliRunner, spawner: StopAwareSpawner
) -> None:
    """Pins that a restart of an exited bot starts it without a stop first."""
    config = write_config(tmp_path / "cfg.json")
    _, scope = load_scope(config)
    assert runner.invoke(bot_group, ["start", config]).exit_code == 0
    drain(spawner)  # the bot exits on its own
    dead = spawner.registry.probe(scope)
    assert dead is not None and dead.state != "live"

    with Watcher(spawner):
        result = runner.invoke(bot_group, ["restart", config, "--wait", "10"])

    assert result.exit_code == 0, result.output
    live = spawner.registry.probe(scope)
    assert live is not None and live.state == "live"
    assert live.pid != dead.pid


def test_restart_all_covers_every_row_and_isolates_a_missing_config(
    tmp_path: Path, runner: CliRunner, spawner: StopAwareSpawner
) -> None:
    """Pins ``--all`` over registry rows: a vanished config path fails only it."""
    keep = write_config(tmp_path / "keep.json", name="bot_keep")
    gone = write_config(tmp_path / "gone.json", name="bot_gone")
    keep_scope = load_scope(keep)[1]
    gone_scope = load_scope(gone)[1]
    assert keep_scope != gone_scope
    assert runner.invoke(bot_group, ["start", keep, gone]).exit_code == 0
    keep_before = spawner.registry.probe(keep_scope)
    assert keep_before is not None and keep_before.state == "live"
    Path(gone).unlink()  # the recorded config path no longer exists on disk

    with Watcher(spawner):
        result = runner.invoke(bot_group, ["restart", "--all", "--wait", "10"])
        keep_after = spawner.registry.probe(keep_scope)
    drain(spawner)

    assert result.exit_code == 1, result.output
    assert keep_after is not None and keep_after.state == "live"
    assert keep_after.pid != keep_before.pid  # the healthy row WAS restarted
    assert result.output.count("config path missing") == 1  # only the gone row


def test_restart_dry_run_spawns_nothing_and_writes_nothing(
    tmp_path: Path, runner: CliRunner, spawner: StopAwareSpawner
) -> None:
    """Pins dry-run purity: no spawn, no stop flag, no new file anywhere."""
    config = write_config(tmp_path / "cfg.json")
    _, scope = load_scope(config)
    assert runner.invoke(bot_group, ["start", config]).exit_code == 0
    drain(spawner)  # a stopped bot: a dry run must still write nothing
    spawner.registry.stop_flag_path(scope).unlink(missing_ok=True)

    before = {p.name for p in tmp_path.iterdir()}
    result = runner.invoke(bot_group, ["restart", config, "--dry-run"])
    after = {p.name for p in tmp_path.iterdir()}

    assert result.exit_code == 0, result.output
    assert spawner.calls == 1  # only the setup start
    assert before == after
    assert not spawner.registry.stop_flag_path(scope).exists()


# --- logs -------------------------------------------------------------------


@pytest.mark.parametrize(("tail_arg", "expected"), [("2", 2), ("all", 5), ("999", 5)])
def test_logs_tail_window(
    tmp_path: Path, runner: CliRunner, tail_arg: str, expected: int
) -> None:
    """Pins the tail window: N lines, ``all`` and N > size for the whole file."""
    config = write_config(tmp_path / "cfg.json")
    _, scope = load_scope(config)
    seed_log(scope, 5)

    result = runner.invoke(bot_group, ["logs", config, "--tail", tail_arg])

    assert result.exit_code == 0, result.output
    emitted = [line for line in result.output.splitlines() if line]
    assert len(emitted) == expected
    assert emitted[-1] == "cycle exit 0"


@pytest.mark.parametrize("command", ["restart", "logs"])
def test_missing_targets_is_a_usage_error(runner: CliRunner, command: str) -> None:
    """Pins the shared usage contract: no configs and no ``--all`` is exit 2."""
    result = runner.invoke(bot_group, [command])
    assert result.exit_code == 2
    assert "give one or more config paths, or --all" in result.output


def test_logs_missing_log_is_reported_and_fails_only_when_all_miss(
    tmp_path: Path, runner: CliRunner
) -> None:
    """Pins the missing-log rule and that ``--all`` reaches unnamed rows.

    A miss is reported in place; the command fails only when EVERY target
    misses, so one young bot never hides a real log.
    """
    present = write_config(tmp_path / "present.json", name="bot_present")
    absent = write_config(tmp_path / "absent.json", name="bot_absent")
    present_scope = load_scope(present)[1]
    absent_scope = load_scope(absent)[1]
    # Two registry rows the ``--all`` call never names: one with a log, one not.
    seed_row(present_scope, present)
    seed_row(absent_scope, absent)
    seed_log(present_scope, 3)

    partial = runner.invoke(bot_group, ["logs", present, absent])
    all_rows = runner.invoke(bot_group, ["logs", "--all"])
    only_missing = runner.invoke(bot_group, ["logs", absent])

    assert partial.exit_code == 0, partial.output
    assert f"no log for {absent_scope}" in partial.output
    assert "cycle exit 0" in partial.output
    assert all_rows.exit_code == 0, all_rows.output
    assert f"no log for {absent_scope}" in all_rows.output
    assert "cycle exit 0" in all_rows.output
    assert only_missing.exit_code == 1, only_missing.output
    assert f"no log for {absent_scope}" in only_missing.output


def test_logs_follow_emits_appended_lines_and_stops_on_signal(
    tmp_path: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins following at both seams: ``follow`` yields an appended line and a
    truncated log is re-read from zero; the CLI streams two targets, exits 0."""

    def once(write: Callable[[], None]) -> Callable[[], str | None]:
        polls: list[int] = []

        def tick() -> str | None:
            polls.append(1)
            if len(polls) == 1:
                write()
                return ""
            return None

        return tick

    plain = tmp_path / "plain.log"
    plain.write_text("cycle exit 0\n")
    seeded = len(b"cycle exit 0\n")

    def append() -> None:
        with plain.open("a") as handle:
            handle.write("cycle exit 1\n")

    assert list(follow(plain, seeded, tick=once(append))) == ["cycle exit 1"]

    rotated = tmp_path / "rotated.log"
    rotated.write_text("old line one\nold line two\n")

    def truncate() -> None:
        rotated.write_text("new line\n")  # truncated (or rotated) in place

    assert list(follow(rotated, rotated.stat().st_size, tick=once(truncate))) == [
        "new line"
    ]

    first = write_config(tmp_path / "a.json", name="bot_a")
    second = write_config(tmp_path / "b.json", name="bot_b")
    first_log = seed_log(load_scope(first)[1], 1)
    second_log = seed_log(load_scope(second)[1], 1)
    polls: list[int] = []

    def cli_tick() -> str | None:
        polls.append(1)
        if len(polls) == 1:
            first_log.write_text(first_log.read_text() + "cycle exit 3\n")
            second_log.write_text(second_log.read_text() + "cycle exit 3\n")
            return ""
        return None

    monkeypatch.setattr(bot_cli, "sleep_tick", cli_tick)
    result = runner.invoke(
        bot_group, ["logs", first, second, "--follow", "--tail", "1"]
    )

    assert result.exit_code == 0, result.output  # a stop signal is not a failure
    assert result.output.count("cycle exit 3") == 2


def test_logs_mutates_nothing_on_disk(tmp_path: Path, runner: CliRunner) -> None:
    """Pins reader-only: the runtime dir is byte-identical before and after."""
    config = write_config(tmp_path / "cfg.json")
    _, scope = load_scope(config)
    seed_log(scope, 3)
    before = digest_runtime()

    plain = runner.invoke(bot_group, ["logs", config])
    missing = runner.invoke(bot_group, ["logs", "--all", "--tail", "all"])

    assert plain.exit_code == 0, plain.output
    assert missing.exit_code == 0, missing.output
    assert digest_runtime() == before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
