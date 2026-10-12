"""Critical-path tests for bot M1 (registry + spawn), per plan §7.

Only the behaviors that can actually break are pinned here: two concurrent
starts spawn exactly one process (the spawning claim serializes probe-and-
spawn), a released lock reads ``exited``/``stale`` and never ``live`` — the
ONE test allowed to fork a real child, because the primitive cannot be
exercised in-process — a dead child fails ``start`` promptly instead of
stalling the full readiness timeout, a sibling's failed claim never aborts the
good bot, and ``--dry-run`` spawns nothing and writes no registry row.
Rendering and row-shape assertions are deliberately absent (AGENTS.md):

aim: pin the spawn handshake, the OS-lock liveness contract, isolation and
dry-run purity.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from src.bot.cli import StartOutcome, bot_group, load_scope, start_one
from src.bot.registry import BotRegistry
from src.db.path import resolve_live_db_path
from src.bot.spawn import (
    READY_TIMEOUT_S,
    adapter_flag_of,
    ready_fd_of,
    run_config_of,
    wait_ready,
)

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


def write_config(path: Path, *, name: str = "bot_test") -> str:
    """Write a valid live strategy config; return its path."""
    path.write_text(json.dumps({**BASE_CONFIG, "name": name}))
    return str(path)


class FakeChildSpawner:
    """In-process stand-in for a spawned bot.

    Mirrors the real child entry contract exactly — parse the scope from the
    child argv, claim the lifetime lock, write the pid row, then signal the
    ready pipe — but stays in-process: the claimed fd is kept open on this
    object, so the simulated bot remains LIVE (flock hold) until the process
    ends, with zero fork and zero imported trading machinery.
    """

    def __init__(self, registry: BotRegistry | None = None) -> None:
        self.registry = registry
        self.calls = 0
        self.held_fds: list[int] = []

    def spawn(self, argv: Sequence[str], log_path: Path) -> int:
        self.calls += 1
        cfg = run_config_of(argv)
        assert cfg is not None
        registry = (
            self.registry
            if self.registry is not None
            else BotRegistry(resolve_live_db_path())
        )
        _, scope, _ = load_scope(cfg, adapter_flag_of(argv))
        fd = registry.acquire_lifetime(scope)
        self.held_fds.append(fd)
        registry.write_pid_record(
            scope,
            pid=os.getpid(),
            started_at=pd.Timestamp.now(),
            config_path=str(Path(cfg).resolve()),
        )
        ready = ready_fd_of(argv)
        if ready is not None:
            # Write but do NOT close: in-process, closing would also close the
            # parent's copy of the write end (a real child has its own dup).
            os.write(ready, b"1")
        return os.getpid()


class NoReadySpawner:
    """A child that dies without ever writing the ready byte.

    In-process the fake holds no copy of the write end, so the parent's close
    drops the last writer and the ready pipe reads EOF — the same signal a
    dead real child's closed dup produces. The parent must fail promptly on
    that EOF; a parent that only noticed via the full ``READY_TIMEOUT_S``
    would make this test sleep a minute.
    """

    def __init__(self) -> None:
        self.calls = 0

    def spawn(self, argv: Sequence[str], log_path: Path) -> int:
        self.calls += 1
        return os.getpid()


class SelectiveClaimSpawner:
    """Claims the good scope; the bad scope's child dies silently.

    Faithful to the fake-child contract on the good side (lifetime lock, pid
    row, then the ready byte); on the bad side it writes nothing — an
    unclaimed dead child — so the parent's failure comes from EOF, not a
    coordinated byte.
    """

    def __init__(self, registry: BotRegistry) -> None:
        self.registry = registry
        self.calls = 0
        self.held_fds: list[int] = []
        self.failed_scopes: set[str] = set()

    def spawn(self, argv: Sequence[str], log_path: Path) -> int:
        self.calls += 1
        cfg = run_config_of(argv)
        assert cfg is not None
        _, scope, _ = load_scope(cfg, adapter_flag_of(argv))
        if scope in self.failed_scopes:
            return os.getpid()  # die before claiming: never touch the pipe
        fd = self.registry.acquire_lifetime(scope)
        self.held_fds.append(fd)
        self.registry.write_pid_record(
            scope,
            pid=os.getpid(),
            started_at=pd.Timestamp.now(),
            config_path=str(Path(cfg).resolve()),
        )
        ready = ready_fd_of(argv)
        if ready is not None:
            os.write(ready, b"1")
        return os.getpid()


class BlockingChildSpawner(FakeChildSpawner):
    """Fake child that parks INSIDE spawn until released.

    The winner stalls mid-handshake while it still holds the spawning claim,
    so a concurrent loser is GUARANTEED to hit ``BotStartInProgressError`` —
    not a racy post-hoc "already live" — and its ``ok`` must read False.
    """

    def __init__(
        self,
        registry: BotRegistry,
        entered: threading.Event,
        release: threading.Event,
    ) -> None:
        super().__init__(registry)
        self.entered = entered
        self.release = release

    def spawn(self, argv: Sequence[str], log_path: Path) -> int:
        self.entered.set()
        assert self.release.wait(timeout=30), "winner's spawn never released"
        return super().spawn(argv, log_path)


def wait_child_exit(pid: int, timeout_s: float = 15.0) -> int:
    """Reap *pid* and return its exit code; fail the test on a hang."""
    deadline = time.monotonic() + timeout_s
    while True:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            return os.waitstatus_to_exitcode(status)
        assert time.monotonic() < deadline, f"child {pid} did not exit"
        time.sleep(0.05)


# --- 1. concurrent double start spawns a scope once --------------------------


def test_concurrent_double_start_spawns_once(tmp_path: Path) -> None:
    db = tmp_path / "live.db"
    runtime = tmp_path / "runtime"
    config = write_config(tmp_path / "cfg.json")

    registry = BotRegistry(db, runtime)
    spawner = FakeChildSpawner(registry)
    _, scope, _ = load_scope(config, None)

    barrier = threading.Barrier(2)
    outcomes: list[StartOutcome] = []
    errors: list[BaseException] = []

    def run_start() -> None:
        barrier.wait()
        try:
            outcomes.append(start_one(registry, spawner, config, None, False))
        except BaseException as exc:  # surface thread failures instead of hanging
            errors.append(exc)

    threads = [threading.Thread(target=run_start) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    for thread in threads:
        assert not thread.is_alive(), "a concurrent start hung"
    assert not errors

    assert spawner.calls == 1  # exactly ONE process ever spawned for the scope
    spawned = [o for o in outcomes if o.spawned]
    assert len(spawned) == 1 and spawned[0].ok
    assert any(o.ok for o in outcomes)
    assert all(not o.spawned or o.ok for o in outcomes)

    record = registry.probe(scope)
    assert record is not None and record.state == "live"


def test_start_is_idempotent_after_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.db.path.LIVE_DB_PATH", tmp_path / "live.db")
    monkeypatch.setenv("IBKR_BOT_RUNTIME_DIR", str(tmp_path / "runtime"))
    config = write_config(tmp_path / "cfg.json")

    spawner = FakeChildSpawner()
    monkeypatch.setattr("src.bot.cli._build_spawner", lambda: spawner)
    runner = CliRunner()

    first = runner.invoke(bot_group, ["start", config])
    assert first.exit_code == 0, first.output
    assert spawner.calls == 1

    second = runner.invoke(bot_group, ["start", config])
    assert second.exit_code == 0, second.output
    assert spawner.calls == 1  # a second start must NOT spawn a second process


# --- 1b. failed-claim path: exit 1, no live row, and BOUNDED (no 60 s hang) ---


def test_start_failed_claim_exits_nonzero_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.db.path.LIVE_DB_PATH", tmp_path / "live.db")
    monkeypatch.setenv("IBKR_BOT_RUNTIME_DIR", str(tmp_path / "runtime"))
    config = write_config(tmp_path / "cfg.json")
    scope = load_scope(config, None)[1]

    spawner = NoReadySpawner()
    monkeypatch.setattr("src.bot.cli._build_spawner", lambda: spawner)
    runner = CliRunner()

    started = time.monotonic()
    result = runner.invoke(bot_group, ["start", config])
    elapsed = time.monotonic() - started

    assert result.exit_code == 1, result.output
    assert "child failed to claim its scope" in result.output
    assert elapsed < 5.0, f"dead child was not bounded: {elapsed:.1f}s"
    record = BotRegistry(resolve_live_db_path()).probe(scope)
    assert record is None or record.state != "live"


# --- 1c. isolation: one bad sibling never aborts the good bot ---------------


def test_start_isolation_bad_sibling_spares_good_bot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.db.path.LIVE_DB_PATH", tmp_path / "live.db")
    monkeypatch.setenv("IBKR_BOT_RUNTIME_DIR", str(tmp_path / "runtime"))
    good_cfg = write_config(tmp_path / "good.json", name="bot_good")
    bad_cfg = write_config(tmp_path / "bad.json", name="bot_bad")
    good_scope = load_scope(good_cfg, None)[1]
    bad_scope = load_scope(bad_cfg, None)[1]
    assert good_scope != bad_scope

    registry = BotRegistry(resolve_live_db_path())
    spawner = SelectiveClaimSpawner(registry)
    spawner.failed_scopes.add(bad_scope)
    monkeypatch.setattr("src.bot.cli._build_spawner", lambda: spawner)
    runner = CliRunner()

    # The FAILING config first: an aborting supervisor would kill the good bot.
    result = runner.invoke(bot_group, ["start", bad_cfg, good_cfg])

    assert result.exit_code == 1, result.output
    assert spawner.calls == 2  # failure never prevented the sibling spawn
    good = registry.probe(good_scope)
    assert good is not None and good.state == "live"
    bad = registry.probe(bad_scope)
    assert bad is None or bad.state != "live"


# --- 1d. double-start hardening: the refused loser is NOT ok -----------------


def test_double_start_loser_is_ok_false(tmp_path: Path) -> None:
    db = tmp_path / "live.db"
    runtime = tmp_path / "runtime"
    config = write_config(tmp_path / "cfg.json")

    registry = BotRegistry(db, runtime)
    entered = threading.Event()
    release = threading.Event()
    spawner = BlockingChildSpawner(registry, entered, release)
    _, scope, _ = load_scope(config, None)

    outcomes: list[StartOutcome] = []
    errors: list[BaseException] = []

    def run_start() -> None:
        try:
            outcomes.append(start_one(registry, spawner, config, None, False))
        except BaseException as exc:  # surface failures instead of hanging
            errors.append(exc)

    winner = threading.Thread(target=run_start)
    winner.start()
    assert entered.wait(timeout=10), "winner never reached spawn"

    # Second start while the winner still HOLDS the spawning claim: the loser
    # must be refused (ok False); a benign interleaving must never flip it.
    loser = start_one(registry, spawner, config, None, False)
    release.set()
    winner.join(timeout=30)
    assert not winner.is_alive(), "winner hung"
    assert not errors

    assert loser.spawned is False and loser.ok is False
    assert len(outcomes) == 1
    assert outcomes[0].spawned and outcomes[0].ok
    record = registry.probe(scope)
    assert record is not None and record.state == "live"


# --- 2. a released lock (dead child) reads exited/stale, never live ----------


def _spawn_lock_holding_child(
    registry: BotRegistry, scope: str, config_path: str
) -> tuple[int, int]:
    """Fork a REAL child that holds the scope lock and signals readiness.

    The loop itself would need network/sync to reach its sleep point, so this
    test forges a minimal child that replicates only the identity contract the
    fork test exists to pin: claim the lifetime lock, write the pid row, signal
    the ready pipe, then park on the stop flag. On sighting it the child stamps
    its own exit 0 (mirroring ``loop.run_loop``) and exits, releasing the lock.
    Returns ``(child_pid, read_fd)``; the child's ready byte is readable on
    *read_fd*.
    """
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            registry.acquire_lifetime(scope)
            started_at = pd.Timestamp.now()
            registry.write_pid_record(
                scope,
                pid=os.getpid(),
                started_at=started_at,
                config_path=str(Path(config_path).resolve()),
            )
            os.write(write_fd, b"1")
            os.close(write_fd)
            flag = registry.stop_flag_path(scope)
            while not flag.exists():
                time.sleep(0.02)
            registry.record_exit_code(scope, 0, pid=os.getpid(), started_at=started_at)
        finally:
            os._exit(0)
    os.close(write_fd)
    return pid, read_fd


def test_released_lock_reads_exited_and_stale(tmp_path: Path) -> None:
    db = tmp_path / "live.db"
    runtime = tmp_path / "runtime"
    registry = BotRegistry(db, runtime)
    config = write_config(tmp_path / "cfg.json")
    _, scope, _ = load_scope(config, None)

    # clean stop: child sees the stop flag, stamps exit 0, releases the lock
    pid, read_fd = _spawn_lock_holding_child(registry, scope, config)
    assert wait_ready(read_fd, READY_TIMEOUT_S), "child never signalled readiness"
    os.close(read_fd)

    record = registry.probe(scope)
    assert record is not None and record.state == "live" and record.pid == pid

    registry.stop_flag_path(scope).write_text("")
    assert wait_child_exit(pid) == 0
    record = registry.probe(scope)
    assert record is not None
    assert record.state == "exited"
    assert record.exit_code == 0

    # crash: SIGKILL drops the flock, no exit code is ever recorded -> stale
    registry.stop_flag_path(scope).unlink()
    pid2, read_fd2 = _spawn_lock_holding_child(registry, scope, config)
    assert wait_ready(read_fd2, READY_TIMEOUT_S), (
        "second child never signalled readiness"
    )
    os.close(read_fd2)

    record = registry.probe(scope)
    assert record is not None and record.state == "live"  # alive right after the claim

    os.kill(pid2, signal.SIGKILL)
    wait_child_exit(pid2)
    record = registry.probe(scope)
    assert record is not None
    assert record.state == "stale"
    assert record.exit_code is None


# --- 6. --dry-run spawns nothing and writes no registry row ------------------


def test_start_dry_run_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.db.path.LIVE_DB_PATH", tmp_path / "live.db")
    monkeypatch.setenv("IBKR_BOT_RUNTIME_DIR", str(tmp_path / "runtime"))
    config = write_config(tmp_path / "cfg.json")

    spawner = FakeChildSpawner()
    monkeypatch.setattr("src.bot.cli._build_spawner", lambda: spawner)
    runner = CliRunner()
    before = {p.name for p in tmp_path.iterdir()}

    result = runner.invoke(bot_group, ["start", config, "--dry-run"])

    assert result.exit_code == 0, result.output
    assert spawner.calls == 0
    after = {p.name for p in tmp_path.iterdir()}
    assert before == after  # no pid row, no lock file, no log: the dir did not move
