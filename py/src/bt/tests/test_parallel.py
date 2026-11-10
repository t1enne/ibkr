"""Tests for the process-pool runner (`src.bt.parallel`).

Validates that `run_in_processes`:
- broadcasts `init_data` via `pool_init` (readable from the per-worker
  `WORKER_STATE` cache),
- runs `fn(task)` for each task and returns results in input order,
- invokes `on_complete(index, result)` in input order,
- handles the empty-tasks, single-task and sequential (`workers=1`) paths.
- clamps a request to `MAX_WORKERS` and to the task count.

Real-process (workers > 1) equivalence against the backtest engine is covered
by ``test_sweep_pooled_matches_sequential_engine`` in the sweep suite — we
avoid a redundant forkserver spawn here to keep `make check` in budget
(spawn is ~2s per pool). Worker functions must be module-level to pickle by
qualified name across the process boundary.
"""

from __future__ import annotations

import time

import pytest

from src.bt.parallel import (
    MAX_WORKERS,
    WORKER_STATE,
    effective_worker_count,
    run_in_processes,
)


def _echo_worker(task: tuple[int, str]) -> tuple[str, int]:
    """Task = (payload, key); returns (key, payload + shared byte)."""
    byte = WORKER_STATE["byte"]
    return (task[1], task[0] + byte)


def _slow_independent(task: tuple[float, int]) -> int:
    """Sleep by task's first field so completion order != input order."""
    delay, ident = task
    time.sleep(delay)
    return ident


def _tasks() -> list[tuple[int, str]]:
    return [(1, "a"), (2, "b"), (3, "c"), (4, "d")]


def _expected(tasks: list[tuple[int, str]], byte: int) -> list[tuple[str, int]]:
    return [(s, payload + byte) for payload, s in tasks]


def test_worker_state_broadcast_to_clean_worker():
    """A fresh worker (no init_data seen before) reads the broadcast cache."""
    res = run_in_processes(_echo_worker, _tasks(), workers=1, init_data={"byte": 5})
    assert res == _expected(_tasks(), 5)


@pytest.mark.slow  # spawns a process pool (forkserver) — skipped via `-m not slow`.
def test_workers_capped_at_task_count():
    """Requesting more workers than tasks must not error or change results."""
    tasks = [(1, "a"), (2, "b")]  # 2 tasks
    # An oversized request is clamped to the task count -> effectively 2 workers.
    res = run_in_processes(
        _echo_worker, tasks, workers=MAX_WORKERS, init_data={"byte": 0}
    )
    assert res == [("a", 1), ("b", 2)]

    # And equals the sequential result.
    seq = run_in_processes(_echo_worker, tasks, workers=1, init_data={"byte": 0})
    assert res == seq


def test_effective_worker_count_clamps_to_host_budget():
    """The pool never grows past MAX_WORKERS, whatever the caller asks for."""
    assert effective_worker_count(999, 1000) == MAX_WORKERS
    assert effective_worker_count(MAX_WORKERS + 1, 1000) == MAX_WORKERS
    assert effective_worker_count(MAX_WORKERS, MAX_WORKERS) == MAX_WORKERS


def test_effective_worker_count_clamps_to_task_count():
    """Fewer tasks than workers -> exactly one worker per task."""
    assert effective_worker_count(4, 2) == 2
    assert effective_worker_count(4, 1) == 1


def test_effective_worker_count_floor_is_sequential():
    """Zero/negative requests fall back to the sequential loop, never a pool."""
    assert effective_worker_count(0, 10) == 1
    assert effective_worker_count(-3, 10) == 1
