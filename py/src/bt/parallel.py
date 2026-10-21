"""Small process-pool runner for parallel sweep / split / optimize.

Motivation: ``run_sweep``, ``run_split`` and ``run_optimize`` are naturally
parallel — sweep across grid combos, the IS/OOS window pair per fold. A
**process** pool (not a thread pool) is the right tool because backtests are
CPU-bound and the backtest engine is pure over immutable inputs, so there is
no shared-memory state to contend on.

How it works
------------
The backtest engine is functionally pure: given a ``StrategyConfig``, a sliced
candle feed and (for DSL strategies) freshly-minted per-run ``ctx.shared``
state, a run is fully deterministic. Process workers get that isolation for
free — each worker is its own interpreter, so nothing bleeds across units of
work (and separate workers never share the same candle feed cursor).

The one expensive thing to move across the process boundary is a pandas
DataFrame feed. To avoid re-pickling the same feed for every task, the pool
bootstraps each worker once with ``pool_init``: shared read-only inputs (the
feed(s), the base config) are stored in a per-worker ``WORKER_STATE`` dict.
Tasks then carry only small per-unit args and reference the cache by key.

``run_in_processes`` submits ``fn(task)`` per task, maps results back to input
order, and invokes ``on_complete(index, result)`` as soon as each task's
result is "unblocked" (all earlier tasks done) so callers can stream output in
a deterministic order even though work completes out of order. ``workers <= 1``
falls back to a plain sequential loop — preserving all existing behaviour and
programmatic/tests paths that never pass ``workers``.

Cost model — throughput, not correctness
----------------------------------------
Every count is identical across worker counts: the engine is a pure function of
its inputs, so pooling changes only wall-clock time. Two costs bound the useful
worker count:

* **Fixed spawn cost (~2s per pooled run)** — a ``forkserver`` plus workers has
to start before any task runs, so a pool only pays when the units it
parallelizes are expensive by comparison. A few combos/folds on a small feed
are slower pooled than sequential; reach for ``workers > 1`` once a single run
would take more than a few seconds (many combos, many folds, heavy intraday
feeds).
* **Memory scales with worker count** — ``pool_init`` gives each worker its own
copy of the shared read-only inputs (notably the candle feed), so peak RSS
rises roughly with ``workers`` on top of the parent process. There is no fixed
per-worker figure: measure actual RSS before raising ``workers`` on a large
universe, and note that available RAM on a given host varies between runs, so
size from a fresh measurement rather than a remembered number.

``effective_workers`` is capped at the task count, so an oversized ``workers``
never spawns idle processes — it only wastes spawn time.

Worker functions must be *module-level* (picklable by qualified name) and read
shared inputs from ``WORKER_STATE``.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable

# Per-worker shared cache. Populated once per worker by ``pool_init``; read by
# module-level worker functions. Keeps shared feeds/config off the task queue
# (pickled once per worker, not once per task).
WORKER_STATE: dict[str, Any] = {}


def pool_init(state: dict[str, Any]) -> None:
    """Fill this worker's shared cache with read-only bootstrapping data.

    Runs once per worker inside the process pool before any task executes.
    ``state`` is shared across all of the worker's tasks.
    """
    WORKER_STATE.clear()
    WORKER_STATE.update(state)


def run_in_processes(
    fn: Callable[[Any], Any],
    tasks: list[Any],
    *,
    workers: int,
    init_data: dict[str, Any] | None = None,
    on_complete: Callable[[int, Any], None] | None = None,
) -> list[Any]:
    """Run ``fn(task)`` for each task, returning results in input order.

    Args:
        fn: module-level, picklable worker that reads shared inputs from
            ``WORKER_STATE`` and returns a picklable result.
        tasks: one payload per unit of work (parallelism granularity).
        workers: max worker processes (capped at the task count so no pool is
            ever started with more workers than units of work). ``<= 1`` runs
            sequentially (no pool).
        init_data: shared read-only inputs broadcast to every worker once
            (feeds, base config, ...). Empty by default.
        on_complete: optional callback ``(index, result)`` invoked in input
            order once each unit's result is available (streaming).

    Returns:
        Results in the same order as ``tasks``. On a worker failure the
        exception propagates from the first failing unit.
    """
    n = len(tasks)
    if n == 0:
        return []

    # Never spawn more processes than there are tasks — a capped pool of N
    # workers running N jobs gets no further speedup from idle extra workers.
    effective_workers = max(1, min(workers, n))

    if effective_workers <= 1 or n == 1:
        # Sequential fallback: mirror the worker cache contract so the same
        # module-level ``fn`` works identically whether pooled or not.
        if init_data:
            pool_init(init_data)
        results: list[Any] = []
        for i, task in enumerate(tasks):
            result = fn(task)
            results.append(result)
            if on_complete is not None:
                on_complete(i, result)
        return results

    results: list[Any] = [None] * n
    done_map: dict[int, Any] = {}
    next_idx = 0

    def _flush() -> None:
        nonlocal next_idx
        while next_idx in done_map:
            result = done_map.pop(next_idx)
            if on_complete is not None:
                on_complete(next_idx, result)
            next_idx += 1

    with ProcessPoolExecutor(
        max_workers=effective_workers,
        initializer=pool_init,
        initargs=(init_data or {},),
    ) as executor:
        index_by_future: dict[Any, int] = {
            executor.submit(fn, task): i for i, task in enumerate(tasks)
        }
        for future in as_completed(index_by_future):
            idx = index_by_future[future]
            results[idx] = future.result()
            done_map[idx] = results[idx]
            _flush()

    return results


__all__ = [
    "WORKER_STATE",
    "pool_init",
    "run_in_processes",
]
