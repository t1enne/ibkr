"""The bot loop — sync, cycle, sleep, repeat (plan §5).

One DETACHED resident process per config: it claims its scope lock for life,
then loops ``sync -> cycle -> sleep``. Every step fails closed — most
importantly, a failed sync never reaches the cycle, so a bot can never trade a
stale tail.

Seams keep the loop testable without sleeping, spawning, downloading or
reaching IBKR: ``SleepFn`` (the stop flag is polled only BETWEEN sleep chunks,
never mid-cycle), ``SyncFn`` (the candle refresh) and ``RunCycleFn`` (one live
cycle, returning its exit code). Retry policy: ``0`` continues, ``1`` is
retried up to a bounded count, ``2``/``3`` stop the bot immediately (``3`` needs
an operator). One bot's exit never aborts its siblings.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import pandas as pd
from click import echo

from src.live.types import LiveConfig
from src.bot.registry import BotRegistry
from src.bot.sync import SyncFn, sync_cfg

#: Exit code for a transient failure (stale data, gateway, config) — retried.
EXIT_TRANSIENT = 1

#: How long one sleep chunk runs before the stop flag is re-read. Bounds how
#: long ``bot stop`` must wait past the current cycle to be observed.
_STOP_POLL_SECONDS = 5.0

#: Seconds in one day — the unit the cadence and staleness defaults share.
_SECONDS_PER_DAY = 86_400

#: ``"1h"``/``"30m"``/``"1d"``/``"1w"`` with a LOWER-CASE unit; the case is
#: matched as-is so ``"1M"``/``"1H"`` fail closed instead of silently resolving
#: to a minute/hour cadence.
_BAR_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([a-z]+)\s*$")

#: Calendar seconds per bar unit (``m`` = minute; upper-case units are errors).
_UNIT_SECONDS = {"m": 60.0, "h": 3_600.0, "d": 86_400.0, "w": 604_800.0}


class SleepFn(Protocol):
    """The cadence sleep; tests fake it, production uses ``time.sleep``."""

    def __call__(self, seconds: float) -> None: ...


class RunCycleFn(Protocol):
    """One live cycle for a config; returns its machine exit code (§6)."""

    def __call__(self, config_path: str, max_age_days: int) -> int: ...


@dataclass(frozen=True)
class BotOptions:
    """The cadence/retry knobs ``bot start`` passes to its child (§2)."""

    interval: int | None = None  # --interval; None => derive from bars[0]
    max_age: int | None = None  # --max-age; None => 2x cadence, clamped >= 1
    no_sync: bool = False
    retries: int = 1


@dataclass(frozen=True)
class Cadence:
    """The derived cadence: seconds to sleep and days before data is stale."""

    interval_s: int
    max_age_days: int


@dataclass(frozen=True)
class _Life:
    """One bot's identity plus the injected seams — everything ``_tick`` needs.

    ``pid`` and ``started_at`` are the identity proof ``record_exit_code``
    demands, so a per-cycle stamp can never clobber a newer life's row.
    """

    config_path: str
    scope: str
    pid: int
    started_at: pd.Timestamp
    cfg: LiveConfig
    options: BotOptions
    cadence: Cadence
    run_cycle: RunCycleFn
    sync: SyncFn
    sleep: SleepFn
    today: Callable[[], date]


def bar_seconds(bar: str) -> int:
    """Calendar seconds one *bar* spans (``"1d"`` -> a day, ``"1h"`` -> an hour).

    The cadence default: a daily strategy runs daily. A bar size with no
    calendar unit is a config error rather than a silent guess, and so is a
    ZERO span (``"0m"``): it would resolve to a zero-second cadence and a hot
    loop that never sleeps (M2 residual).
    """
    match = _BAR_RE.match(bar)
    if match is None or match.group(2) not in _UNIT_SECONDS:
        raise ValueError(f"cannot derive a cadence from bar {bar!r}")
    seconds = int(float(match.group(1)) * _UNIT_SECONDS[match.group(2)])
    if seconds <= 0:
        raise ValueError(f"bar {bar!r} spans no time; a cadence must be positive")
    return seconds


def derive_cadence(bar: str, *, interval: int | None, max_age: int | None) -> Cadence:
    """The loop's cadence from ``bars[0]`` and any explicit overrides (§2).

    Default interval is one bar. Default ``max_age_days`` is twice that cadence
    in days, clamped ``>= 1``: a weekend must not trip the gate, and ``<= 0``
    can never silently disable it (D14). An explicit flag always wins.
    """
    default_interval = bar_seconds(bar)
    interval_s = interval if interval is not None else default_interval
    # An explicit --interval of 0 (or less) is a hot loop with no sleep, not a
    # cadence: reject it rather than busy-cycle against the gateway.
    if interval_s <= 0:
        raise ValueError(f"interval {interval_s} is not a positive cadence")
    if max_age is not None:
        return Cadence(interval_s, max_age)
    cadence_days = default_interval / _SECONDS_PER_DAY
    return Cadence(interval_s, max(1, int(2 * cadence_days)))


def _default_sleep(seconds: float) -> None:
    """Production sleep — the one place the loop touches the wall clock."""
    time.sleep(seconds)


def run_loop(
    config_path: str,
    *,
    cfg: LiveConfig,
    scope: str,
    registry: BotRegistry,
    options: BotOptions,
    run_cycle: RunCycleFn,
    sync: SyncFn = sync_cfg,
    sleep: SleepFn = _default_sleep,
    today: Callable[[], date] = date.today,
    on_ready: Callable[[], None] | None = None,
) -> int:
    """The resident loop for one bot; returns the process exit code (§5).

    Derive the cadence, claim the scope lock for life, then loop
    ``sync -> cycle -> record -> sleep``. *on_ready* fires once the lock and pid
    row exist — the spawn handshake's ready signal (§4.1).
    """
    cadence = derive_cadence(
        cfg.bars[0], interval=options.interval, max_age=options.max_age
    )
    with registry.hold_lifetime(scope):
        pid = os.getpid()
        started_at = pd.Timestamp.now()
        registry.write_pid_record(
            scope, pid=pid, started_at=started_at, config_path=config_path
        )
        if on_ready is not None:
            on_ready()
        life = _Life(
            config_path=config_path,
            scope=scope,
            pid=pid,
            started_at=started_at,
            cfg=cfg,
            options=options,
            cadence=cadence,
            run_cycle=run_cycle,
            sync=sync,
            sleep=sleep,
            today=today,
        )
        code = _loop_body(registry, life)
        registry.record_exit_code(scope, code, pid=pid, started_at=started_at)
        return code


def _loop_body(registry: BotRegistry, life: _Life) -> int:
    """``sync -> cycle -> stamp -> sleep`` until a stop flag or terminal code.

    The exit code lands on the pid row EVERY cycle, not only at exit, so ``bot
    status`` reads a live bot's most recent outcome. ``1`` is retried up to the
    bound; ``2``/``3`` are terminal.
    """
    failures = 0
    while True:
        code = _tick(life)
        _log(registry, life.scope, code)
        registry.record_exit_code(
            life.scope, code, pid=life.pid, started_at=life.started_at
        )
        if code == 0:
            failures = 0
        elif code == EXIT_TRANSIENT:
            failures += 1
            if failures > life.options.retries:
                return code
        else:
            return code
        if _sleep_until_stop(registry, life):
            return 0


def _tick(life: _Life) -> int:
    """One sync-then-cycle pass; a failed sync returns 1 WITHOUT the cycle."""
    if not life.options.no_sync:
        try:
            life.sync(life.cfg, today=life.today())
        except Exception as exc:  # sync raise/timeout => exit 1, never a cycle
            echo(f"sync failed: {exc}", err=True)
            return EXIT_TRANSIENT
    return life.run_cycle(life.config_path, life.cadence.max_age_days)


def _sleep_until_stop(registry: BotRegistry, life: _Life) -> bool:
    """Sleep one interval in chunks; return True when a stop flag was seen.

    The interval is split so ``bot stop``'s bounded await can observe the flag.
    The flag is read ONLY between chunks, never inside a cycle, so no live cycle
    is interrupted between placement and ``_persist``.
    """
    remaining = float(life.cadence.interval_s)
    while remaining > 0.0:
        life.sleep(min(remaining, _STOP_POLL_SECONDS))
        if registry.stop_flag_path(life.scope).exists():
            return True
        remaining -= _STOP_POLL_SECONDS
    return False


def _log(registry: BotRegistry, scope: str, code: int) -> None:
    """Append one cycle's exit code to the bot's log (registry owns the path)."""
    try:
        with registry.log_path(scope).open("a") as handle:
            handle.write(f"cycle exit {code}\n")
    except OSError:
        # A log we cannot write must not stop the bot: the pid row records the
        # same code and liveness is the lock, not the log.
        echo(f"bot {scope}: cycle exit {code}", err=True)


__all__ = [
    "EXIT_TRANSIENT",
    "BotOptions",
    "Cadence",
    "RunCycleFn",
    "SleepFn",
    "bar_seconds",
    "derive_cadence",
    "run_loop",
]
