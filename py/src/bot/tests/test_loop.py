"""Critical-path tests for bot M2 (the loop), per plan §7.

Faked seams only — no real sleeps, network, IBKR or fork. What is pinned is
the behavior that can silently regress:

* §7.4 ORDERING: a sync raise must never reach the cycle. The exit code alone
  cannot prove this — a cycle-call count of 0 does.
* §7.3: ``stop`` writes the flag and awaits, and NEVER signals a live process.
* §7.5: one bot's non-zero exit leaves its siblings' records untouched.
* §5/§6 retry policy: exit 1 retries up to the bound then exits 1; 2 and 3 are
  terminal (the cycle is called exactly once).
* §2 cadence: default interval from ``bars[0]``, default ``max_age_days`` =
  twice the cadence clamped ``>= 1``, and an explicit flag always wins.

aim: pin the loop's ordering, retry bounds, stop semantics and cadence defaults.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from src.bot.cli import bot_group, load_scope
from src.bot.loop import BotOptions, Cadence, derive_cadence, run_loop
from src.bot.registry import BotRegistry
from src.bot.stop import StopOutcome, stop_one
from src.live.types import LiveConfig

BASE_CONFIG = {
    "name": "bot_loop",
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


def write_config(path: Path, *, name: str = "bot_loop") -> str:
    """Write a valid live strategy config; return its path."""
    path.write_text(json.dumps({**BASE_CONFIG, "name": name}))
    return str(path)


def _mini_cfg(*, bars: str = "1d", warmup: str = "300d") -> LiveConfig:
    """A minimal LiveConfig: only the cadence-relevant fields exercise anything."""
    return LiveConfig(
        strategy_type="vwatr_div_dsl",
        symbols=("AAPL",),
        initial_capital=50_000.0,
        strategy_params={},
        bars=(bars,),
        warmup=warmup,
    )


class CycleCounter:
    """The ``RunCycleFn`` seam: records each call and replays scripted codes."""

    def __init__(self, codes: Sequence[int]) -> None:
        self.codes = list(codes)
        self.calls = 0

    def __call__(self, config_path: str, max_age_days: int) -> int:
        self.calls += 1
        index = self.calls - 1
        return self.codes[index] if index < len(self.codes) else self.codes[-1]


class RaisingSync:
    """The ``SyncFn`` seam: always raises, so no cycle may ever be reached."""

    def __call__(self, cfg: LiveConfig, *, today) -> None:
        raise RuntimeError("sync blew up")


class CountingSync:
    """The ``SyncFn`` seam: records calls and succeeds."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, cfg: LiveConfig, *, today) -> None:
        self.calls += 1


class StopOnSleep:
    """The ``SleepFn`` seam: writes the stop flag, so the loop exits at its
    sleep point (the ONLY place the flag is checked — §5.7)."""

    def __init__(self, flag: Path) -> None:
        self.flag = flag
        self.slept = 0

    def __call__(self, seconds: float) -> None:
        self.slept += 1
        self.flag.write_text("")


def noop_sleep(seconds: float) -> None:
    """The ``SleepFn`` seam that never sleeps and never writes the flag."""
    del seconds


class ProbeOnSleep:
    """The ``SleepFn`` seam: samples the pid row DURING the sleep window.

    The sleep is where the loop is committed to a code that is already stamped
    (§5.6), so a sample taken here proves the row read a live bot's last cycle
    code — not a value only written once, after the loop returned.
    """

    def __init__(self, registry: BotRegistry, scope: str, stop: Path) -> None:
        self.registry = registry
        self.scope = scope
        self.stop = stop
        self.sampled: list[int | None] = []

    def __call__(self, seconds: float) -> None:
        del seconds
        row = self.registry.probe(self.scope)
        assert row is not None
        self.sampled.append(row.exit_code)
        self.stop.write_text("")  # exit at the first sleep point


def _registry(tmp_path: Path) -> BotRegistry:
    return BotRegistry(tmp_path / "live.db", tmp_path / "runtime")


# --- §7.4 ordering: a sync raise never reaches the cycle ---------------------


def test_sync_raise_never_invokes_cycle(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    scope = "sim_bot_loop_deadbeef"
    cycle = CycleCounter([0])

    code = run_loop(
        "/cfg.json",
        cfg=_mini_cfg(),
        scope=scope,
        registry=registry,
        options=BotOptions(retries=0),
        run_cycle=cycle,
        sync=RaisingSync(),
        sleep=noop_sleep,
    )

    assert cycle.calls == 0  # the cycle is NEVER invoked on a sync raise (ORDER)
    assert code == 1  # and the failure is transient, so a scheduler can retry


def test_sync_timeout_never_invokes_cycle_and_exits_1(tmp_path: Path) -> None:
    """A sync TIMEOUT is the same fail-closed path as a raise: no cycle, exit 1.

    ``sync_cfg`` bounds ``download`` with ``asyncio.wait_for``, which surfaces a
    ``TimeoutError`` — a distinct transient failure that must not be allowed to
    fall through to the cycle, and must map to exit 1 (retryable), not 3.
    """
    registry = _registry(tmp_path)
    cycle = CycleCounter([0])

    class TimingOutSync:
        def __call__(self, cfg: LiveConfig, *, today: object) -> None:
            del cfg, today
            raise TimeoutError("download exceeded its deadline")

    code = run_loop(
        "/cfg.json",
        cfg=_mini_cfg(),
        scope="sim_bot_loop_deadbeef",
        registry=registry,
        options=BotOptions(retries=0),
        run_cycle=cycle,
        sync=TimingOutSync(),
        sleep=noop_sleep,
    )

    assert cycle.calls == 0  # a timed-out sync never reaches the cycle
    assert code == 1  # transient: a scheduler may retry


# --- §5/§6 retry policy ------------------------------------------------------


def test_exit_1_retries_up_to_bound_then_exits_1(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    scope = "sim_bot_loop_deadbeef"
    cycle = CycleCounter([1])  # always transient

    # A no-op sleeper: the flag is never written, so the loop runs to its retry
    # bound instead of exiting clean at the first sleep (§5.7 checks it only there).
    code = run_loop(
        "/cfg.json",
        cfg=_mini_cfg(),
        scope=scope,
        registry=registry,
        options=BotOptions(retries=2),
        run_cycle=cycle,
        sync=CountingSync(),
        sleep=noop_sleep,
    )

    assert code == 1
    assert cycle.calls == 3  # first try + 2 retries, then the bound bites


@pytest.mark.parametrize("terminal", [2, 3])
def test_terminal_exit_returns_immediately(tmp_path: Path, terminal: int) -> None:
    registry = _registry(tmp_path)
    scope = "sim_bot_loop_deadbeef"
    cycle = CycleCounter([terminal])
    flag = registry.stop_flag_path(scope)

    code = run_loop(
        "/cfg.json",
        cfg=_mini_cfg(),
        scope=scope,
        registry=registry,
        options=BotOptions(retries=5),
        run_cycle=cycle,
        sync=CountingSync(),
        sleep=StopOnSleep(flag),
    )

    assert code == terminal
    assert cycle.calls == 1  # 2 and 3 are terminal: exactly one cycle, no retry


# --- §5.6 per-cycle record stamping (status must see a LIVE bot's code) ------


def test_each_cycle_stamps_the_pid_row_while_live(tmp_path: Path) -> None:
    """A live bot's pid row carries its last cycle code, not ``None``.

    Without a per-cycle stamp the row reads ``exit_code=None`` for the whole
    life, making ``bot status``' severity provably inert. The sample is taken
    INSIDE the sleep window, when the bot is still live — the exact moment
    ``status`` would probe it.
    """
    registry = _registry(tmp_path)
    scope = "sim_bot_loop_deadbeef"
    sleeper = ProbeOnSleep(registry, scope, registry.stop_flag_path(scope))

    code = run_loop(
        "/cfg.json",
        cfg=_mini_cfg(),
        scope=scope,
        registry=registry,
        options=BotOptions(retries=3),
        run_cycle=CycleCounter([1]),  # transient: retry, do not exit
        sync=CountingSync(),
        sleep=sleeper,
    )

    assert code == 0  # the stop flag ended it cleanly
    assert sleeper.sampled == [1]  # the LIVE row already held this cycle's code
    final = registry.probe(scope)
    assert final is not None
    assert final.exit_code == 0  # and the clean stop stamps 0 at exit


# --- §7.5 isolation: a sibling's exit never touches another's record ---------


def test_sibling_exit_leaves_other_record_untouched(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    sibling, victim = "sim_bot_a_11111111", "sim_bot_b_22222222"
    started = pd.Timestamp.now()
    for scope in (sibling, victim):
        registry.write_pid_record(
            scope, pid=4242, started_at=started, config_path=f"/{scope}.json"
        )

    # A terminal (unsafe) cycle for the sibling: the bot dies on its own.
    code = run_loop(
        f"/{sibling}.json",
        cfg=_mini_cfg(),
        scope=sibling,
        registry=registry,
        options=BotOptions(),
        run_cycle=CycleCounter([3]),
        sync=CountingSync(),
        sleep=noop_sleep,
    )
    assert code == 3

    # The sibling's OWN row DID take the code (per-cycle stamping, §5.6) — this
    # is what makes the untouched victim below a real isolation claim, not a
    # tautology: stamping happened, and it landed on exactly one row.
    sibling_row = registry.probe(sibling)
    assert sibling_row is not None
    assert sibling_row.exit_code == 3

    victim_row = registry.probe(victim)
    assert victim_row is not None
    assert victim_row.exit_code is None  # the sibling's death did not stamp it
    assert victim_row.pid == 4242
    assert victim_row.state == "stale"  # never claimed, so never live


# --- §7.3 stop: writes the flag, awaits, and NEVER signals -------------------


def test_stop_writes_flag_and_never_signals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path)
    scope = "sim_bot_loop_deadbeef"
    registry.write_pid_record(
        scope,
        pid=4242,
        started_at=pd.Timestamp.now(),
        config_path="/cfg.json",
    )

    # Trip ANY attempt to signal a live process: escalation is manual (§4).
    def _forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("bot stop must NEVER signal a live process")

    monkeypatch.setattr("os.kill", _forbidden)
    monkeypatch.setattr("os.killpg", _forbidden)
    monkeypatch.setattr("signal.raise_signal", _forbidden)

    # Nothing holds the lock (the row was written alone), so the await clears
    # at once — but the flag MUST have been written first.
    outcome = stop_one(registry, scope, wait=True, timeout_s=5.0)

    assert isinstance(outcome, StopOutcome)
    assert registry.stop_flag_path(scope).exists()
    assert outcome.stopped and not outcome.still_live


def test_stop_reports_still_live_on_timeout(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    scope = "sim_bot_loop_deadbeef"
    fd = registry.acquire_lifetime(scope)  # simulate a live bot holding the lock
    try:
        registry.write_pid_record(
            scope,
            pid=4242,
            started_at=pd.Timestamp.now(),
            config_path="/cfg.json",
        )
        outcome = stop_one(registry, scope, wait=True, timeout_s=0.3)
    finally:
        import os

        os.close(fd)  # release: the process-exit equivalent

    assert registry.stop_flag_path(scope).exists()
    assert outcome.still_live and not outcome.stopped


def test_stop_cli_writes_flag_while_the_lock_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``stop --no-wait`` against a genuinely-live bot writes the flag and returns.

    The bot HOLDS the scope lock, so a ``stop`` that tried to take that lock
    would raise (``claim``/``hold_lifetime`` refuse an already-held lock) and
    the invoke would fail — the assertion is a real regression guard, not a
    tautology. ``--no-wait`` must also not block on the process disappearing.
    """
    monkeypatch.setattr("src.db.path.LIVE_DB_PATH", tmp_path / "live.db")
    monkeypatch.setenv("IBKR_BOT_RUNTIME_DIR", str(tmp_path / "runtime"))
    config = write_config(tmp_path / "cfg.json")
    scope = load_scope(config, None)[1]

    registry = BotRegistry(tmp_path / "live.db", tmp_path / "runtime")
    registry.write_pid_record(
        scope, pid=4242, started_at=pd.Timestamp.now(), config_path=config
    )
    fd = registry.acquire_lifetime(scope)  # the bot holds the lock for life
    try:
        result = CliRunner().invoke(bot_group, ["stop", config, "--no-wait"])
    finally:
        import os

        os.close(fd)

    assert result.exit_code == 0, result.output
    assert registry.stop_flag_path(scope).exists()


# --- §2 cadence defaults -----------------------------------------------------


def test_default_interval_from_first_bar() -> None:
    cadence = derive_cadence("1d", interval=None, max_age=None)
    assert isinstance(cadence, Cadence)
    assert cadence.interval_s == 86_400
    assert cadence.max_age_days == 2  # 2 x a 1-day cadence, clamped >= 1


def test_default_max_age_clamped_to_one_for_sub_daily() -> None:
    # An hourly cadence rounds 2 x (1/24) down to 0, which would DISABLE the gate
    # silently (D14) — so it clamps to 1.
    assert derive_cadence("1h", interval=None, max_age=None).max_age_days == 1
    assert derive_cadence("1m", interval=None, max_age=None).max_age_days == 1


def test_explicit_flags_win_over_defaults() -> None:
    cadence = derive_cadence("1d", interval=60, max_age=9)
    assert cadence.interval_s == 60
    assert cadence.max_age_days == 9


def test_explicit_max_age_wins_even_below_cadence() -> None:
    cadence = derive_cadence("1w", interval=3600, max_age=3)
    assert cadence.max_age_days == 3  # an explicit flag is never re-clamped


def test_cadence_bar_without_calendar_unit_is_an_error() -> None:
    with pytest.raises(ValueError):
        derive_cadence("tick", interval=None, max_age=None)


def test_uppercase_unit_is_rejected_not_silently_minutes() -> None:
    # "1M" would resolve to a MINUTE if the unit were case-folded: a silently
    # wrong cadence. It must fail closed instead (upper-case units are errors).
    with pytest.raises(ValueError):
        derive_cadence("1M", interval=None, max_age=None)


def test_zero_span_bar_is_rejected() -> None:
    # "0m" yields interval_s = 0 and a hot loop with no sleep (M2 residual).
    with pytest.raises(ValueError):
        derive_cadence("0m", interval=None, max_age=None)


def test_non_positive_explicit_interval_is_rejected() -> None:
    # --interval 0 (or negative) must fail closed, not busy-cycle.
    with pytest.raises(ValueError):
        derive_cadence("1d", interval=0, max_age=None)
    with pytest.raises(ValueError):
        derive_cadence("1d", interval=-5, max_age=None)
