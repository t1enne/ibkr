"""Bot CLI — ``ibkr bot start`` / ``status`` (unattended supervisors, §2).

This layer commands the registry only and exits promptly. ``start`` is
idempotent — an already-live scope is reported, never respawned — and takes the
short-lived spawning claim so two concurrent invocations cannot both spawn. The
parent reports ``live`` only after the child signals it holds its scope lock,
and exits non-zero if any child failed to claim. ``_run`` is the hidden child
entry: claim, signal, then run the resident loop (§5).
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

import click

from src.config import BotConfig, read_bot_config
from src.db.path import resolve_live_db_path
from src.live.adapter import resolve_adapter_name
from src.live.lease import CycleInProgressError
from src.live.ledger import LedgerReadError
from src.live.pipeline import PortfolioFetchError
from src.live.freshness import StaleDataError
from src.live.run_once import load_live_config, run_once
from src.live.scope import (
    AdapterName,
    ScopeParts,
    config_hash as config_scope_hash,
    config_name_of,
    scope_of,
)
from src.live.types import LiveConfig

from src.bot.liveness import BotAlreadyRunningError, BotStartInProgressError
from src.bot.loop import EXIT_TRANSIENT, BotOptions, RunCycleFn, run_loop
from src.bot.registry import BotRegistry
from src.bot.spawn import (
    READY_TIMEOUT_S,
    PopenSpawner,
    child_argv,
    wait_ready,
)
from src.bot.stop import STOP_WAIT_S, StopOutcome, stop_one
from src.bot.types import BotLiveness, BotRecord, BotSpawner

#: The adapters ``--adapter`` accepts, derived from the ONE typing literal.
_ADAPTERS = get_args(AdapterName)


@dataclass(frozen=True)
class StartOutcome:
    """One config's start result.

    ``ok`` means a bot is live for that scope afterwards (spawned and claimed,
    or already live); ``spawned`` says whether THIS invocation started it.
    """

    config_path: str
    scope: str
    ok: bool
    spawned: bool
    message: str


@click.group(name="bot")
def bot_group() -> None:
    """Unattended trading supervisors — start detached bots, report liveness."""


@bot_group.command("start")
@click.argument(
    "configs",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the plan; spawn nothing, write nothing.",
)
@click.option(
    "--adapter",
    type=click.Choice(_ADAPTERS),
    default=None,
    help="Broker adapter to run through. Default: the config chooses (live run precedence).",
)
@click.option(
    "--interval",
    "-i",
    type=int,
    default=None,
    help="Cadence in seconds. Default: one bar of the config's `bars[0]`.",
)
@click.option(
    "--max-age",
    "-a",
    type=int,
    default=None,
    help="Staleness gate in days. Default: twice the cadence, clamped to >= 1.",
)
@click.option("--no-sync", is_flag=True, help="Skip the pre-cycle candle sync.")
@click.option(
    "--retries",
    type=int,
    default=None,
    help="Transient (exit 1) cycles to retry before exiting. Default: 1.",
)
@click.option(
    "--format", "-F", "fmt", type=click.Choice(["text", "json"]), default="text"
)
def bot_start(
    configs: tuple[str, ...],
    dry_run: bool,
    adapter: str | None,
    interval: int | None,
    max_age: int | None,
    no_sync: bool,
    retries: int | None,
    fmt: str,
) -> None:
    """Spawn one detached bot per config; exit 0 when every child claimed.

    Idempotent: a scope that is already live is reported, never respawned.
    """
    cfg = read_bot_config()
    options = _merge_bot_options(interval, max_age, no_sync, retries, cfg, BotOptions())
    registry = _build_registry()
    spawner = _build_spawner()
    outcomes: list[StartOutcome] = []
    for config in configs:
        try:
            outcomes.append(
                start_one(registry, spawner, config, adapter, dry_run, options)
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    _render_start(outcomes, fmt)
    if not dry_run and not all(outcome.ok for outcome in outcomes):
        raise click.exceptions.Exit(1)


@bot_group.command("status")
@click.option(
    "--format", "-F", "fmt", type=click.Choice(["text", "json"]), default="text"
)
def bot_status(fmt: str) -> None:
    """Liveness table of every bot in the registry (live/exited/stale)."""
    records = _build_registry().list_bots()
    _render_status(records, fmt)
    severity = _live_severity(records)
    if severity:
        raise click.exceptions.Exit(severity)


@bot_group.command("stop")
@click.argument(
    "configs",
    nargs=-1,
    required=False,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--all",
    "all_scopes",
    is_flag=True,
    help="Stop every bot the registry knows, not just the named configs.",
)
@click.option(
    "--no-wait",
    is_flag=True,
    help="Write the stop flag and return; do not await the bot's exit.",
)
@click.option(
    "--wait",
    "wait_s",
    type=float,
    default=STOP_WAIT_S,
    help="Seconds to await a bot's exit before reporting it still live.",
)
def bot_stop(
    configs: tuple[str, ...], all_scopes: bool, no_wait: bool, wait_s: float
) -> None:
    """Request a clean stop and await exit; never signals a live bot (\u00a74)."""
    if not all_scopes and not configs:
        raise click.UsageError("give one or more config paths, or --all")
    registry = _build_registry()
    scopes = _stop_scopes(registry, configs, all_scopes)
    outcomes: list[StopOutcome] = []
    for scope in scopes:
        current = registry.probe(scope)
        if current is None or current.state != "live":
            outcomes.append(StopOutcome(scope, True, False, "not live"))
            continue
        outcomes.append(stop_one(registry, scope, wait=not no_wait, timeout_s=wait_s))
    _render_stop(outcomes)
    if not all(outcome.stopped for outcome in outcomes):
        raise click.exceptions.Exit(1)


def _stop_scopes(
    registry: BotRegistry, configs: tuple[str, ...], all_scopes: bool
) -> list[str]:
    """The scopes to stop: every registry row for ``--all``, else the configs'."""
    if all_scopes:
        return [record.scope for record in registry.list_bots()]
    scopes: list[str] = []
    for config in configs:
        try:
            scopes.append(load_scope(config, None)[1])
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    return scopes


def _render_stop(outcomes: Sequence[StopOutcome]) -> None:
    """One line per scope: what was requested and whether it cleared."""
    if not outcomes:
        click.echo("no bots to stop")
        return
    for outcome in outcomes:
        click.echo(f"{outcome.scope}: {outcome.message}")


@bot_group.command("_run", hidden=True)
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--adapter",
    type=click.Choice(_ADAPTERS),
    default=None,
    hidden=True,
    help="Resolved adapter name (keeps parent/child scope identical).",
)
@click.option("--ready-fd", type=int, default=-1, hidden=True)
@click.option("--interval", "-i", type=int, default=None, hidden=True)
@click.option("--max-age", "-a", type=int, default=None, hidden=True)
@click.option("--no-sync", is_flag=True, hidden=True)
@click.option("--retries", type=int, default=None, hidden=True)
def bot_run_child(
    config_path: str,
    adapter: str | None,
    ready_fd: int,
    interval: int | None,
    max_age: int | None,
    no_sync: bool,
    retries: int | None,
) -> None:
    """Hidden child entry: claim the scope lock, signal readiness, run.

    The ready byte fires only after the scope lock AND the pid row exist, so
    the parent never reports a bot live before it genuinely is. What follows is
    the resident loop (§5).
    """
    cfg, scope, _ = load_scope(config_path, adapter)
    registry = _build_registry()
    options = _merge_bot_options(
        interval, max_age, no_sync, retries, read_bot_config(), BotOptions()
    )
    try:
        code = run_loop(
            str(Path(config_path).resolve()),
            cfg=cfg,
            scope=scope,
            registry=registry,
            options=options,
            run_cycle=_run_cycle_fn(adapter),
            on_ready=lambda: _signal(ready_fd, b"1"),
        )
    except BotAlreadyRunningError as exc:
        _signal(ready_fd, b"0")
        raise click.ClickException(str(exc)) from exc
    if code:
        raise click.exceptions.Exit(code)


def _run_cycle_fn(adapter: str | None) -> RunCycleFn:
    """The production cycle seam: one live cycle through ``run_once``.

    ``run_once`` takes its OWN lease, so the bot takes none and the two can
    never nest a lock. Its unsafe code (3) is returned as-is.
    """

    def run(config_path: str, max_age_days: int) -> int:
        try:
            return run_once(config_path, max_age_days=max_age_days, adapter=adapter)
        except (
            ValueError,
            StaleDataError,
            PortfolioFetchError,
            CycleInProgressError,
            LedgerReadError,
        ) as exc:
            _log_line(f"cycle failed: {exc}")
            return EXIT_TRANSIENT

    return run


def load_scope(
    config_path: str, adapter_flag: str | None = None
) -> tuple[LiveConfig, str, str]:
    """Load+validate the config, resolve the adapter (``live run`` precedence),
    and derive the scope key — the ONE identity this bot owns.

    Returns ``(cfg, scope, resolved_adapter)``; the child receives the resolved
    adapter so parent and child mint the identical scope.
    """
    cfg = load_live_config(config_path)
    resolved = resolve_adapter_name(adapter_flag, cfg)
    scope = scope_of(ScopeParts(resolved, config_name_of(cfg), config_scope_hash(cfg)))
    return cfg, scope, resolved


def start_one(
    registry: BotLiveness,
    spawner: BotSpawner,
    config_path: str,
    adapter_flag: str | None,
    dry_run: bool,
    options: BotOptions | None = None,
) -> StartOutcome:
    """Start (or report) one config's bot — the atomic unit of ``bot start``.

    Dry runs probe without side effects and never spawn. A live scope is
    reported, never respawned; the probe is repeated INSIDE the spawning claim,
    which is what serializes concurrent starts. ``live`` is reported only after
    the child's ready byte.
    """
    opts = options if options is not None else BotOptions()
    _, scope, adapter = load_scope(config_path, adapter_flag)
    if dry_run:
        current = registry.probe(scope)
        if current is not None and current.state == "live":
            return StartOutcome(config_path, scope, True, False, "already live")
        return StartOutcome(config_path, scope, True, False, "would spawn")
    current = registry.probe(scope)
    if current is not None and current.state == "live":
        return StartOutcome(config_path, scope, True, False, "already live")
    try:
        with registry.claim(scope):
            current = registry.probe(scope)
            if current is not None and current.state == "live":
                return StartOutcome(config_path, scope, True, False, "already live")
            _clear_stop_flag(registry, scope)
            read_fd, write_fd = os.pipe()
            try:
                try:
                    pid = spawner.spawn(
                        child_argv(
                            config_path,
                            ready_fd=write_fd,
                            adapter=adapter,
                            interval=opts.interval,
                            max_age=opts.max_age,
                            no_sync=opts.no_sync,
                            retries=opts.retries,
                        ),
                        registry.log_path(scope),
                    )
                finally:
                    os.close(write_fd)
                claimed = wait_ready(read_fd, READY_TIMEOUT_S)
            finally:
                os.close(read_fd)
            if not claimed:
                return StartOutcome(
                    config_path, scope, False, True, "child failed to claim its scope"
                )
            return StartOutcome(config_path, scope, True, True, f"live (pid {pid})")
    except BotStartInProgressError as exc:
        return StartOutcome(config_path, scope, False, False, str(exc))


def _merge_bot_options(
    interval: int | None,
    max_age: int | None,
    no_sync: bool,
    retries: int | None,
    cfg: BotConfig,
    default: BotOptions,
) -> BotOptions:
    """CLI flag > config.toml [bot] > code default, one level."""
    return BotOptions(
        interval=interval
        if interval is not None
        else cfg.interval
        if cfg.interval is not None
        else default.interval,
        max_age=max_age
        if max_age is not None
        else cfg.max_age
        if cfg.max_age is not None
        else default.max_age,
        no_sync=no_sync or cfg.no_sync or False,
        retries=retries
        if retries is not None
        else cfg.retries
        if cfg.retries is not None
        else default.retries,
    )


def _clear_stop_flag(registry: BotLiveness, scope: str) -> None:
    """Drop a (possibly stale) stop request before a fresh spawn.

    Only the spawn path clears it: while a bot is live the flag must survive so
    the bot can see it. Best effort — a flag we cannot unlink only matters when
    a stop races a restart.
    """
    try:
        registry.stop_flag_path(scope).unlink(missing_ok=True)
    except OSError:
        pass


def _signal(ready_fd: int, payload: bytes) -> None:
    """Write the ready byte to the parent's pipe, when one was passed."""
    if ready_fd >= 0:
        os.write(ready_fd, payload)
        os.close(ready_fd)


def _log_line(message: str) -> None:
    """Emit a bot diagnostic to stderr (the spawned bot's log)."""
    click.echo(message, err=True)


def _build_registry() -> BotRegistry:
    """The registry this process owns: the live book's db + the runtime dir."""
    return BotRegistry(resolve_live_db_path())


def _build_spawner() -> BotSpawner:
    """The spawner; ``bot start`` calls this so tests can substitute a fake."""
    return PopenSpawner()


def _render_start(outcomes: Sequence[StartOutcome], fmt: str) -> None:
    if fmt == "json":
        click.echo(json.dumps([_outcome_dict(o) for o in outcomes], indent=2))
        return
    for outcome in outcomes:
        click.echo(f"{outcome.config_path} -> {outcome.scope}: {outcome.message}")


def _outcome_dict(outcome: StartOutcome) -> dict[str, object]:
    """The machine-parseable shape of one start result."""
    return {
        "config_path": outcome.config_path,
        "scope": outcome.scope,
        "ok": outcome.ok,
        "spawned": outcome.spawned,
        "message": outcome.message,
    }


def _render_status(records: Sequence[BotRecord], fmt: str) -> None:
    if fmt == "json":
        click.echo(json.dumps([_record_dict(r) for r in records], indent=2))
        return
    if not records:
        click.echo("no bots in the registry")
        return
    click.echo("SCOPE  STATE  PID  STARTED_AT  CONFIG_PATH  EXIT")
    for record in records:
        click.echo(
            f"{record.scope}  {record.state}  {record.pid}  "
            f"{record.started_at.isoformat()}  {record.config_path}  "
            f"{'' if record.exit_code is None else record.exit_code}"
        )


def _record_dict(record: BotRecord) -> dict[str, object]:
    """The machine-parseable shape of one registry row."""
    return {
        "scope": record.scope,
        "config_path": record.config_path,
        "pid": record.pid,
        "started_at": record.started_at.isoformat(),
        "state": record.state,
        "exit_code": record.exit_code,
    }


def _live_severity(records: Sequence[BotRecord]) -> int:
    """The most severe code a STILL-LIVE bot reports (``status``' exit, §6).

    Effectively only a transient code (1) appears here: ``2``/``3`` exit the
    bot, so a live bot cannot STAY at one. The code is stamped before the
    lifetime lock drops, so a terminating bot is briefly ``live`` with its final
    code — harmless (it is about to become ``exited``).
    """
    worst = 0
    for record in records:
        if record.state == "live" and record.exit_code is not None:
            worst = max(worst, record.exit_code)
    return worst


__all__ = [
    "StartOutcome",
    "bot_group",
    "load_scope",
    "start_one",
]
