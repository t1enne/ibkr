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
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import click
import pandas as pd

from src.config import BotConfig, read_bot_config, read_live_config
from src.db.path import resolve_live_db_path
from src.live.lease import CycleInProgressError
from src.live.ledger import LedgerReadError
from src.live.options import LiveOptions, adapter_names, merge_live_options
from src.live.pipeline import PortfolioFetchError
from src.live.freshness import StaleDataError
from src.live.run_once import load_strat_config, run_once
from src.live.scope import (
    ScopeParts,
    config_hash as config_scope_hash,
    config_name_of,
    scope_of,
)
from src.live.types import LiveConfig

from src.bot.liveness import BotAlreadyRunningError, BotStartInProgressError
from src.bot.logs import (
    DEFAULT_TAIL,
    LogTail,
    follow,
    read_tail,
    sleep_tick,
)
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
_ADAPTERS = adapter_names()


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


@dataclass(frozen=True)
class RestartOutcome:
    """One target's restart result (``bot restart``).

    ``ok`` means a bot is live for that scope afterwards; ``restarted`` says
    THIS invocation stopped a live bot and spawned its replacement (False when
    the target was already dead and merely started).
    """

    config_path: str
    scope: str
    ok: bool
    restarted: bool
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
    help="Broker adapter to run through. Default: `[live] adapter` in config.toml, "
    "then `ibkr`.",
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
    options = _bot_options_of(interval, max_age, no_sync, retries)
    live_options = _resolve_live_options(adapter)
    registry = _build_registry()
    spawner = _build_spawner()
    outcomes: list[StartOutcome] = []
    for config in configs:
        try:
            outcomes.append(
                start_one(registry, spawner, config, live_options, dry_run, options)
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    _render_start(outcomes, fmt)
    if not dry_run and not all(outcome.ok for outcome in outcomes):
        raise click.exceptions.Exit(1)


def _resolve_live_options(adapter: str | None) -> LiveOptions:
    """The merged ``[live]`` options — a config typo is exit 1, not a traceback."""
    try:
        return merge_live_options(adapter, None, read_live_config(), LiveOptions())
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _bot_options_of(
    interval: int | None, max_age: int | None, no_sync: bool, retries: int | None
) -> BotOptions:
    """The CLI flags merged over ``config.toml [bot]`` and code defaults."""
    return _merge_bot_options(
        interval, max_age, no_sync, retries, read_bot_config(), BotOptions()
    )


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
            scopes.append(load_scope(config)[1])
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


@bot_group.command("restart")
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
    help="Restart every bot the registry knows, not just the named configs.",
)
@click.option(
    "--wait",
    "wait_s",
    type=float,
    default=STOP_WAIT_S,
    help="Seconds to await a bot's exit before refusing to replace it.",
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
    help="Broker adapter to run through. Default: `[live] adapter` in config.toml, "
    "then `ibkr`.",
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
def bot_restart(
    configs: tuple[str, ...],
    all_scopes: bool,
    wait_s: float,
    dry_run: bool,
    adapter: str | None,
    interval: int | None,
    max_age: int | None,
    no_sync: bool,
    retries: int | None,
    fmt: str,
) -> None:
    """Stop then start each target; exit 0 when every target is live.

    A bot still live when the wait runs out is reported, never replaced: the
    stop is a request, so a restart never signals and never escalates. A target
    already dead is just started.
    """
    if not all_scopes and not configs:
        raise click.UsageError("give one or more config paths, or --all")
    live_options = _resolve_live_options(adapter)
    options = _bot_options_of(interval, max_age, no_sync, retries)
    registry = _build_registry()
    spawner = _build_spawner()
    targets = _all_targets(registry) if all_scopes else _config_targets(configs)
    outcomes = [
        restart_one(registry, spawner, target, live_options, dry_run, wait_s, options)
        for target in targets
    ]
    _render_restart(outcomes, fmt)
    if not dry_run and not all(outcome.ok for outcome in outcomes):
        raise click.exceptions.Exit(1)


def _config_targets(configs: tuple[str, ...]) -> list[tuple[str, str]]:
    """``(config_path, scope)`` per named config; an unknown config is exit 1."""
    targets: list[tuple[str, str]] = []
    for config in configs:
        try:
            targets.append((config, load_scope(config)[1]))
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    return targets


def _all_targets(registry: BotRegistry) -> list[tuple[str, str]]:
    """``(recorded config_path, scope)`` per registry row — the bot's identity."""
    return [(record.config_path, record.scope) for record in registry.list_bots()]


def restart_one(
    registry: BotLiveness,
    spawner: BotSpawner,
    target: tuple[str, str],
    live_options: LiveOptions | None,
    dry_run: bool,
    wait_s: float = STOP_WAIT_S,
    options: BotOptions | None = None,
) -> RestartOutcome:
    """Stop (only when live) then start one bot — the unit of ``bot restart``.

    The stop is ``stop_one``'s request plus a bounded await, never a signal. A
    scope still locked at the deadline is left ALONE and reported: replacing it
    would mean two writers on a live book. A dead target goes straight to
    :func:`start_one`, whose claim and ready byte make the start idempotent.
    """
    config_path, scope = target
    if dry_run:
        return RestartOutcome(config_path, scope, True, False, "would restart")
    if not Path(config_path).is_file():
        return RestartOutcome(config_path, scope, False, False, "config path missing")
    current = registry.probe(scope)
    was_live = current is not None and current.state == "live"
    if was_live:
        stop = stop_one(registry, scope, wait=True, timeout_s=wait_s)
        if stop.still_live:
            return RestartOutcome(config_path, scope, False, False, stop.message)
    start = start_one(registry, spawner, config_path, live_options, dry_run, options)
    message = (
        f"stopped, then {start.message}" if was_live else f"not live; {start.message}"
    )
    return RestartOutcome(config_path, scope, start.ok, start.spawned, message)


def _render_restart(outcomes: Sequence[RestartOutcome], fmt: str) -> None:
    """One terse line per target, or the JSON list of the same.

    A still-live target says so by its stop message (``still live``); nothing
    else in the line implies a replacement happened.
    """
    if fmt == "json":
        click.echo(json.dumps([_restart_dict(o) for o in outcomes], indent=2))
        return
    if not outcomes:
        click.echo("no bots to restart")
        return
    for outcome in outcomes:
        click.echo(f"{outcome.scope}: {outcome.message}")


def _restart_dict(outcome: RestartOutcome) -> dict[str, object]:
    """The machine-parseable shape of one restart result."""
    return {
        "config_path": outcome.config_path,
        "scope": outcome.scope,
        "ok": outcome.ok,
        "restarted": outcome.restarted,
        "message": outcome.message,
    }


@bot_group.command("logs")
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
    help="Every bot the registry knows, not just the named configs.",
)
@click.option(
    "--follow",
    "-f",
    "follow_logs",
    is_flag=True,
    help="Stream appended lines until Ctrl-C (a bounded poll: no busy spin). "
    "A truncated or rotated log is re-read from its start.",
)
@click.option(
    "--tail",
    "-n",
    "tail",
    default=str(DEFAULT_TAIL),
    show_default=True,
    metavar="N|all",
    help="Print only the last N lines; `all` prints the whole file.",
)
@click.option(
    "--timestamps",
    "-t",
    is_flag=True,
    help="Prefix each target's lines with the log file's last-modified time. "
    "The log records no per-line time, so this is per-file, not per-line.",
)
@click.option(
    "--format", "-F", "fmt", type=click.Choice(["text", "json"]), default="text"
)
def bot_logs(
    configs: tuple[str, ...],
    all_scopes: bool,
    follow_logs: bool,
    tail: str,
    timestamps: bool,
    fmt: str,
) -> None:
    """Print each target's bot log; -f keeps streaming it.

    Reading only: nothing is created, locked or spawned. Several targets each
    get a header (==> scope <==) before their lines; one target prints bare
    lines so pipes stay clean. Under --follow the targets are read interleaved,
    and a log that shrinks or vanishes is reopened from its start, so a
    truncation or a logrotate swap is survivable. A target with no log yet is
    reported in its place; the command fails only when EVERY target is
    missing, so one young bot never hides a real log.
    """
    if not all_scopes and not configs:
        raise click.UsageError("give one or more config paths, or --all")
    limit = _tail_limit(tail)
    registry = _build_registry()
    targets = [
        _target_of(registry, target)
        for target in _log_targets(registry, configs, all_scopes)
    ]
    tails = [read_tail(config, scope, path, limit) for config, scope, path in targets]
    try:
        _emit_logs(tails, follow_logs, timestamps, fmt)
    except KeyboardInterrupt:
        return
    if tails and all(tail.missing for tail in tails):
        raise click.exceptions.Exit(1)


def _log_targets(
    registry: BotRegistry, configs: tuple[str, ...], all_scopes: bool
) -> list[tuple[str, str]]:
    """The same ``(config_path, scope)`` rule ``stop``/``restart`` use."""
    return _all_targets(registry) if all_scopes else _config_targets(configs)


def _target_of(registry: BotRegistry, target: tuple[str, str]) -> tuple[str, str, Path]:
    """``(config_path, scope, log_path)`` — the registry owns where the log is."""
    config_path, scope = target
    return config_path, scope, registry.log_path(scope)


def _tail_limit(tail: str) -> int | None:
    """``--tail`` as a line count, or ``None`` for the literal ``all``."""
    if tail.strip().lower() == "all":
        return None
    try:
        count = int(tail)
    except ValueError as exc:
        raise click.UsageError(
            f"--tail takes a line count or `all`, not {tail!r}"
        ) from exc
    if count < 0:
        raise click.UsageError("--tail takes a count >= 0")
    return count


def _emit_logs(
    tails: Sequence[LogTail], follow_logs: bool, timestamps: bool, fmt: str
) -> None:
    """Text lines (optionally streamed) or one JSON object per target.

    Under ``--follow`` the targets are polled in one tight round so their lines
    interleave; the shared tick means one Ctrl-C ends the whole command. JSON
    is one object per target — already parseable, so ``--follow`` stays text.
    """
    if fmt == "json":
        for tail in tails:
            click.echo(json.dumps(_log_dict(tail), indent=2))
        return
    if not tails:
        click.echo("no bots to log", err=True)
        return
    header = len(tails) > 1
    present: list[LogTail] = []
    for tail in tails:
        if tail.missing:
            click.echo(f"no log for {tail.scope} ({tail.path})", err=True)
            continue
        present.append(tail)
        _print_lines(tail, tail.lines, header, timestamps)
    if follow_logs:
        _interleave(present, header, timestamps)


def _interleave(tails: Sequence[LogTail], header: bool, timestamps: bool) -> None:
    """Poll every followed log once per round so their lines interleave.

    One generator per target, each ended by the SAME ``sleep_tick`` returning
    ``None``; ending them together is what makes one Ctrl-C (or, in a test, one
    stop signal) release every stream instead of stranding a sibling.
    """
    stop: list[bool] = []

    def tick() -> str | None:
        if stop:
            return None
        return sleep_tick()

    live = [(tail, follow(tail.path, tail.offset, tick)) for tail in tails]
    while live:
        still: list[tuple[LogTail, Iterator[str]]] = []
        for tail, stream in live:
            line = next(stream, None)
            if line is not None:
                _print_lines(tail, (line,), header, timestamps)
                still.append((tail, stream))
        live = still


def _print_lines(
    tail: LogTail, lines: Sequence[str], header: bool, timestamps: bool
) -> None:
    """One target's lines, with the compose-style header and mtime prefix."""
    if not lines:
        return
    if header:
        click.echo(f"==> {tail.scope} <==")
    prefix = _stamp(tail.path) if timestamps else ""
    for line in lines:
        click.echo(f"{prefix}{line}")


def _stamp(path: Path) -> str:
    """The log file's last write as an ISO time, or nothing when unknowable.

    Per LINE time is not recoverable: ``loop._log`` writes a bare
    ``cycle exit N``. The file's own mtime is the most this reader can honestly
    say, and it is exactly what ``--timestamps``' help promises.
    """
    try:
        mtime = Path(path).stat().st_mtime
    except OSError:
        return ""
    return f"[{pd.Timestamp.fromtimestamp(mtime, tz='UTC').isoformat()}] "


def _log_dict(tail: LogTail) -> dict[str, object]:
    """The machine-parseable shape of one target's log."""
    return {
        "config_path": tail.config_path,
        "scope": tail.scope,
        "path": str(tail.path),
        "lines": list(tail.lines),
        "missing": tail.missing,
    }


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
    live_options = merge_live_options(adapter, None, read_live_config(), LiveOptions())
    cfg, scope = load_scope(config_path, live_options)
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
            run_cycle=_run_cycle_fn(live_options),
            on_ready=lambda: _signal(ready_fd, b"1"),
        )
    except BotAlreadyRunningError as exc:
        _signal(ready_fd, b"0")
        raise click.ClickException(str(exc)) from exc
    if code:
        raise click.exceptions.Exit(code)


def _run_cycle_fn(live_options: LiveOptions) -> RunCycleFn:
    """The production cycle seam: one live cycle through ``run_once``.

    ``run_once`` takes its OWN lease, so the bot takes none and the two can
    never nest a lock. Its unsafe code (3) is returned as-is. The adapter is
    already resolved, so every tick of this bot trades the same backend.

    The loop's CADENCE-derived *max_age_days* replaces the option's, because a
    resident bot ages against its own bar interval — that is a per-tick fact no
    ``[live]`` file setting can know. Every other knob rides through untouched.
    """

    def run(config_path: str, max_age_days: int) -> int:
        try:
            return run_once(
                config_path,
                options=replace(live_options, max_age=max_age_days),
            )
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
    config_path: str, live_options: LiveOptions | None = None
) -> tuple[LiveConfig, str]:
    """Load+validate the config and derive the scope key — the bot's identity.

    *live_options* is the ALREADY MERGED adapter choice (the same
    :func:`merge_live_options` ``live run`` uses), so parent and child mint the
    identical scope from one resolution rather than each re-deriving it. Absent,
    the ``[live]`` table decides on its own — what ``bot stop`` needs, since it
    mints no scope but must address the one a start already minted.
    """
    adapter = (
        live_options
        if live_options is not None
        else merge_live_options(None, None, read_live_config(), LiveOptions())
    )
    cfg = load_strat_config(config_path)
    scope = scope_of(
        ScopeParts(adapter.adapter, config_name_of(cfg), config_scope_hash(cfg))
    )
    return cfg, scope


def start_one(
    registry: BotLiveness,
    spawner: BotSpawner,
    config_path: str,
    live_options: LiveOptions | None,
    dry_run: bool,
    options: BotOptions | None = None,
) -> StartOutcome:
    """Start (or report) one config's bot — the atomic unit of ``bot start``.

    Dry runs probe without side effects and never spawn. A live scope is
    reported, never respawned; the probe is repeated INSIDE the spawning claim,
    which is what serializes concurrent starts. ``live`` is reported only after
    the child's ready byte. ``None`` for *live_options* resolves the ``[live]``
    table, exactly as ``load_scope`` does.
    """
    resolved = (
        live_options
        if live_options is not None
        else merge_live_options(None, None, read_live_config(), LiveOptions())
    )
    opts = options if options is not None else BotOptions()
    _, scope = load_scope(config_path, resolved)
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
                            adapter=resolved.adapter,
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
    "RestartOutcome",
    "StartOutcome",
    "bot_group",
    "load_scope",
    "restart_one",
    "start_one",
]
