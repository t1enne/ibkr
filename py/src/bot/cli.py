"""``ibkr bot`` — wiring only: registry commands, spawn, and the child entry.

Every output shape lives in :mod:`src.bot.render`, every log-format rule in
:mod:`src.bot.logschema`, every TARGET rule in :mod:`src.bot.targets` and the
resolved plan in :mod:`src.bot.plan`. This module decides *what to call*, never
how it reads.

This layer commands the registry only and exits promptly. ``start`` is
idempotent — an already-live scope is reported, never respawned — and takes the
short-lived spawning claim so two concurrent invocations cannot both spawn. The
parent reports ``live`` only after the child signals it holds its scope lock,
and exits non-zero if any child failed to claim. ``_run`` is the hidden child
entry: claim, signal, then run the resident loop.

Exit codes are the machine contract and stay ``0/1/2/3``: a bad or ambiguous
TARGET is a usage error (2), an operational failure is a ``ClickException`` (1),
and ``3`` only ever comes out of a cycle, never out of ``bot`` itself.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import click
import pandas as pd

from src.bot.journal import log_path_of
from src.bot.liveness import BotAlreadyRunningError, BotStartInProgressError
from src.bot.logs import DEFAULT_TAIL, follow, read_tail, sleep_tick
from src.bot.logschema import (
    LOG_KEEP_ENV,
    LOG_LEVEL_ENV,
    LOG_MAX_BYTES_ENV,
    OutputFormat,
    resolve_log_level,
)
from src.bot.loop import EXIT_TRANSIENT, BotOptions, RunCycleFn, run_loop
from src.bot.plan import resolve_plan
from src.bot.registry import BotRegistry, runtime_dir
from src.bot.render import (
    BotDefaults,
    BotPaths,
    LogStream,
    RestartOutcome,
    StartOutcome,
    render_info,
    render_inspect,
    render_logs,
    render_ls,
    render_quiet,
    render_restart,
    render_start,
    render_stop,
)
from src.bot.spawn import READY_TIMEOUT_S, PopenSpawner, child_argv, wait_ready
from src.bot.stop import STOP_WAIT_S, StopOutcome, stop_one
from src.bot.targets import BotTarget, TargetError, registry_rows, resolve_targets
from src.bot.types import BotLiveness, BotRecord, BotSpawner
from src.config import BotConfig, read_bot_config, read_live_config
from src.db.path import resolve_live_db_path
from src.live.freshness import StaleDataError
from src.live.lease import CycleInProgressError
from src.live.ledger import LedgerReadError
from src.live.options import LiveOptions, adapter_names, merge_live_options
from src.live.pipeline import PortfolioFetchError
from src.live.run_once import load_strat_config, run_once
from src.live.scope import (
    ScopeParts,
    config_hash as config_scope_hash,
    config_name_of,
    scope_of,
)
from src.live.types import LiveConfig
from src.shared.style import COLOR, PLAIN, Styler

#: The adapters ``--adapter`` accepts, derived from the ONE typing literal.
_ADAPTERS = adapter_names()

#: The output encodings every command group accepts.
_FORMATS = ("text", "json")

#: A target that names nothing, or names two bots, is a USAGE error (exit 2).
#: ``TargetError`` only: a plain ``ValueError`` from the mint path (a bad config)
#: is an operational failure (1), never a usage error.
_TARGET_ERRORS = (TargetError,)


@click.group(name="bot")
def bot_group() -> None:
    """Unattended trading supervisors — start detached bots, report liveness."""


# --- start ------------------------------------------------------------------


@bot_group.command("start")
@click.argument("configs", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Print the plan; spawn nothing.")
@click.option(
    "--adapter",
    type=click.Choice(_ADAPTERS),
    default=None,
    help="Broker adapter to run through. Default: `[live] adapter` in config.toml.",
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
@click.option("--format", "-F", "fmt", type=click.Choice(_FORMATS), default="text")
def bot_start(
    configs: tuple[str, ...],
    dry_run: bool,
    adapter: str | None,
    interval: int | None,
    max_age: int | None,
    no_sync: bool,
    retries: int | None,
    fmt: OutputFormat,
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
        token = _target_of_config(registry, config)
        outcomes.append(
            start_one(registry, spawner, token, live_options, dry_run, options)
        )
    click.echo(render_start(outcomes, fmt))
    # A dry run normally exits 0, but a REFUSED target is a real conflict, not a
    # plan: the operator must see the same non-zero the real start would give.
    _fail_on_refused(outcomes)


def _fail_on_refused(outcomes: Sequence[StartOutcome | RestartOutcome]) -> None:
    """Exit 1 when any outcome is not ``ok`` — a refusal counts in a dry run too.

    ``--dry-run`` is how an operator FINDS a config/row scope conflict before a
    scheduler restart loop trips on it, so it must not report success while
    printing a refusal.
    """
    if not all(outcome.ok for outcome in outcomes):
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


def _target_of_config(registry: BotLiveness, config: str) -> BotTarget:
    """One config path as a resolved target (``start`` always names a config).

    Registry-row-first, exactly like every other command: a config whose row
    already exists addresses the scope that bot holds, so a scheduler re-running
    ``start`` over a live bot can never mint a second scope for it.
    """
    return _one_target(registry, config)


# --- ls ---------------------------------------------------------------------


@bot_group.command("ls")
@click.argument("tokens", nargs=-1, required=False)
@click.option(
    "-q", "--quiet", is_flag=True, help="Print one scope per line; no header."
)
@click.option("--format", "-F", "fmt", type=click.Choice(_FORMATS), default="text")
def bot_ls(tokens: tuple[str, ...], quiet: bool, fmt: OutputFormat) -> None:
    """Every bot the registry knows, live/exited/stale alike.

    Exited and stale rows are shown by default on purpose: the alerting contract
    pages on rows that never come back, which a live-only view would hide.
    """
    _ls(_build_registry(), tokens, quiet, fmt)


def _ls(
    registry: BotRegistry, tokens: Sequence[str], quiet: bool, fmt: OutputFormat
) -> None:
    """``ls``' body, shared with the ``status`` alias (ONE implementation)."""
    rows = _rows_of(registry, tokens)
    if quiet:
        if fmt == "json":
            click.echo(render_ls(rows, fmt))
        elif rows:
            click.echo(render_quiet(rows))
    else:
        click.echo(render_ls(rows, fmt, _styler(fmt)))
    severity = _live_severity(rows)
    if severity:
        raise click.exceptions.Exit(severity)


@bot_group.command("status", hidden=True)
@click.argument("tokens", nargs=-1, required=False)
@click.option("-q", "--quiet", is_flag=True, help="Print one scope per line.")
@click.option("--format", "-F", "fmt", type=click.Choice(_FORMATS), default="text")
def bot_status(tokens: tuple[str, ...], quiet: bool, fmt: OutputFormat) -> None:
    """Alias of ``ls`` (kept so existing invocations keep working)."""
    _ls(_build_registry(), tokens, quiet, fmt)


def _rows_of(registry: BotRegistry, tokens: Sequence[str]) -> list[BotRecord]:
    """All rows, or just the targeted ones — in the registry's own order."""
    rows = registry.list_bots()
    if not tokens:
        return rows
    wanted = {target.scope for target in _targets(registry, tokens)}
    return [row for row in rows if row.scope in wanted]


# --- inspect ----------------------------------------------------------------


@bot_group.command("inspect")
@click.argument("tokens", nargs=-1, required=True)
@click.option(
    "--adapter",
    type=click.Choice(_ADAPTERS),
    default=None,
    help="Adapter whose scope path a fresh target would use.",
)
@click.option("--format", "-F", "fmt", type=click.Choice(_FORMATS), default="text")
def bot_inspect(
    tokens: tuple[str, ...], adapter: str | None, fmt: OutputFormat
) -> None:
    """The resolved plan for each TARGET: identity, cadence, and every path.

    Read-only: nothing is created, locked or spawned, and a scope is never
    re-minted for a bot the registry already knows.
    """
    registry = _build_registry()
    live_options = _resolve_live_options(adapter)
    plans = [
        resolve_plan(
            registry, target, live_options, _bot_options_of(None, None, False, None)
        )
        for target in _targets(registry, tokens)
    ]
    click.echo("\n".join(render_inspect(plan, fmt) for plan in plans))


# --- info -------------------------------------------------------------------


@bot_group.command("info")
@click.option("--format", "-F", "fmt", type=click.Choice(_FORMATS), default="text")
def bot_info(fmt: OutputFormat) -> None:
    """Where state lives and which defaults apply. Read-only."""
    registry = _build_registry()
    rows = registry.list_bots()
    paths = BotPaths(resolve_live_db_path(), runtime_dir())
    defaults = _defaults_of(_bot_options_of(None, None, False, None))
    if fmt == "json":
        click.echo(_info_json(paths, defaults, rows))
        return
    click.echo(render_info(paths, defaults, rows, _log_env()))


def _defaults_of(options: BotOptions) -> BotDefaults:
    """The effective ``[bot]`` defaults, as ``info`` states them."""
    return BotDefaults(
        interval_s="1 bar" if options.interval is None else f"{options.interval}s",
        max_age_days="2x cadence" if options.max_age is None else f"{options.max_age}d",
        retries=options.retries,
        sync=not options.no_sync,
    )


def _log_env() -> dict[str, str]:
    """The log env vars actually set (so ``info`` can name its overrides)."""
    names = (LOG_LEVEL_ENV, LOG_MAX_BYTES_ENV, LOG_KEEP_ENV)
    return {name: os.environ[name] for name in names if name in os.environ}


def _info_json(
    paths: BotPaths, defaults: BotDefaults, rows: Sequence[BotRecord]
) -> str:
    """``info -F json``: the same facts, machine-readable."""
    return json.dumps(
        {
            "db_path": str(paths.db_path),
            "runtime": str(paths.runtime),
            "defaults": {
                "interval": defaults.interval_s,
                "max_age": defaults.max_age_days,
                "retries": defaults.retries,
                "sync": defaults.sync,
            },
            "bots": len(rows),
        },
        indent=2,
    )


# --- logs -------------------------------------------------------------------


@bot_group.command("logs")
@click.argument("tokens", nargs=-1, required=False)
@click.option("--all", "all_bots", is_flag=True, help="Every bot the registry knows.")
@click.option(
    "--follow",
    "-f",
    "follow_logs",
    is_flag=True,
    help="Stream appended lines until Ctrl-C; a rotated log is reopened.",
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
    "--since",
    default=None,
    metavar="DURATION",
    help="Only lines newer than this, e.g. 15m, 2h, 1d.",
)
@click.option(
    "--level",
    default=None,
    metavar="LEVEL",
    help="Only lines at or above LEVEL (DEBUG|INFO|WARN|ERROR).",
)
@click.option("--format", "-F", "fmt", type=click.Choice(_FORMATS), default="text")
def bot_logs(
    tokens: tuple[str, ...],
    all_bots: bool,
    follow_logs: bool,
    tail: str,
    since: str | None,
    level: str | None,
    fmt: OutputFormat,
) -> None:
    """Print each target's log; -f keeps streaming it.

    Reading only: nothing is created, locked or spawned. A target with no log
    yet is reported in its place, and the command fails only when EVERY target
    is missing, so one young bot never hides a real log.
    """
    limit = _tail_limit(tail)
    cutoff = _since_of(since)
    minimum = _level_of(level)
    registry = _build_registry()
    targets = _all_as_targets(registry) if all_bots else _targets(registry, tokens)
    streams = [_stream_of(target, limit) for target in targets]
    header = len(streams) > 1
    try:
        click.echo(
            render_logs(streams, fmt, header=header, since=cutoff, minimum=minimum)
        )
        if follow_logs:
            _interleave(streams, fmt, header, cutoff, minimum)
    except KeyboardInterrupt:
        return
    if streams and all(stream.missing for stream in streams):
        raise click.exceptions.Exit(1)


def _stream_of(target: BotTarget, limit: int | None) -> LogStream:
    """One target's log as a renderable stream (the ONE path rule owns it)."""
    path = log_path_of(target.scope)
    tail = read_tail(target.scope, path, limit)
    return LogStream(tail.scope, tail.path, tail.lines, tail.offset, tail.missing)


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


def _since_of(since: str | None) -> pd.Timestamp | None:
    """``--since 15m|2h|1d`` as a UTC cutoff timestamp, or ``None`` when absent."""
    if since is None or not since.strip():
        return None
    return pd.Timestamp.now(tz="UTC") - _duration(since)


def _duration(text: str) -> pd.Timedelta:
    """One ``<n><unit>`` duration as a :class:`pandas.Timedelta` (m/h/d/w)."""
    unit = text.strip()[-1:]
    number = text.strip()[:-1]
    scales = {"m": "min", "h": "h", "d": "D", "w": "W"}
    if unit not in scales:
        raise click.UsageError(f"--since takes <n>m|h|d|w, not {text!r}")
    try:
        count = int(number)
    except ValueError as exc:
        raise click.UsageError(f"--since takes <n>m|h|d|w, not {text!r}") from exc
    if count <= 0:
        raise click.UsageError(f"--since takes a positive count, not {text!r}")
    span = pd.Timedelta(count, scales[unit])
    if not isinstance(span, pd.Timedelta):
        raise click.UsageError(f"--since takes <n>m|h|d|w, not {text!r}")
    return span


def _level_of(level: str | None) -> str | None:
    """``--level`` normalised, or ``None`` when absent (a typo is a usage error)."""
    if level is None or not level.strip():
        return None
    try:
        return resolve_log_level(level)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc


def _interleave(
    streams: Sequence[LogStream],
    fmt: OutputFormat,
    header: bool,
    cutoff: pd.Timestamp | None,
    minimum: str | None,
) -> None:
    """Poll every followed log once per round so their lines interleave.

    One generator per target, each ended by the SAME ``sleep_tick`` returning
    ``None``; ending them together is what makes one Ctrl-C release every stream
    instead of stranding a sibling.
    """
    stop: list[bool] = []

    def tick() -> str | None:
        if stop:
            return None
        return sleep_tick()

    live = [
        (stream, follow(stream.path, stream.offset, tick))
        for stream in streams
        if not stream.missing
    ]
    while live:
        still: list[tuple[LogStream, Iterator[str]]] = []
        for stream, lines in live:
            line = next(lines, None)
            if line is None:
                continue
            printed = render_logs(
                [replace(stream, lines=(line,))],
                fmt,
                header=header,
                since=cutoff,
                minimum=minimum,
            )
            if printed:
                click.echo(printed)
            still.append((stream, lines))
        live = still


# --- stop / restart ---------------------------------------------------------


@bot_group.command("stop")
@click.argument("tokens", nargs=-1, required=False)
@click.option("--all", "all_bots", is_flag=True, help="Every bot the registry knows.")
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
@click.option("--format", "-F", "fmt", type=click.Choice(_FORMATS), default="text")
def bot_stop(
    tokens: tuple[str, ...],
    all_bots: bool,
    no_wait: bool,
    wait_s: float,
    fmt: OutputFormat,
) -> None:
    """Request a clean stop and await exit; never signals a live bot.

    The stop is a request: a flag the loop reads at its sleep point, so no cycle
    is interrupted and no process is ever signalled.
    """
    registry = _build_registry()
    targets = _all_as_targets(registry) if all_bots else _targets(registry, tokens)
    outcomes = [_stop_target(registry, target, no_wait, wait_s) for target in targets]
    click.echo(render_stop(outcomes, fmt))
    if not all(outcome.stopped for outcome in outcomes):
        raise click.exceptions.Exit(1)


def _stop_target(
    registry: BotRegistry, target: BotTarget, no_wait: bool, wait_s: float
) -> StopOutcome:
    """One target's stop: a not-live target is reported, never signalled."""
    current = registry.probe(target.scope)
    if current is None or current.state != "live":
        return StopOutcome(target.scope, True, False, "not live")
    return stop_one(registry, target.scope, wait=not no_wait, timeout_s=wait_s)


@bot_group.command("restart")
@click.argument("tokens", nargs=-1, required=False)
@click.option("--all", "all_bots", is_flag=True, help="Every bot the registry knows.")
@click.option(
    "--wait",
    "wait_s",
    type=float,
    default=STOP_WAIT_S,
    help="Seconds to await a bot's exit before refusing to replace it.",
)
@click.option("--dry-run", is_flag=True, help="Print the plan; spawn nothing.")
@click.option(
    "--adapter",
    type=click.Choice(_ADAPTERS),
    default=None,
    help="Broker adapter to run through. Default: `[live] adapter` in config.toml.",
)
@click.option("--interval", "-i", type=int, default=None, help="Cadence in seconds.")
@click.option("--max-age", "-a", type=int, default=None, help="Staleness gate in days.")
@click.option("--no-sync", is_flag=True, help="Skip the pre-cycle candle sync.")
@click.option("--retries", type=int, default=None, help="Transient cycles to retry.")
@click.option("--format", "-F", "fmt", type=click.Choice(_FORMATS), default="text")
def bot_restart(
    tokens: tuple[str, ...],
    all_bots: bool,
    wait_s: float,
    dry_run: bool,
    adapter: str | None,
    interval: int | None,
    max_age: int | None,
    no_sync: bool,
    retries: int | None,
    fmt: OutputFormat,
) -> None:
    """Stop then start each target; exit 0 when every target is live.

    A bot still live when the wait runs out is reported, never replaced: the
    stop is a request, so a restart never signals and never escalates.
    """
    live_options = _resolve_live_options(adapter)
    options = _bot_options_of(interval, max_age, no_sync, retries)
    registry = _build_registry()
    spawner = _build_spawner()
    targets = _all_as_targets(registry) if all_bots else _targets(registry, tokens)
    outcomes = [
        restart_one(registry, spawner, target, live_options, dry_run, wait_s, options)
        for target in targets
    ]
    click.echo(render_restart(outcomes, fmt))
    _fail_on_refused(outcomes)


def restart_one(
    registry: BotLiveness,
    spawner: BotSpawner,
    target: BotTarget,
    live_options: LiveOptions | None,
    dry_run: bool,
    wait_s: float = STOP_WAIT_S,
    options: BotOptions | None = None,
) -> RestartOutcome:
    """Stop (only when live) then start one bot — the unit of ``bot restart``.

    The stop is :func:`stop_one`'s request plus a bounded await, never a signal.
    A scope still locked at the deadline is left ALONE and reported: replacing it
    would mean two writers on a live book. A dead target goes straight to
    :func:`start_one`, whose claim and ready byte make the start idempotent.
    """
    if not Path(target.config_path).is_file():
        return RestartOutcome(
            target.config_path, target.scope, False, False, "config path missing"
        )
    conflict = scope_conflict(target)
    if conflict is not None:
        return RestartOutcome(target.config_path, target.scope, False, False, conflict)
    if dry_run:  # after the divergence check: a dry run REPORTS it, writes nothing
        return RestartOutcome(
            target.config_path, target.scope, True, False, "would restart"
        )
    current = registry.probe(target.scope)
    was_live = current is not None and current.state == "live"
    if was_live:
        stop = stop_one(registry, target.scope, wait=True, timeout_s=wait_s)
        if stop.still_live:
            return RestartOutcome(
                target.config_path, target.scope, False, False, stop.message
            )
    start = start_one(registry, spawner, target, live_options, dry_run, options)
    message = (
        f"stopped, then {start.message}" if was_live else f"not live; {start.message}"
    )
    return RestartOutcome(
        target.config_path, target.scope, start.ok, start.spawned, message
    )


# --- the child entry --------------------------------------------------------


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

    The ready byte fires only after the scope lock AND the pid row exist, so the
    parent never reports a bot live before it genuinely is.
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

    ``run_once`` takes its OWN lease, so the bot takes none and the two can never
    nest a lock. Its unsafe code (3) is returned as-is. The loop's
    cadence-derived *max_age_days* replaces the option's, because a resident bot
    ages against its own bar interval — a per-tick fact no ``[live]`` file
    setting can know.
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
            click.echo(f"cycle failed: {exc}", err=True)
            return EXIT_TRANSIENT

    return run


# --- shared pieces ----------------------------------------------------------


def _one_target(registry: BotLiveness, token: str) -> BotTarget:
    """One TARGET token resolved, with a usage error when it names no bot."""
    return _targets(registry, (token,))[0]


def _targets(registry: BotLiveness, tokens: Sequence[str]) -> tuple[BotTarget, ...]:
    """Every token resolved; ambiguity/missing is exit 2, never a guess."""
    if not tokens:
        raise click.UsageError(
            "give one or more TARGETs (name, scope or config), or --all"
        )
    try:
        return resolve_targets(registry, tokens, mint=_mint)
    except _TARGET_ERRORS as exc:
        raise click.UsageError(str(exc)) from exc


def _all_as_targets(registry: BotLiveness) -> tuple[BotTarget, ...]:
    """Every registry row as a target — ``--all``, addressed by recorded identity."""
    return tuple(
        BotTarget(row.config_path, row.scope, True) for row in registry_rows(registry)
    )


def _mint(config_path: str) -> str:
    """The ONE scope derivation from a config on disk (``load_scope``)."""
    return load_scope(config_path)[1]


def load_scope(
    config_path: str, live_options: LiveOptions | None = None
) -> tuple[LiveConfig, str]:
    """Load+validate the config and derive the scope key — the bot's identity.

    *live_options* is the ALREADY MERGED adapter choice (the same
    :func:`merge_live_options` ``live run`` uses), so parent and child mint the
    identical scope from one resolution rather than each re-deriving it.
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
    target: BotTarget,
    live_options: LiveOptions | None,
    dry_run: bool,
    options: BotOptions | None = None,
) -> StartOutcome:
    """Start (or report) one target's bot — the atomic unit of ``bot start``.

    Dry runs probe without side effects and never spawn. A live scope is
    reported, never respawned; the probe is repeated INSIDE the spawning claim,
    which is what serializes concurrent starts. ``live`` is reported only after
    the child's ready byte — so a target that is stale, exited or unknown can
    never be reported live on the strength of a row alone.

    A target whose config no longer mints the scope its row holds is REFUSED
    before any claim, spawn or stop-flag clear: see :func:`scope_conflict`.
    """
    resolved = (
        live_options
        if live_options is not None
        else merge_live_options(None, None, read_live_config(), LiveOptions())
    )
    opts = options if options is not None else BotOptions()
    scope = target.scope
    conflict = scope_conflict(target)
    if conflict is not None:
        return StartOutcome(target.config_path, scope, False, False, conflict)
    if dry_run:
        current = registry.probe(scope)
        message = (
            "already live"
            if current is not None and current.state == "live"
            else "would spawn"
        )
        return StartOutcome(target.config_path, scope, True, False, message)
    try:
        with registry.claim(scope):
            current = registry.probe(scope)
            if current is not None and current.state == "live":
                return StartOutcome(
                    target.config_path, scope, True, False, "already live"
                )
            return _spawn_claimed(registry, spawner, target, resolved, opts)
    except BotStartInProgressError as exc:
        return StartOutcome(target.config_path, scope, False, False, str(exc))


def scope_conflict(target: BotTarget) -> str | None:
    """The refusal text when a target's config re-mints away from its row.

    The parent claims and reports ``target.scope`` (the registry row) while the
    child mints its OWN scope from the config. After a strategy-intent edit those
    differ, so the claim would protect a scope nobody is spawning while the child
    trades another — the double-exposure hazard AGENTS.md §Live trading names.

    A config that no longer mints its row's scope is therefore never spawned
    under either identity: the operator stops it while flat and starts fresh, so
    the re-mint is a decision, never a side effect of a restart. Gated on
    ``from_registry``: a config with no row has no identity to contradict.
    """
    if not target.from_registry:
        return None
    minted = _minted_scope(target.config_path)
    if minted is None or minted == target.scope:
        return None
    return (
        f"refused: {target.config_path} now mints scope {minted}, but "
        f"{target.scope} holds this bot's identity — stop it while flat, "
        "then start again"
    )


def _minted_scope(config_path: str) -> str | None:
    """What *config_path* mints now, or ``None`` when it no longer loads.

    The adapter comes from the same resolution a spawn uses when no ``--adapter``
    was given, so the compare is not poisoned by a flag. A config that raises is
    not a re-mint — it is a broken config, and the start reports that on its own.
    """
    try:
        return _mint(config_path)
    except ValueError:
        return None


def _spawn_claimed(
    registry: BotLiveness,
    spawner: BotSpawner,
    target: BotTarget,
    resolved: LiveOptions,
    opts: BotOptions,
) -> StartOutcome:
    """Spawn inside the claim and wait for the ready byte (the handshake)."""
    scope = target.scope
    _clear_stop_flag(registry, scope)
    read_fd, write_fd = os.pipe()
    try:
        try:
            pid = spawner.spawn(
                child_argv(
                    target.config_path,
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
            target.config_path, scope, False, True, "child failed to claim its scope"
        )
    return StartOutcome(target.config_path, scope, True, True, f"live (pid {pid})")


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


def _styler(fmt: OutputFormat) -> Styler:
    """ANSI only for a human's terminal AND the human text format."""
    return COLOR if fmt == "text" and _is_tty() else PLAIN


def _is_tty() -> bool:
    """Whether stdout is a terminal (a pipe or cron log never gets colour)."""
    return click.get_text_stream("stdout").isatty()


def _build_registry() -> BotRegistry:
    """The registry this process owns: the live book's db + the runtime dir."""
    return BotRegistry(resolve_live_db_path())


def _build_spawner() -> BotSpawner:
    """The spawner; ``bot start`` calls this so tests can substitute a fake."""
    return PopenSpawner()


def _live_severity(records: Sequence[BotRecord]) -> int:
    """The most severe code a STILL-LIVE bot reports (``ls``' exit, §6).

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
    "StopOutcome",
    "bot_group",
    "load_scope",
    "restart_one",
    "start_one",
]
