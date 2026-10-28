"""Every ``ibkr bot`` output shape, in one place (plan §4).

Rendering is a pure function of its arguments — no ``sys.stdout``, no clock read
inside a formatter — so the caller decides styling (a TTY gets ``COLOR``, a pipe
or ``-F json`` gets ``PLAIN``) and a report's bytes never depend on where they
went. The one exception is :func:`render_ls`'s ``now``, passed in for the same
reason: an age is a function of the observation time, not of the renderer.

Nothing here decides an exit code; the CLI does. A renderer that returned text
AND a code would make the two drift.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.bt.table import Col, Table, render
from src.bot.logschema import (
    DEFAULT_LOG_KEEP,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_MAX_BYTES,
    OutputFormat,
    line_to_json,
    parse_line,
)
from src.bot.plan import BotPlan
from src.bot.stop import StopOutcome
from src.bot.targets import name_of
from src.bot.types import BotRecord
from src.shared.style import PLAIN, Styler

#: Table column separator, in one place for the tables and the field blocks.
#: ``src.bt.table``'s default, imported rather than retyped beside a claim.
SEP = "  "


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


@dataclass(frozen=True)
class LogStream:
    """One target's log for rendering: its lines and where a follow resumes."""

    scope: str
    path: Path
    lines: tuple[str, ...]
    offset: int
    missing: bool


@dataclass(frozen=True)
class BotPaths:
    """Where a deployment's state lives — what ``info`` reports."""

    db_path: Path
    runtime: Path


@dataclass(frozen=True)
class BotDefaults:
    """The effective code defaults ``info`` states (never a live reading)."""

    interval_s: str
    max_age_days: str
    retries: int
    sync: bool


def render_ls(
    rows: Sequence[BotRecord],
    fmt: OutputFormat,
    styler: Styler = PLAIN,
    now: pd.Timestamp | None = None,
) -> str:
    """The registry table: identity, state, timing and the recorded config.

    AGE is time since ``started_at``. The pid row carries no per-cycle
    timestamp, so there is no LAST column: a second age column could only
    repeat AGE.
    """
    if fmt == "json":
        return _json([_record_dict(row) for row in rows])
    if not rows:
        return "no bots in the registry"
    stamp = now if now is not None else pd.Timestamp.now(tz="UTC")
    table = Table(
        columns=(
            Col("NAME"),
            Col("SCOPE"),
            Col("STATE"),
            Col("PID", align=">"),
            Col("EXIT", align=">"),
            Col("AGE", align=">"),
            Col("CONFIG"),
        ),
        rows=tuple(_ls_row(row, stamp) for row in rows),
    )
    return "\n".join(render(table, sep=SEP, styler=styler))


def render_quiet(rows: Sequence[BotRecord]) -> str:
    """One scope per line — the ``-q`` / scripting form of ``ls``."""
    return "\n".join(row.scope for row in rows)


def render_start(outcomes: Sequence[StartOutcome], fmt: OutputFormat) -> str:
    """Per config: what happened and what is true now."""
    if fmt == "json":
        return _json(
            [
                {
                    "config_path": o.config_path,
                    "scope": o.scope,
                    "ok": o.ok,
                    "spawned": o.spawned,
                    "message": o.message,
                }
                for o in outcomes
            ]
        )
    table = Table(
        columns=(Col("NAME"), Col("SCOPE"), Col("RESULT")),
        rows=tuple((name_of(o.config_path), o.scope, o.message) for o in outcomes),
    )
    return "\n".join(render(table, sep=SEP))


def render_stop(outcomes: Sequence[StopOutcome], fmt: OutputFormat) -> str:
    """Per target: whether the stop request cleared."""
    if fmt == "json":
        return _json(
            [
                {
                    "scope": o.scope,
                    "stopped": o.stopped,
                    "still_live": o.still_live,
                    "message": o.message,
                }
                for o in outcomes
            ]
        )
    if not outcomes:
        return "no bots to stop"
    table = Table(
        columns=(Col("SCOPE"), Col("RESULT")),
        rows=tuple((o.scope, o.message) for o in outcomes),
    )
    return "\n".join(render(table, sep=SEP))


def render_restart(outcomes: Sequence[RestartOutcome], fmt: OutputFormat) -> str:
    """Per target: what happened and what is true now."""
    if fmt == "json":
        return _json(
            [
                {
                    "config_path": o.config_path,
                    "scope": o.scope,
                    "ok": o.ok,
                    "restarted": o.restarted,
                    "message": o.message,
                }
                for o in outcomes
            ]
        )
    if not outcomes:
        return "no bots to restart"
    table = Table(
        columns=(Col("NAME"), Col("SCOPE"), Col("RESULT")),
        rows=tuple((name_of(o.config_path), o.scope, o.message) for o in outcomes),
    )
    return "\n".join(render(table, sep=SEP))


def render_inspect(plan: BotPlan, fmt: OutputFormat) -> str:
    """The resolved plan as a field-per-line block, or the object verbatim."""
    if fmt == "json":
        return _json(_plan_dict(plan))
    cadence = (
        f"interval {plan.interval_s}s   max_age {plan.max_age_days}d   "
        f"retries {plan.retries}   sync {'on' if plan.sync else 'off'}"
    )
    fields: tuple[tuple[str, str], ...] = (
        ("NAME", name_of(plan.target.config_path)),
        ("SCOPE", plan.target.scope),
        ("ADAPTER", plan.adapter),
        ("CONFIG", plan.target.config_path),
        ("IDENTITY", "registry row" if plan.target.from_registry else "config"),
        ("STATE", plan.state),
        ("EXIT", "-" if plan.exit_code is None else str(plan.exit_code)),
        ("CADENCE", cadence),
        ("DB", str(plan.db_path)),
        ("LOCK", str(plan.lock_path)),
        ("PID ROW", str(plan.pid_path)),
        ("STOP FLAG", str(plan.stop_flag_path)),
        ("LOG", _log_cell(plan.log_path)),
    )
    width = max(len(label) for label, _ in fields)
    return "\n".join(f"{label.ljust(width)}{SEP}{value}" for label, value in fields)


def render_info(
    paths: BotPaths,
    defaults: BotDefaults,
    rows: Sequence[BotRecord],
    env: Mapping[str, str] | None = None,
) -> str:
    """Where state lives, which defaults apply, and what the registry holds."""
    counts = _state_counts(rows)
    summary = ", ".join(f"{count} {state}" for state, count in counts.items())
    overrides = dict(env or {})
    fields: tuple[tuple[str, str], ...] = (
        ("DB", str(paths.db_path)),
        ("RUNTIME", _with_env(str(paths.runtime), "IBKR_BOT_RUNTIME_DIR", overrides)),
        (
            "DEFAULTS",
            f"interval {defaults.interval_s}   max_age {defaults.max_age_days}   "
            f"retries {defaults.retries}   sync {'on' if defaults.sync else 'off'}",
        ),
        ("LOG", _log_settings(overrides)),
        ("REGISTRY", f"{len(rows)} bots ({summary})" if rows else "empty"),
    )
    width = max(len(label) for label, _ in fields)
    return "\n".join(f"{label.ljust(width)}{SEP}{value}" for label, value in fields)


def render_logs(
    streams: Sequence[LogStream],
    fmt: OutputFormat,
    *,
    header: bool = False,
    since: pd.Timestamp | None = None,
    minimum: str | None = None,
) -> str:
    """The log lines of one or more targets, parsed when they are ours.

    A legacy line that does not parse is passed through verbatim (so a
    historical file still reads) and is never filtered: ``--since``/``--level``
    only apply to lines whose timestamp and level are actually known. A missing
    log is reported in the caller's own format — ``-F json`` stays parseable.
    """
    blocks: list[str] = []
    for stream in streams:
        if stream.missing:
            blocks.append(_missing_text(stream, fmt))
            continue
        lines = [
            _log_line_text(line, fmt)
            for line in stream.lines
            if _selected(line, since, minimum)
        ]
        if not lines:
            continue
        if header:
            lines.insert(0, f"==> {stream.scope} <==")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _missing_text(stream: LogStream, fmt: OutputFormat) -> str:
    """The report for a target with no log yet, in *fmt*.

    ``-F json`` gets an object, never the plain-text sentence, so ``--follow``
    and ``--all`` output stays one parseable JSON value per line.
    """
    if fmt == "json":
        return json.dumps(
            {"scope": stream.scope, "missing": True, "path": str(stream.path)}
        )
    return f"no log for {stream.scope} ({stream.path})"


def _selected(text: str, since: pd.Timestamp | None, minimum: str | None) -> bool:
    """Whether one raw line survives the ``--since``/``--level`` filters."""
    if since is None and minimum is None:
        return True
    parsed = parse_line(text)
    if parsed is None:
        return True  # an unparseable legacy line is never guessed at
    if since is not None and parsed.at < since:
        return False
    if minimum is not None and _rank(parsed.level) < _rank(minimum):
        return False
    return True


def _rank(level: str) -> int:
    """A level's severity rank (DEBUG lowest), for a ``--level`` threshold."""
    order = ("DEBUG", "INFO", "WARN", "ERROR")
    return order.index(level) if level in order else 0


def _log_line_text(text: str, fmt: OutputFormat) -> str:
    """One line as it should be printed, in *fmt*.

    JSON is one object per line for a parsed line and the raw text as a JSON
    string for a legacy one, so ``-F json`` output is always line-delimited
    parseable. Text prints the line itself, whose own first field IS its
    timestamp — nothing prefixes a second one.
    """
    parsed = parse_line(text)
    if fmt == "json":
        if parsed is None:
            return json.dumps(text)
        return json.dumps(line_to_json(parsed))
    return text


def _ls_row(row: BotRecord, now: pd.Timestamp) -> tuple[str, ...]:
    """One ``ls`` row: identity, derived state, age and the recorded config."""
    return (
        name_of(row.config_path),
        row.scope,
        row.state,
        str(row.pid),
        "-" if row.exit_code is None else str(row.exit_code),
        _age(row.started_at, now),
        row.config_path,
    )


def _age(at: pd.Timestamp, now: pd.Timestamp) -> str:
    """*at* as a compact age (``42s``/``9m``/``6h``/``2d``), never negative."""
    then = at.tz_localize("UTC") if at.tzinfo is None else at.tz_convert("UTC")
    seconds = max(0, int((now - then).total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3_600:
        return f"{seconds // 60}m"
    if seconds < 86_400:
        return f"{seconds // 3_600}h"
    return f"{seconds // 86_400}d"


def _log_cell(path: Path) -> str:
    """A log's path plus its size and rotation count when it exists."""
    try:
        size = path.stat().st_size
    except OSError:
        return str(path)
    generations = len(list(path.parent.glob(f"{path.name}.*")))
    return f"{path}{SEP}{_bytes(size)}{SEP}{generations} rotate"


def _bytes(size: int) -> str:
    """*size* in the coarsest binary unit that keeps it readable."""
    for unit, scale in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if size >= scale:
            return f"{size / scale:.1f} {unit}"
    return f"{size} B"


def _with_env(value: str, name: str, env: Mapping[str, str]) -> str:
    """*value* annotated with the env var that can override it, when set."""
    return f"{value}{SEP}({name})" if name in env else value


def _log_settings(env: Mapping[str, str]) -> str:
    """The log level and bounds in effect, naming the env var that sets each.

    The fallbacks are the imported constants, so changing one default cannot
    leave ``info`` stating a bound the bot no longer uses.
    """
    level = env.get("IBKR_BOT_LOG_LEVEL") or DEFAULT_LOG_LEVEL
    bound = env.get("IBKR_BOT_LOG_MAX_BYTES") or _bytes(DEFAULT_LOG_MAX_BYTES)
    keep = env.get("IBKR_BOT_LOG_KEEP") or str(DEFAULT_LOG_KEEP)
    return f"level {level}   max {bound}   keep {keep}"


def _state_counts(rows: Sequence[BotRecord]) -> dict[str, int]:
    """Rows per state, in the fixed order ``live``/``exited``/``stale``."""
    counts = {"live": 0, "exited": 0, "stale": 0}
    for row in rows:
        counts[row.state] = counts.get(row.state, 0) + 1
    return counts


def _record_dict(row: BotRecord) -> dict[str, object]:
    """The machine-parseable shape of one registry row (JSON keys unchanged)."""
    return {
        "scope": row.scope,
        "config_path": row.config_path,
        "pid": row.pid,
        "started_at": row.started_at.isoformat(),
        "state": row.state,
        "exit_code": row.exit_code,
    }


def _plan_dict(plan: BotPlan) -> dict[str, object]:
    """The machine-parseable shape of one resolved plan."""
    return {
        "target": {
            "config_path": plan.target.config_path,
            "scope": plan.target.scope,
            "from_registry": plan.target.from_registry,
        },
        "adapter": plan.adapter,
        "interval_s": plan.interval_s,
        "max_age_days": plan.max_age_days,
        "retries": plan.retries,
        "sync": plan.sync,
        "db_path": str(plan.db_path),
        "lock_path": str(plan.lock_path),
        "pid_path": str(plan.pid_path),
        "stop_flag_path": str(plan.stop_flag_path),
        "log_path": str(plan.log_path),
        "state": plan.state,
        "exit_code": plan.exit_code,
        "started_at": None if plan.started_at is None else plan.started_at.isoformat(),
    }


def _json(payload: object) -> str:
    """*payload* as indented JSON — the one encoding every ``-F json`` uses."""
    return json.dumps(payload, indent=2)


__all__ = [
    "BotDefaults",
    "BotPaths",
    "LogStream",
    "RestartOutcome",
    "StartOutcome",
    "StopOutcome",
    "render_info",
    "render_inspect",
    "render_logs",
    "render_ls",
    "render_quiet",
    "render_restart",
    "render_start",
    "render_stop",
]
