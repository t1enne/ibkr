"""The bot's only writer of its own log (plan §3).

One append-only file per scope, self-rotating by size, so a long-running
deployment has no unbounded growth to fix out of band. The writer is a seam: a
test injects its own to capture lines without a file. A dry run needs no fake —
it spawns no child, so no journal exists at all.

A write that fails never changes a cycle's exit code — the pid row is the
authority and the failure is swallowed to stderr, exactly as the bare
``cycle exit N`` writer did before it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import pandas as pd
from click import echo

from src.bot.logschema import (
    DEFAULT_LOG_KEEP,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_MAX_BYTES,
    LOG_KEEP_ENV,
    LOG_LEVEL_ENV,
    LOG_MAX_BYTES_ENV,
    Level,
    LogLine,
    env_of,
    level_at_least,
    level_of,
    render_line,
    resolve_log_keep,
    resolve_log_level,
    resolve_log_max_bytes,
)
from src.bot.registry import BotRegistry, runtime_dir
from src.bot.types import BotLiveness


class Journal(Protocol):
    """The bot's only writer of its own log (a capture fake is the test seam)."""

    def emit(self, level: Level, event: str, fields: Mapping[str, str]) -> None:
        """Append one structured line (a no-op when filtering or silenced)."""
        ...


def _utc_now() -> pd.Timestamp:
    """The wall clock, UTC — the ONE clock a log line's timestamp comes from."""
    return pd.Timestamp.now(tz="UTC")


@dataclass(frozen=True)
class FileJournal:
    """Append-only, size-rotated writer for one scope's log file.

    ``scope`` and ``pid`` ride on every line so an interleaved read stays
    self-identifying; a caller-supplied field with either name is overwritten by
    the real one (identity is never a caller's opinion).
    """

    path: Path
    scope: str
    pid: int
    min_level: Level = DEFAULT_LOG_LEVEL
    max_bytes: int = DEFAULT_LOG_MAX_BYTES
    keep: int = DEFAULT_LOG_KEEP
    clock: Callable[[], pd.Timestamp] = field(default=_utc_now)

    def emit(self, level: Level, event: str, fields: Mapping[str, str]) -> None:
        """Append one line, rotating first when the file is over its bound."""
        try:
            if not level_at_least(level_of(level), self.min_level):
                return
            line = render_line(
                LogLine(
                    at=self.clock(),
                    level=level_of(level),
                    event=event,
                    fields=_identified(fields, self.scope, self.pid),
                )
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            rotate(self.path, max_bytes=self.max_bytes, keep=self.keep)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            # A log we cannot write must not stop the bot: the pid row records
            # the same code and liveness is the lock, not the log.
            echo(f"bot {self.scope}: {event} (log write failed: {exc})", err=True)


def _identified(fields: Mapping[str, str], scope: str, pid: int) -> dict[str, str]:
    """*fields* plus the identity pair, with ``scope``/``pid`` forced to ours."""
    merged = {key: str(value) for key, value in fields.items()}
    merged["scope"] = scope
    merged["pid"] = str(pid)
    return merged


def rotate(path: Path, *, max_bytes: int, keep: int) -> None:
    """Roll *path* over when it is at or over *max_bytes*.

    ``.log`` -> ``.log.1`` -> ``.log.2`` … keeping *keep* generations; ``keep``
    of ``0`` truncates instead, and ``max_bytes`` of ``0`` disables rotation
    entirely. Renaming (not copying) keeps a rotation's cost independent of the
    file's size, and a follower sees the file shrink and reopens from zero.
    """
    assert max_bytes >= 0, "a byte bound is non-negative"
    assert keep >= 0, "a generation count is non-negative"
    if max_bytes == 0:
        return
    try:
        size = path.stat().st_size
    except OSError:
        return
    if size < max_bytes:
        return
    if keep == 0:
        path.write_bytes(b"")
        return
    # Oldest first: drop the generation that falls off the end, then shift down.
    path.with_name(f"{path.name}.{keep}").unlink(missing_ok=True)
    for index in range(keep, 0, -1):
        source = path if index == 1 else path.with_name(f"{path.name}.{index - 1}")
        if source.exists():
            source.replace(path.with_name(f"{path.name}.{index}"))


def log_path_of(scope: str) -> Path:
    """``<runtime>/<scope_tag>.log`` for the DEPLOYMENT's runtime dir.

    ONE implementation: ``BotRegistry.log_path`` owns the path shape, and this
    is it evaluated against :func:`runtime_dir` at call time. A caller that
    already holds a registry passes it to :func:`build_journal` instead, so an
    explicit-runtime registry can never split the structured log from the file
    ``logs``/``inspect`` read.
    """
    return BotRegistry(_null_db(), runtime=runtime_dir()).log_path(scope)


def _null_db() -> str:
    """A throwaway db argument: ``log_path`` never reads it."""
    return ""


def build_journal(
    scope: str,
    pid: int,
    registry: BotLiveness,
    level: Level | None = None,
) -> Journal:
    """The resident bot's journal: one file per scope, bounded and self-rotating.

    The bounds and the level are read from the environment at CALL time, so a
    deployment changes them with an env var and a restart, and a test repoints
    them in-process. The path comes from *registry*'s own rule, so the writer and
    ``bot logs`` cannot disagree about which file is the bot's.
    """
    return FileJournal(
        path=registry.log_path(scope),
        scope=scope,
        pid=pid,
        min_level=level if level is not None else _env_level(),
        max_bytes=resolve_log_max_bytes(env_of(LOG_MAX_BYTES_ENV)),
        keep=resolve_log_keep(env_of(LOG_KEEP_ENV)),
    )


def _env_level() -> Level:
    """``IBKR_BOT_LOG_LEVEL``, defaulting to INFO (a typo raises, never guesses)."""
    return resolve_log_level(env_of(LOG_LEVEL_ENV))


__all__ = [
    "FileJournal",
    "Journal",
    "build_journal",
    "log_path_of",
    "rotate",
]
