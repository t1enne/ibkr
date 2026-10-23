"""Structured timestamped log output — compact tables for pipe/cron logs.

Every line starts with ``[YYYY-MM-DD HH:MM:SS]`` so a ``tail -f`` or ``cat``
reads like a real log. Styled with ``click.style`` when on a TTY; plain text
when piped. Tables via ``src.bt.table``. Use this everywhere a running process
writes progress or diagnostics to stdout/stderr — replaces raw ``print()``,
bare ``logger.warning(...)`` and ``click.echo`` without a timestamp.

The helpers are **thin wrappers** over a shared ``click.echo`` — they always
print to stderr unless the ``file`` param says otherwise.
"""

from __future__ import annotations

import datetime
import sys
from typing import IO

import click

from src.shared.style import Styler


def _ts() -> str:
    return datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def _styler(file: IO[str] | None = None) -> Styler:
    """ANSI on TTY, plain for pipe/cron."""
    f = file or sys.stderr
    return Styler(enabled=f.isatty())


#: Gates :func:`info` only. ``warn``/``error``/``retry``/``success`` always speak:
#: a quiet run must still surface what went wrong, and only the per-chunk
#: chatter is what a progress bar cannot share a terminal with.
_VERBOSE: bool = True


def set_verbose(enabled: bool) -> None:
    """Suppress (or restore) the ordinary ``info`` stream for this process.

    Set by a command that owns the terminal with a progress bar; the caller
    must restore it, so a later command in the same process still logs.
    """
    global _VERBOSE  # noqa: PLW0603
    _VERBOSE = enabled


def info(msg: str, *, file: IO[str] | None = None) -> None:
    """Timestamped info line — ordinary progress. Silent when not verbose."""
    if not _VERBOSE:
        return
    click.echo(_ts() + " " + msg, err=file is None, nl=True, file=file)


def warn(msg: str, *, file: IO[str] | None = None) -> None:
    """Timestamped warning."""
    s = _styler(file)
    click.echo(
        _ts() + " " + s.role(msg, "warn"),
        err=file is None,
        file=file,
    )


def error(msg: str, *, file: IO[str] | None = None) -> None:
    """Timestamped error."""
    s = _styler(file)
    click.echo(
        _ts() + " " + s.role(msg, "bad"),
        err=file is None,
        file=file,
    )


def retry(
    name: str, attempt: int, total: int, detail: str, *, file: IO[str] | None = None
) -> None:
    """Compact retry line — NO flood of 4 lines per attempt."""
    s = _styler(file)
    label = f"attempt {attempt}/{total}"
    if attempt == total:
        label = s.role("exhausted", "bad")
    click.echo(
        _ts() + " " + s.role("RETRY", "warn") + f" {name} {detail} ({label})",
        err=file is None,
        file=file,
    )


def success(name: str, detail: str = "", *, file: IO[str] | None = None) -> None:
    """A retried operation succeeded on a later attempt."""
    s = _styler(file)
    click.echo(
        _ts() + " " + s.role("OK", "good") + f" {name}{' ' + detail if detail else ''}",
        err=file is None,
        file=file,
    )


def step(
    done: int, total: int, label: str = "", *, file: IO[str] | None = None
) -> None:
    """Compact progress line — ``[HH:MM:SS] N/M label``."""
    pct = f"{done / total * 100:.0f}%" if total else ""
    click.echo(
        _ts() + f" {done}/{total} {pct} {label}".rstrip(),
        err=file is None,
        file=file,
    )
