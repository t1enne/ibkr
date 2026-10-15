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

from src.bt.table import Col, Table, render as render_table
from src.shared.style import Styler


def _ts() -> str:
    return datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def _styler(file: IO[str] | None = None) -> Styler:
    """ANSI on TTY, plain for pipe/cron."""
    f = file or sys.stderr
    return Styler(enabled=f.isatty())


def info(msg: str, *, file: IO[str] | None = None) -> None:
    """Timestamped info line — ordinary progress."""
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


def gap_table(
    heading: str,
    rows: list[tuple[str, str, str]],
    *,
    file: IO[str] | None = None,
) -> None:
    """A compact table of gaps — one row per symbol, 3 cols.

    ``rows`` is ``(symbol, gap_count, gap_range_or_reason)``. Renders as a
    titled table so 200 lines become ~10.
    """
    s = _styler(file)
    cols = (
        Col("symbol", role="strong"),
        Col("gaps", ">"),
        Col("range"),
    )
    table_lines = render_table(
        Table(columns=cols, rows=tuple(rows)),
        sep="  ",
        styler=s,
    )
    out = [_ts() + " " + s.role(heading + ":", "title")]
    out.extend(table_lines)
    click.echo("\n".join(out), err=file is None, file=file)


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
