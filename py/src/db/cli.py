"""``ibkr db`` — schema migrations for both sqlite files.

Four commands:

* ``ibkr db migrate``          — apply pending migrations to BOTH files.
* ``ibkr db migrate --down``   — unwind (requires ``--yes``; refuses an
  irreversible target rather than partially applying).
* ``ibkr db migrate --dry-run``— list what WOULD run; write nothing.
* ``ibkr db status``           — report every migration and whether it is applied.
* ``ibkr db preflight``        — report the data file's series-key problems.

``--dry-run`` on ``migrate`` writes nothing at all: it lists names only, so it
cannot reveal a preflight refusal — ``db preflight`` runs the same checks and
prints the offending rows, also writing nothing.
"""

from __future__ import annotations

import click
from peewee import SqliteDatabase

from src.db.migrations import preflight
from src.db.migrations.runner import pending, run_down, run_pending, status
from src.db.migrations.types import MigrationRefused, Registry
from src.db.migrations.versions import DATA_MIGRATIONS, LIVE_MIGRATIONS
from src.db.path import resolve_db_path, resolve_live_db_path


@click.group(name="db")
def db_group() -> None:
    """Schema migrations for both sqlite files."""


def _files() -> tuple[tuple[str, Registry], ...]:
    """The (label, registry) pairs, with each file's path resolved lazily.

    A function rather than a constant so a ``IBKR_DB_PATH``/``IBKR_LIVE_DB_PATH``
    override applies to the process that actually runs the command.
    """
    return (
        ("data", DATA_MIGRATIONS),
        ("live", LIVE_MIGRATIONS),
    )


def _path_for(label: str) -> str:
    """The resolved sqlite path for a registry label."""
    return str(resolve_db_path() if label == "data" else resolve_live_db_path())


def _connect(path: str) -> SqliteDatabase:
    """A WAL peewee handle on *path* (created on first write, never on connect)."""
    return SqliteDatabase(path, pragmas={"journal_mode": "wal"})


@db_group.command("migrate")
@click.option(
    "--down",
    "down",
    is_flag=True,
    help="Unwind applied migrations instead of applying pending ones.",
)
@click.option(
    "--target",
    default=None,
    help="With --down: stop after unwinding down to (and including) this name.",
)
@click.option(
    "--yes",
    "confirmed",
    is_flag=True,
    help="Required for --down: acknowledge that unwinding rewrites the schema.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="List what would run; write nothing.",
)
def db_migrate(down: bool, target: str | None, confirmed: bool, dry_run: bool) -> None:
    """Apply pending migrations (or unwind with ``--down``) on both files.

    The two files carry SEPARATE histories and are migrated separately: the live
    book must never be made to wait behind a bulk candle rewrite.
    """
    if down and not confirmed:
        raise click.UsageError(
            "--down rewrites the schema of a durable file. Verify a backup exists, "
            "then re-run with --yes."
        )
    rc = 0
    for label, registry in _files():
        path = _path_for(label)
        db = _connect(path)
        try:
            if dry_run:
                names = [m.name for m in pending(db, registry)]
                click.echo(f"{label:4} {path}: {len(names)} pending {names}")
            elif down:
                names = run_down(db, registry, target)
                click.echo(f"{label:4} {path}: unwound {list(names)}")
            else:
                names = run_pending(db, registry)
                click.echo(f"{label:4} {path}: applied {list(names)}")
        except MigrationRefused as exc:
            click.echo(f"{label:4} {path}: {exc}", err=True)
            rc = 1
        finally:
            db.close()
    if rc:
        raise SystemExit(rc)


@db_group.command("status")
def db_status() -> None:
    """Report every migration and whether its name is recorded as applied."""
    for label, registry in _files():
        path = _path_for(label)
        db = _connect(path)
        try:
            for row in status(db, registry):
                mark = "applied" if row.applied else "pending"
                rev = "reversible" if row.reversible else "irreversible"
                click.echo(f"{label:4} {row.name:24} {mark:8} {rev}")
        finally:
            db.close()


@db_group.command("preflight")
def db_preflight() -> None:
    """Report the data file's series-key problems; write nothing.

    The same check ``data_0003`` refuses on, printed instead of raised: an
    operator sees the offending rows before deciding what to fix. Exits non-zero
    when anything is found, so it also works as a gate.
    """
    # Lazy on purpose: ``src.db`` must not import ``src.data`` at module scope
    # (see src/db/__init__). This command is the composition edge that reads the
    # LIVE venue vocabulary, so it asks for it only when it runs.
    from src.data.venue import try_series_key

    path = _path_for("data")
    db = _connect(path)
    try:
        findings = preflight.report(db, try_series_key)
    finally:
        db.close()

    if findings.ok:
        click.echo(f"data {path}: series keys clean")
        return
    for line in findings.lines():
        click.echo(f"data {path}: {line}")
    raise SystemExit(1)


db_group.add_command(db_migrate)
db_group.add_command(db_status)
db_group.add_command(db_preflight)

__all__ = ["db_group"]
