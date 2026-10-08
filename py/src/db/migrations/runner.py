"""Migration runner — apply pending ``up``s, unwind ``down``s, report status.

Concurrency: the whole batch of pending ``up``s plus their bookkeeping writes runs
inside ONE ``db.atomic(lock_type="IMMEDIATE")``. ``BEGIN IMMEDIATE`` takes the
SQLite write lock up front, so two processes racing the same first migration
SERIALIZE: the second blocks, then re-reads the bookkeeping and finds nothing
pending. There is therefore no separate lock table — the write lock IS the mutex,
and a check-then-act window cannot exist because there is no separate check.

The same holds for :func:`run_down`: the unwinds and their ``unrecord`` calls share
one transaction, so a failure mid-unwind rolls the schema AND the bookkeeping back
together. A partially unwound schema with a partially emptied bookkeeping table is
the one state an operator could not reason about.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import peewee

from src.db.migrations import bookkeeping
from src.db.migrations.types import IrreversibleMigrationError, Registry


@dataclass(frozen=True)
class MigrationStatus:
    """One row of ``status``: the migration and whether its name is recorded."""

    name: str
    applied: bool
    reversible: bool


def pending(db: peewee.SqliteDatabase, registry: Registry) -> Registry:
    """The registry entries not yet recorded as applied, in registry order.

    A read only: no bookkeeping table is created, so ``status`` on a fresh file
    reports everything pending and writes nothing.
    """
    done = bookkeeping.applied_names(db)
    return tuple(m for m in registry if m.name not in done)


def run_pending(db: peewee.SqliteDatabase, registry: Registry) -> tuple[str, ...]:
    """Apply every pending migration in one immediate transaction; return names.

    A migration that raises aborts the whole batch and its bookkeeping row is
    never written, so the DB is left exactly as it was found.
    """
    todo = pending(db, registry)
    if not todo:
        return ()
    stamp = _now()
    with db.atomic(lock_type="IMMEDIATE"):
        # Inside the transaction: a batch that aborts must not leave a
        # half-created bookkeeping table behind either (the schema changes rolled
        # back, so the table would have nothing to describe).
        bookkeeping.ensure_table(db)
        for migration in todo:
            migration.up(db)
            bookkeeping.record(db, migration.name, stamp)
    return tuple(m.name for m in todo)


def run_down(
    db: peewee.SqliteDatabase, registry: Registry, target: str | None = None
) -> tuple[str, ...]:
    """Unwind applied migrations in reverse registry order; return the names.

    With no *target*, every applied migration is unwound newest-first. With a
    *target* (a migration name), the unwind stops AFTER that migration's ``down``
    runs, so *target* itself remains applied. An unknown or unapplied target is
    refused rather than silently unwinding everything.

    Raises :class:`IrreversibleMigrationError` — BEFORE any write — when the set to
    unwind contains a migration with no ``down``: refusing beats partially applying
    an unwind the operator cannot then reason about.
    """
    applied = bookkeeping.applied_names(db)
    ordered = tuple(m for m in registry if m.name in applied)
    if target is not None:
        indexes = [i for i, m in enumerate(ordered) if m.name == target]
        if not indexes:
            raise IrreversibleMigrationError(
                f"cannot unwind to {target!r}: it is not applied "
                f"(applied: {', '.join(m.name for m in ordered) or 'none'})"
            )
        ordered = ordered[: indexes[0] + 1]

    irreversible = [m.name for m in ordered if not m.reversible]
    if irreversible:
        raise IrreversibleMigrationError(
            "refusing to unwind: no safe inverse for "
            + ", ".join(reversed(irreversible))
            + " — unwinding live migrations would drop durable rows, and this "
            "repo's policy is rename-never-drop (restore from a backup instead)"
        )

    undo = tuple(reversed(ordered))
    if not undo:
        return ()
    with db.atomic(lock_type="IMMEDIATE"):
        for migration in undo:
            assert migration.down is not None  # guarded above
            migration.down(db)
            bookkeeping.unrecord(db, migration.name)
    return tuple(m.name for m in undo)


def status(
    db: peewee.SqliteDatabase, registry: Registry
) -> tuple[MigrationStatus, ...]:
    """Every registry entry with whether it is applied (read-only, writes nothing)."""
    done = bookkeeping.applied_names(db)
    return tuple(
        MigrationStatus(name=m.name, applied=m.name in done, reversible=m.reversible)
        for m in registry
    )


def _now() -> str:
    """ISO-8601 UTC timestamp for the bookkeeping row."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = ["MigrationStatus", "pending", "run_down", "run_pending", "status"]
