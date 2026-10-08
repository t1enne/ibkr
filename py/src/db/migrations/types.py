"""Migration value types — the data the framework operates on.

A ``Migration`` is DATA (a frozen record of name + two transitions), never a
subclass: the registry is an explicit ordered tuple, so there is no discovery, no
filename scan, and no base class whose ordering could change under a rename.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

from peewee import SqliteDatabase


class MigrationRefused(RuntimeError):
    """A migration declined to run, leaving the schema untouched.

    Every refusal path derives from this so a caller can report one actionable
    line instead of a traceback: the operator's next move is always to fix the
    precondition (adopt the book, verify a backup) and re-run.
    """


class IrreversibleMigrationError(MigrationRefused):
    """A ``down`` was requested for a migration that has no safe inverse.

    Raised LOUDLY and BEFORE any write: a partial unwind of a live book is worse
    than a refused one, because the operator cannot tell where it stopped.
    """


@dataclass(frozen=True)
class Migration:
    """One forward migration and (where meaningful) its inverse.

    ``up`` is a self-contained transition: it owns its own DDL/DML and must be
    idempotent, because an operator's DB may already have been migrated by hand
    before this framework existed (the baseline absorbs that case).

    ``down`` is ``None`` when no SAFE inverse exists. For this repo's live tables
    that is the common case: the policy is rename-never-drop, so the "inverse" of
    a re-key would be to drop the rows the re-key preserved. A refused ``down``
    beats a lossy one.

    ``is_live`` marks a migration over the durable live book. The runner uses it
    only for reporting; the registries are kept separate by the caller so the
    first-write path never replays a slow bulk-data migration.
    """

    name: str
    up: Callable[[SqliteDatabase], None]
    down: Callable[[SqliteDatabase], None] | None = None

    @property
    def reversible(self) -> bool:
        """Whether this migration can be unwound."""
        return self.down is not None


#: An ordered registry. Order is the contract — see ``versions/__init__``.
Registry: TypeAlias = tuple[Migration, ...]

__all__ = ["IrreversibleMigrationError", "Migration", "MigrationRefused", "Registry"]
