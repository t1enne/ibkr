"""The migration framework: value types, bookkeeping, runner, DDL helpers.

Deliberately does NOT re-export ``versions``: the version modules import
``src.live``, so re-exporting them here would make ``import src.db`` pull in the
live package — the pkg-init cycle this layout exists to avoid. Import the registry
you need directly from :mod:`src.db.migrations.versions`.
"""

from __future__ import annotations

from src.db.migrations.bookkeeping import MIGRATION_TABLE
from src.db.migrations.helpers import add_column
from src.db.migrations.runner import (
    MigrationStatus,
    pending,
    run_down,
    run_pending,
    status,
)
from src.db.migrations.types import IrreversibleMigrationError, Migration, Registry

__all__ = [
    "MIGRATION_TABLE",
    "IrreversibleMigrationError",
    "Migration",
    "MigrationStatus",
    "Registry",
    "add_column",
    "pending",
    "run_down",
    "run_pending",
    "status",
]
