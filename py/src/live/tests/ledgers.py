"""Migrated temp books for the live tests.

``SqliteLedger`` no longer bootstraps its schema: migrations are owned by
``src.db`` and applied by ``ibkr db migrate`` (``run_pending(LIVE_MIGRATIONS)``).
A ledger bound to an unmigrated file is expected to FAIL LOUD (``no such
table``), never to lazily build a half-schema — so every live test that needs a
usable book migrates its own temp file first, exactly as an operator would.

Tests that deliberately exercise an unmigrated or drifted file (a missing
table reads as an empty book, a drifted one raises ``LedgerReadError``) must
NOT use these helpers.
"""

from __future__ import annotations

from pathlib import Path

import peewee

from src.db.migrations.runner import run_pending
from src.db.migrations.versions import LIVE_MIGRATIONS
from src.live.ledger import SqliteLedger


def migrate_live(path: str | Path) -> None:
    """Apply every ``LIVE_MIGRATIONS`` entry to *path* (idempotent)."""
    db = peewee.SqliteDatabase(str(path))
    try:
        run_pending(db, LIVE_MIGRATIONS)
    finally:
        db.close()


def live_ledger(path: str | Path) -> SqliteLedger:
    """A :class:`SqliteLedger` on *path*, with the live schema already applied."""
    migrate_live(path)
    return SqliteLedger(path)


__all__ = ["live_ledger", "migrate_live"]
