"""``live_0001_baseline`` — build the live book's schema.

The live file is disposable and carries no legacy shape: this baseline simply
creates the target tables from :data:`src.live.models.LIVE_MODELS`.
``create_tables`` is ``IF NOT EXISTS``, so re-running against an already-migrated
file is a no-op and the migration is idempotent on the fresh case.

The ledger writes NO DDL of its own — ``ibkr db migrate`` is the only DDL owner —
so this migration is the ONE place the live schema is created. A ledger bound to
an unmigrated file fails loud (``no such table``) rather than building a
half-schema under a live order.

``down`` is ``None``: dropping the durable book is not a safe inverse, and a
refused unwind beats a lossy one.
"""

from __future__ import annotations

import peewee

from src.live.models import LIVE_MODELS

#: The name recorded in ``peewee_migration``.
NAME = "live_0001_baseline"


def up(db: peewee.SqliteDatabase) -> None:
    """Create every live table (idempotent, ``CREATE TABLE IF NOT EXISTS``).

    The models are bound to a placeholder database, so the ``bind_ctx`` is what
    makes ``create_tables`` target the file the RUNNER is working on.
    """
    with db.bind_ctx(LIVE_MODELS):
        db.create_tables(LIVE_MODELS)


__all__ = ["NAME", "up"]
