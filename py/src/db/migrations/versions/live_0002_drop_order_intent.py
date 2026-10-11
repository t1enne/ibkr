"""``live_0002`` — drop the durable open-order store (``live_order_intent``).

Step 1 of the broker-authoritative open-state change: the broker's reads alone
decide whether an order exists, so the durable intent table is no longer the
owner of OPEN state. ``live_execution`` (the fill fold) and ``live_strategy``
(ownership/audit) stay — only the open-order intent store is dropped.

This drop is IRREVERSIBLE: the rows are order-intent history that cannot be
reconstructed from the broker (which is the whole point of removing them), so
``down()`` is absent and ``run_down`` refuses loudly.

The ``up()`` runs only while the live file is a migrated book (the baseline's
bookkeeping row must be present) and no live cycle holds its lease (a mid-flight
cycle must never find its intent table gone). A fresh file drops through: the
baseline no longer creates the table, so there is nothing to lose.
"""

from __future__ import annotations

import os
import peewee

from src.db.migrations.types import MigrationRefused
from src.db.path import resolve_live_db_path

#: The name recorded in ``peewee_migration``.
NAME = "live_0002_drop_order_intent"

#: The migration whose bookkeeping row marks a live file as a migrated book.
_BOOK_MIGRATION = "live_0001_baseline"

#: The table being dropped.
_ORDER_INTENT_TABLE = "live_order_intent"


class OrderIntentDropRefused(MigrationRefused):
    """Dropping was declined because the book is not migrated or a cycle is live."""


def _live_file_is_migrated(db: peewee.SqliteDatabase) -> bool:
    """Whether *db* (the live file) carries the baseline's bookkeeping row."""
    applied = db.execute_sql(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='peewee_migration'"
    ).fetchone()
    if applied is None:
        return False
    return bool(
        db.execute_sql(
            "SELECT name FROM peewee_migration WHERE name=?", (_BOOK_MIGRATION,)
        ).fetchone()
    )


def _assert_no_cycle_in_flight() -> None:
    """Refuse while ANY live scope's cycle lease is held.

    Leases are per-scope advisory ``flock`` files named ``<db>.<scope_tag>.cycle.lock``
    beside the live file (``src.live.lease.lease_path``). A NON-blocking exclusive
    ``flock`` succeeds on a stale file (the kernel releases it when the holder's
    process exits), so only a genuinely held lease refuses. A mid-flight cycle
    must never find its intent table gone mid-read.
    """
    live_path = resolve_live_db_path()
    lock_files = (
        sorted(live_path.parent.glob(f"{live_path.name}.*.cycle.lock"))
        if live_path.parent.exists()
        else ()
    )
    for lock_file in lock_files:
        fd = os.open(lock_file, os.O_RDWR)
        try:
            try:
                os.lockf(fd, os.F_LOCK, 0)
            except OSError:
                raise OrderIntentDropRefused(
                    f"refusing to drop {_ORDER_INTENT_TABLE}: a live cycle holds the "
                    f"lease {lock_file} — no cycle may be mid-flight during the drop"
                ) from None
        finally:
            os.close(fd)


def up(db: peewee.SqliteDatabase) -> None:
    """Verify the book is migrated and idle, then drop ``live_order_intent``.

    Runs before the ``DROP``, so a refusal leaves the schema untouched. The drop
    is a no-op on a fresh file (the baseline no longer creates the table).
    """
    if not _live_file_is_migrated(db):
        raise OrderIntentDropRefused(
            f"refusing to drop {_ORDER_INTENT_TABLE}: {db.database} is not a "
            f"migrated live book (no {_BOOK_MIGRATION} bookkeeping row)"
        )
    _assert_no_cycle_in_flight()
    db.execute_sql(f"DROP TABLE IF EXISTS {_ORDER_INTENT_TABLE}")


__all__ = ["NAME", "OrderIntentDropRefused", "up"]
