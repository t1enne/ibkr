"""Process-global sqlite handles: the data file and the live book.

Two module-level peewee instances, one per file (see :mod:`src.db.path`), plus a
raw ``sqlite3`` escape hatch for callers that only need SQL (the candle sync
path, ad-hoc probes). Both are WAL and both close at interpreter exit.

WHY PROCESS-GLOBAL, AND WHY ONLY HERE: peewee binds a model at CLASS level, so a
model can be pointed at exactly one database without a rebind. That is fine — and
correct — for ``symbol``/``candle``/``fundamental``, which have one concrete
target each. It is NOT fine for the ``live_*`` models: tests genuinely run two
ledgers on two paths inside one process, so the live models keep the per-instance
``bind_ctx`` pattern (see :mod:`src.live.models`) and a process-wide proxy would
make those two ledgers retarget each other. Hence two modules, two rules.
"""

from __future__ import annotations

import atexit
import sqlite3
from pathlib import Path
from typing import Optional

from peewee import SqliteDatabase

from src.db.path import resolve_db_path, resolve_live_db_path

#: WAL: a reader never blocks the writer, which is what lets a long candle read
#: run while the live cycle commits.
_PRAGMAS = {"journal_mode": "wal"}

#: The candle/research/symbol file. Every non-live model binds here.
db = SqliteDatabase(str(resolve_db_path()), pragmas=_PRAGMAS)

#: The durable live book. Only the migration runner touches it through this
#: handle; the ledger builds its OWN instance per path (see the module docstring).
live_db = SqliteDatabase(str(resolve_live_db_path()), pragmas=_PRAGMAS)

atexit.register(db.close)
atexit.register(live_db.close)


def get_connection(db_path: Optional[str | Path | None] = None) -> sqlite3.Connection:
    """A raw ``sqlite3`` connection to the candle database.

    The escape hatch for callers holding only SQL (no ORM models). Prefer the
    peewee ``db`` handle when models are involved: this returns a connection the
    caller must close.
    """
    return sqlite3.connect(str(resolve_db_path(db_path)))


def get_live_connection(
    db_path: Optional[str | Path | None] = None,
) -> sqlite3.Connection:
    """A raw ``sqlite3`` connection to the live book (see :func:`get_connection`)."""
    return sqlite3.connect(str(resolve_live_db_path(db_path)))


__all__ = ["db", "live_db", "get_connection", "get_live_connection"]
