"""Shared plumbing for the sqlite ledger and its mixins.

The ledger's models and behaviour are split across modules (``ledger`` and
``ledger_migration``) so each class stays under the repo's size budget. This
module owns the pieces ALL of them need and that have no natural home: the peewee
template binding, the base model, the read-failure type, the raw ``sqlite_master``
/ ``PRAGMA`` helpers, and the epoch-millisecond codec.

The template binding is a PLACEHOLDER: the model CLASSES are defined against it,
but each ``SqliteLedger`` rebinds them (via ``bind_ctx``) to its OWN
``SqliteDatabase`` for every operation, so two ledgers on two paths never share a
connection or retarget each other — peewee binds a model at CLASS level, so a
single module-global database cannot serve two live instances.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import cast

import pandas as pd
import peewee
from peewee import Model, SqliteDatabase

_TEMPLATE_DB = SqliteDatabase(None)


class _Base(Model):
    class Meta:
        database = _TEMPLATE_DB


class _SqliteOps:
    """The per-instance seam the ledger mixins build on: a bound db + write txn.

    Declared here so the model-group mixins (the book/intent mixins in
    ``ledger``) can call ``self._write()`` without importing the
    concrete ``SqliteLedger`` (which would be circular). The concrete ledger
    supplies both; the stubs are never reached.
    """

    _database: SqliteDatabase

    def _write(self) -> AbstractContextManager[None]:
        raise NotImplementedError  # provided by the concrete SqliteLedger


class LedgerReadError(RuntimeError):
    """A durable read failed for a reason OTHER than the table being absent.

    Only a genuinely missing table (the ``--dry-run`` case) means "unwritten";
    a lock, a corrupt image or a shape-drifted column must fail loudly, never be
    reported as an empty book — an empty book reads downstream as "flat" and
    re-opens everything.
    """


def _is_missing_table(error: peewee.OperationalError) -> bool:
    """Whether *error* is the genuine ``no such table`` (dry-run) case.

    peewee wraps every ``sqlite3.OperationalError`` (a lock, a bad column, a
    missing table) as the same ``peewee.OperationalError`` type, so the message
    is the only discriminator left.
    """
    return "no such table" in str(error)


def _table_exists(db: SqliteDatabase, name: str) -> bool:
    """Whether *name* is a table in *db*."""
    return bool(
        db.execute_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchall()
    )


def _table_columns(db: SqliteDatabase, name: str) -> set[str]:
    """The column names of *name* in *db* (empty when the table does not exist)."""
    return {str(row[1]) for row in db.execute_sql(f"PRAGMA table_info({name})")}


def _primary_key_columns(db: SqliteDatabase, name: str) -> set[str]:
    """The primary-key column names of *name* (from ``PRAGMA table_info``)."""
    return {
        str(row[1]) for row in db.execute_sql(f"PRAGMA table_info({name})") if row[5]
    }


# Timestamps round-trip through INTEGER epoch milliseconds — the same clock the
# candle table uses, so a book row and a bar are comparable without a tz step.
_MS = 1000


def _ms(ts: pd.Timestamp) -> int:
    """Epoch-milliseconds of *ts*.

    A tz-naive input is read as LOCAL time, so the stored integer is the absolute
    instant; :func:`_ts` reads it back tagged UTC. The round trip preserves the
    absolute time, not the tz label (display tz differs, the instant does not).
    """
    return int(ts.timestamp() * _MS)


def _ts(value: int | None) -> pd.Timestamp | None:
    """Epoch-ms -> UTC ``Timestamp`` (see :func:`_ms`: absolute instant preserved)."""
    if value is None:
        return None
    return cast("pd.Timestamp", pd.Timestamp(value, unit="ms", tz="UTC"))
