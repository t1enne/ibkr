"""``data_0001`` — baseline: claim the candle/research tables already there.

``symbol`` and ``candle`` were created by the TypeScript data pipeline (they exist
on any DB this repo reads) and ``fundamental`` by the earlier bootstrap call; their
columns already match :mod:`src.db.models`, so this migration's job is not to build
them but to mark them as owned and current — which is what makes the framework's
bookkeeping agree with reality on an existing file.

``create_tables`` is ``IF NOT EXISTS``, so running it against existing tables is a
no-op, and it is what creates them on a genuinely fresh file.

``fundamental`` carries a SECOND piece of schema ``create_tables`` cannot express:
its natural-key UNIQUE index, which is what makes the fundamentals insert
idempotent. So the migration also runs
:func:`~src.db.models.fundamentals.ensure_natural_key_index`, which collapses any
duplicate keys a pre-constraint table already holds (otherwise ``CREATE UNIQUE
INDEX`` would fail) and then creates the index under its stable name. Both steps are
idempotent, so an already-indexed file is untouched.

NOTE: this migration is already recorded as applied on the operator's DB, so edits
to ``up`` reach only a FRESH file. The operator's ``fundamental`` table already
carries ``fundamental_natural_key_idx``, so there is nothing outstanding there —
but a schema change that must reach existing files belongs in a NEW migration, never
in an edit to this one.

``down`` is a genuine no-op, not a refusal: this migration changed no durable row,
so undoing it means only forgetting the bookkeeping row. That is the one place in
this repo where a round trip is meaningful — it never existed for the live book,
whose migrations all rename or copy rows.
"""

from __future__ import annotations

import peewee

from src.db.models import DATA_MODELS, ensure_natural_key_index

NAME = "data_0001_baseline"


def up(db: peewee.SqliteDatabase) -> None:
    """Create the candle/research tables and the fundamentals natural-key index.

    The models are permanently bound to the process-global data handle, so the
    ``bind_ctx`` is what makes ``create_tables`` target the db the RUNNER is
    working on (the CLI may be pointed at a different file).
    """
    with db.bind_ctx(DATA_MODELS):
        db.create_tables(DATA_MODELS)
        ensure_natural_key_index(db)


def down(db: peewee.SqliteDatabase) -> None:
    """No-op: this migration owns no rows and drops nothing.

    Deliberately does NOT drop the tables. An operator unwinding the bookkeeping is
    asking to forget a marker, not to destroy the bulk data the TS pipeline and the
    fundamental ingest filled.
    """
