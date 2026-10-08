"""peewee model for the ``fundamental`` table (sparse fiscal rows).

Lives here with ``symbol``/``candle`` for one reason: the migration framework owns
every table in the candle/research file, and a model defined outside
:mod:`src.db` is one the baseline cannot create, index or migrate. The model binds
to the ONE concrete process-global :data:`src.db.connection.db` — this table has a
single target in the process, so the per-instance rebind the ``live_*`` models need
buys nothing here.

The row shape is one row per ``(ticker, statement, field, period)`` filing rather
than a daily grid, so a restatement stays a separate row and as-first-stated PIT
falls out of the data model. The normalized statement dataclasses and the
``FundamentalRow`` value type are DOMAIN types, not storage, and stay in
:mod:`src.data.fundamentals.schema`.
"""

from __future__ import annotations

from peewee import CharField, FloatField, IntegerField, Model, SqliteDatabase

from src.db.connection import db

#: Natural key of a stated fact, enforced UNIQUE — the constraint that makes
#: :func:`src.data.fundamentals.schema.insert_fundamentals` idempotent, mirroring
#: the candle table's ticker+timestamp index. ``form`` is deliberately EXCLUDED:
#: ``filed`` already disambiguates two filings of one period, while a period-only
#: key would collide the original with its restatement and destroy the PIT
#: distinction.
NATURAL_KEY_COLUMNS: tuple[str, ...] = (
    "ticker",
    "statement",
    "field",
    "period_start",
    "period_end",
    "filed",
)

#: Name of the UNIQUE index backing :data:`NATURAL_KEY_COLUMNS`.
NATURAL_KEY_INDEX = "fundamental_natural_key_idx"

#: Peewee derives index names from the table+columns (an overlong hashed name), so
#: the index is created explicitly under a stable name instead. The DDL lives next
#: to :data:`NATURAL_KEY_COLUMNS` so the two cannot drift apart.
NATURAL_KEY_DDL = (
    f"CREATE UNIQUE INDEX IF NOT EXISTS {NATURAL_KEY_INDEX} "
    f"ON fundamental ({', '.join(NATURAL_KEY_COLUMNS)})"
)

#: Delete every row sharing a natural key except the lowest ``id`` (the first
#: write, which is what the current insert would have kept anyway). A grouped
#: subquery rather than a window function, so it runs on any SQLite.
_COLLAPSE_DUPES = (
    "DELETE FROM fundamental WHERE id NOT IN ("
    "SELECT MIN(id) FROM fundamental "
    f"GROUP BY {', '.join(NATURAL_KEY_COLUMNS)})"
)


class FundamentalSchema(Model):
    """Sparse fiscal fundamentals row."""

    ticker = CharField()
    statement = CharField()
    field = CharField()
    value = FloatField()
    period_start = IntegerField()
    period_end = IntegerField()
    filed = IntegerField()
    form = CharField()

    class Meta:
        database = db
        table_name = "fundamental"


#: The models this module owns, for a migration or a ``create_tables`` call.
FUNDAMENTAL_MODELS: tuple[type[Model], ...] = (FundamentalSchema,)


def collapse_duplicate_keys(db_conn: SqliteDatabase) -> int:
    """Remove exact-duplicate natural-key rows; return how many were removed.

    Needed only to make the UNIQUE index creatable on a table that predates it.
    Safe when there is nothing to collapse (the DELETE matches no rows) and on a
    table created in the same call (it is empty).
    """
    removed = db_conn.execute_sql(_COLLAPSE_DUPES).rowcount
    return int(removed) if removed and removed > 0 else 0


def ensure_natural_key_index(db_conn: SqliteDatabase) -> int:
    """Collapse duplicate keys, then create the natural-key UNIQUE index.

    The index is created separately from ``create_tables`` because that call does
    not ALTER an existing table: a DB written before the constraint existed already
    has the table and would otherwise never gain the constraint that makes the
    insert idempotent. A table that predates it may also hold the very duplicates
    the constraint forbids, which would make ``CREATE UNIQUE INDEX`` fail — those
    are collapsed first. ``IF NOT EXISTS`` keeps the whole thing idempotent and
    safe on a table created in the same call.

    Returns the number of duplicate rows removed.
    """
    removed = collapse_duplicate_keys(db_conn)
    db_conn.execute_sql(NATURAL_KEY_DDL)
    return removed


__all__ = [
    "FUNDAMENTAL_MODELS",
    "FundamentalSchema",
    "NATURAL_KEY_COLUMNS",
    "NATURAL_KEY_DDL",
    "NATURAL_KEY_INDEX",
    "collapse_duplicate_keys",
    "ensure_natural_key_index",
]
