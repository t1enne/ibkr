"""peewee models for the candle/research file: ``symbol`` and ``candle``.

Both bind to the ONE concrete process-global :data:`src.db.connection.db`, created
from :func:`src.db.path.resolve_db_path`. There is no per-instance rebind here:
these two tables have exactly one target in the process, so a single class-level
binding is the simplest correct thing.

The path used to be ``os.getcwd()/".."/"data"/"db.sqlite"`` (see
:mod:`src.data.types` history): that read or WROTE a different file depending on
the caller's working directory. Resolving it file-relative fixes that, and is a
deliberate behaviour change.
"""

from __future__ import annotations

from peewee import CharField, FloatField, IntegerField, Model

from src.db.connection import db


class SymbolSchema(Model):
    conid = IntegerField(primary_key=True)
    ticker = CharField()
    name = CharField(null=True)
    market = CharField()
    currency = CharField()

    class Meta:
        database = db
        table_name = "symbol"


class CandleSchema(Model):
    conid = IntegerField()
    ticker = CharField()
    timestamp = IntegerField()
    open = FloatField()
    high = FloatField()
    low = FloatField()
    close = FloatField()
    volume = FloatField()

    class Meta:
        database = db
        table_name = "candle"


#: The models this module owns, for a migration or a ``create_tables`` call that
#: needs all of them in one place.
CANDLE_MODELS: tuple[type[Model], ...] = (SymbolSchema, CandleSchema)

__all__ = ["CandleSchema", "CANDLE_MODELS", "SymbolSchema"]
