"""The database layer: paths, connections, models, introspection, migrations.

Layout rationale and the one rule that matters for importing it:

* :mod:`src.db.path`, :mod:`src.db.connection`, :mod:`src.db.models`,
  :mod:`src.db.introspect` are LEAVES — they import nothing from ``src.*``, so any
  layer can use them without dragging a package initializer into the import graph.
* :mod:`src.db.migrations` is the framework and is likewise independent.
* ONLY ``src.db.migrations.versions.**`` may reach back into ``src.live`` /
  ``src.data`` (a migration must describe the shapes it is migrating FROM).

This module therefore re-exports only the leaves and the framework. It must NOT
re-export ``migrations.versions.*`` — doing so would import ``src.live`` on every
``import src.db`` and re-create the package-initializer cycle this layout exists to
remove.
"""

from __future__ import annotations

from src.db.connection import db, get_connection, get_live_connection, live_db
from src.db.introspect import primary_key_columns, table_columns, table_exists
from src.db.models import (
    CANDLE_MODELS,
    DATA_MODELS,
    CandleSchema,
    FundamentalSchema,
    SymbolSchema,
)
from src.db.path import (
    DEFAULT_DB_PATH,
    LIVE_DB_PATH,
    resolve_db_path,
    resolve_live_db_path,
)

__all__ = [
    "CANDLE_MODELS",
    "CandleSchema",
    "DATA_MODELS",
    "DEFAULT_DB_PATH",
    "FundamentalSchema",
    "LIVE_DB_PATH",
    "SymbolSchema",
    "db",
    "get_connection",
    "get_live_connection",
    "live_db",
    "primary_key_columns",
    "resolve_db_path",
    "resolve_live_db_path",
    "table_columns",
    "table_exists",
]
