"""peewee models for the candle/research file.

Every model with a single concrete target belongs here (see
:mod:`src.db.connection` for the split rationale): the migration framework can
only create, index or migrate a table whose model it can import. The ``live_*``
models are per-instance and live in :mod:`src.live.models`.

:data:`DATA_MODELS` is the whole file's schema in one tuple — what a baseline
creates and what a migration reasons about.
"""

from __future__ import annotations

from peewee import Model

from src.db.models.candles import CANDLE_MODELS, CandleSchema, SymbolSchema
from src.db.models.fundamentals import (
    FUNDAMENTAL_MODELS,
    FundamentalSchema,
    NATURAL_KEY_COLUMNS,
    NATURAL_KEY_DDL,
    NATURAL_KEY_INDEX,
    ensure_natural_key_index,
)

#: Every table in the candle/research file. ``create_tables`` builds these; the
#: natural-key index on ``fundamental`` is created separately (see
#: :func:`~src.db.models.fundamentals.ensure_natural_key_index`).
DATA_MODELS: tuple[type[Model], ...] = CANDLE_MODELS + FUNDAMENTAL_MODELS

__all__ = [
    "CANDLE_MODELS",
    "CandleSchema",
    "DATA_MODELS",
    "FUNDAMENTAL_MODELS",
    "FundamentalSchema",
    "NATURAL_KEY_COLUMNS",
    "NATURAL_KEY_DDL",
    "NATURAL_KEY_INDEX",
    "SymbolSchema",
    "ensure_natural_key_index",
]
