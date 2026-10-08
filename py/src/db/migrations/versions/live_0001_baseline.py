"""``live_0001`` — baseline: absorb the live schema as it stands, idempotently.

The first half of the body is today's ``src.live.ledger_migration.migrate()``
lifted VERBATIM (guards intact). Those guards are not decoration — they are the
safety argument for a DB that was already hand-migrated before this framework
existed:

* ``_position_columns`` distinguishes pre-4.1 / conid-keyed / current, so a
  current book is never renamed away on every boot;
* the re-key steps early-return when the current key is present;
* ``_add_*_columns`` are ``IF NOT EXISTS``-style and additive;
* ``_free_legacy_name`` picks a free name, so a re-created legacy table is
  preserved rather than overwritten.

Do NOT "clean these up" to assume a version. An operator's DB has no
``peewee_migration`` row, so this migration WILL run against whatever shape that
file happens to be in, and every one of those guards is load-bearing there.

``up`` is the WHOLE transition, not half of one: it renames the legacy shapes to
their kept copies, builds the current schema, and then folds the copies' rows into
it. Splitting the fold out into the ledger would make the runner's bookkeeping
claim "applied" for a book that is still empty — the exact state that reads
downstream as FLAT and re-opens a live position.

``down`` is ``None`` in the registry: the transition renames and folds durable
book rows. Its only inverse would be to drop the preserved copies, which this repo
forbids — a refused unwind beats a lossy one.
"""

from __future__ import annotations

import logging

import peewee

from src.live.ledger_migration import (
    _restore_intents,
    _restore_positions,
    migrate,
)
from src.live.models import LIVE_MODELS

logger = logging.getLogger("src.live.ledger")

#: The name recorded in ``peewee_migration``. Pinned as a constant so the version
#: tuple and this module cannot disagree.
NAME = "live_0001_baseline"


def up(db: peewee.SqliteDatabase) -> None:
    """Absorb the live schema in place: re-key, create current, fold the copies in.

    ``bind_ctx`` matters here for the same reason as in the data baseline: the live
    models are permanently bound to a ``SqliteDatabase(None)`` template, so the
    binding is what points ``create_tables`` at the runner's file.
    """
    legacy = migrate(db)
    with db.bind_ctx(LIVE_MODELS):
        db.create_tables(LIVE_MODELS)
    _restore_positions(db, legacy)
    if legacy.intents is not None:
        _restore_intents(db, legacy.intents)


__all__ = ["NAME", "up"]
