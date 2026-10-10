"""The ordered migration registries.

ORDER IS THE CONTRACT. Each registry is an explicit tuple, in the order the
migrations must run. There is no filename scan and no discovery: a rename cannot
silently reorder history, and a new migration is added by editing this file where
a reviewer sees it next to its neighbours.

TWO registries, never one (the live/data file split):

* :data:`LIVE_MIGRATIONS` — the durable live book in ``data/live.db``. It is one
  clean baseline that builds the target schema; the live file carries no legacy
  shape, so there is nothing to absorb or re-key.
* :data:`DATA_MIGRATIONS` — the candle/research file in ``data/ibkr.db``.

``ibkr db migrate`` runs both (one per file); ``ibkr db status`` reports both.
"""

from __future__ import annotations

from src.db.migrations.types import Migration, Registry
from src.db.migrations.versions import (
    data_0001_baseline,
    data_0002_drop_migrated_live_tables,
    live_0001_baseline,
)

#: The live book's history, oldest first. ``down`` is omitted where no safe
#: inverse exists (see each module's docstring).
LIVE_MIGRATIONS: Registry = (
    Migration(name=live_0001_baseline.NAME, up=live_0001_baseline.up, down=None),
)

#: The candle/research file's history, oldest first.
DATA_MIGRATIONS: Registry = (
    Migration(
        name=data_0001_baseline.NAME,
        up=data_0001_baseline.up,
        down=data_0001_baseline.down,
    ),
    Migration(
        name=data_0002_drop_migrated_live_tables.NAME,
        up=data_0002_drop_migrated_live_tables.up,
        down=None,
    ),
)

__all__ = ["DATA_MIGRATIONS", "LIVE_MIGRATIONS"]
