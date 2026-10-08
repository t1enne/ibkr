"""The ordered migration registries.

ORDER IS THE CONTRACT. Each registry is an explicit tuple, in the order the
migrations must run. There is no filename scan and no discovery: a rename cannot
silently reorder history, and a new migration is added by editing this file where
a reviewer sees it next to its neighbours.

TWO registries, never one (D-Q7 / the file split):

* :data:`LIVE_MIGRATIONS` — the durable live book. The ledger's first-write path
  runs THIS and only this, because the live path must never replay a slow bulk
  migration (a multi-million-row candle rewrite) before it can place an order.
* :data:`DATA_MIGRATIONS` — the candle/research file.

``ibkr db migrate`` runs both (one per file); ``ibkr db status`` reports both.
"""

from __future__ import annotations

from src.db.migrations.types import Migration, Registry
from src.db.migrations.versions import (
    data_0001_baseline,
    data_0002_drop_migrated_live_tables,
    live_0001_baseline,
    live_0002_drop_legacy_artifacts,
)

#: The live book's history, oldest first. ``down`` is omitted where no safe
#: inverse exists (see each module's docstring).
LIVE_MIGRATIONS: Registry = (
    Migration(name=live_0001_baseline.NAME, up=live_0001_baseline.up, down=None),
    Migration(
        name=live_0002_drop_legacy_artifacts.NAME,
        up=live_0002_drop_legacy_artifacts.up,
        down=None,
    ),
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
