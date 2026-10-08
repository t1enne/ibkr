"""One path truth per sqlite file — the candle/research file and the live book.

Path resolution is a LEAF concern: nothing in this module imports from ``src.*``,
so any layer (models, migrations, CLI, tests) can ask for a path without
dragging a package initializer into the import graph.

Two files, deliberately separate (D-Q7):

* the **data** file holds ``symbol``/``candle``/``fundamental`` plus the TS-created
  ``kysely_migration`` — bulk, regenerable research data;
* the **live** file holds the ``live_*`` book — durable position state that no
  download can rebuild, and which a bloated or corrupt research write must not be
  able to take down with it.

Both default to the repo's ``data/`` directory resolved **file-relative** (never
``os.getcwd()``, which silently reads the wrong file from another directory) and
both honour an environment override so a test can point at a temp file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

#: Overrides the data file (candles/research). Tests set this.
_DB_PATH_ENV = "IBKR_DB_PATH"

#: Overrides the live book file. Separate from ``IBKR_DB_PATH`` so a test can
#: isolate the candle DB from the book (the two are unrelated lifetimes).
_LIVE_DB_PATH_ENV = "IBKR_LIVE_DB_PATH"

#: ``py/src/db/path.py`` -> ``py/src/db`` -> ``py/src`` -> ``py`` -> ``ibkr``.
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: The bulk candle/research/resampled-fundamentals file.
DEFAULT_DB_PATH = Path(
    os.environ.get(_DB_PATH_ENV) or _REPO_ROOT / "data" / "db.sqlite"
)

#: The durable live book (see module docstring for why it is its own file).
LIVE_DB_PATH = Path(
    os.environ.get(_LIVE_DB_PATH_ENV) or _REPO_ROOT / "data" / "live.db"
)


def resolve_db_path(override: Optional[str | Path] = None) -> Path:
    """The candle/research sqlite path: *override*, else ``DEFAULT_DB_PATH``.

    ``DEFAULT_DB_PATH`` is read at import, so an env override set afterwards
    needs a fresh process (or an explicit *override*).
    """
    return Path(override) if override is not None else DEFAULT_DB_PATH


def resolve_live_db_path(override: Optional[str | Path] = None) -> Path:
    """The live book sqlite path: *override*, else ``LIVE_DB_PATH``."""
    return Path(override) if override is not None else LIVE_DB_PATH
