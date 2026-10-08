"""The repo's ``config.toml`` — operator defaults that are not secrets.

Secrets stay in ``.env`` (``IBKR_USERNAME``/``IBKR_PASSWORD``); this file holds
the non-secret defaults a CLI cannot guess, so a default lives in ONE place
instead of a literal repeated per call site.

Resolution is a LEAF concern: nothing here imports from ``src.*``, so any layer
can read a setting without dragging a package initializer into the import graph.
The path is resolved **file-relative** (never ``os.getcwd()``, which silently
reads another directory's file) and honours ``IBKR_CONFIG_PATH`` so a test can
point at a temp file.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import cast

#: Overrides the settings file. Tests set this.
_CONFIG_PATH_ENV = "IBKR_CONFIG_PATH"

#: ``py/src/config.py`` -> ``py/src`` -> ``py``.
_ROOT = Path(__file__).resolve().parents[1]

#: The operator settings file (read at import, like the sqlite paths).
CONFIG_PATH = Path(os.environ.get(_CONFIG_PATH_ENV) or _ROOT / "config.toml")

#: Adapters a live run may name.
_ADAPTERS = ("ibkr", "sim")

#: Used when the file, the table or the key is absent — the file is an override
#: surface, not a requirement.
DEFAULT_LIVE_ADAPTER = "ibkr"


def live_adapter() -> str:
    """``[live] adapter`` from ``config.toml``, else :data:`DEFAULT_LIVE_ADAPTER`.

    A missing file or table is the default, never an error. A value outside the
    adapter vocabulary raises: a typo must not silently trade the wrong side.
    """
    value = _table("live").get("adapter", DEFAULT_LIVE_ADAPTER)
    if value not in _ADAPTERS:
        raise ValueError(
            f"live.adapter must be one of {sorted(_ADAPTERS)}, got {value!r}"
        )
    return cast("str", value)


def _table(name: str) -> Mapping[str, object]:
    """The ``[name]`` table of :data:`CONFIG_PATH`, or ``{}`` when absent."""
    try:
        with CONFIG_PATH.open("rb") as handle:
            parsed = cast("Mapping[str, object]", tomllib.load(handle))
    except FileNotFoundError:
        return {}
    table = parsed.get(name, {})
    if not isinstance(table, dict):
        raise ValueError(f"[{name}] must be a table in {CONFIG_PATH}")
    return cast("Mapping[str, object]", table)
