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
from dataclasses import dataclass
from pathlib import Path
from typing import cast as _cast


#: Overrides the settings file. Tests set this.
_CONFIG_PATH_ENV = "IBKR_CONFIG_PATH"

#: ``py/src/config.py`` -> ``py/src`` -> ``py``.
_ROOT = Path(__file__).resolve().parents[1]

#: The operator settings file (read at import, like the sqlite paths).
CONFIG_PATH = Path(os.environ.get(_CONFIG_PATH_ENV) or _ROOT / "config.toml")


def _table(name: str) -> Mapping[str, object]:
    """The ``[name]`` table of :data:`CONFIG_PATH`, or ``{}`` when absent."""
    try:
        with CONFIG_PATH.open("rb") as handle:
            parsed = _cast("Mapping[str, object]", tomllib.load(handle))
    except FileNotFoundError:
        return {}
    table = parsed.get(name, {})
    if not isinstance(table, dict):
        raise ValueError(f"[{name}] must be a table in {CONFIG_PATH}")
    return _cast("Mapping[str, object]", table)


@dataclass(frozen=True)
class BotConfig:
    """``[bot]`` section values, all optional. ``None`` = absent from file.

    Entrypoint code merges this with CLI flags + code defaults.
    """

    interval: int | None = None  # cadence seconds
    max_age: int | None = None  # staleness days
    retries: int | None = None  # transient retry count
    no_sync: bool | None = None  # skip pre-cycle candle sync


def read_bot_config() -> BotConfig:
    """Parse the ``[bot]`` TOML table into a :class:`BotConfig`.

    Missing file, missing table, and missing keys all produce the default
    (``None``-filled) config — the file is an override surface, not a
    requirement. Bad types or invalid values raise eagerly so a typo cannot
    silently pass a wrong cadence or a stale tail.
    """
    raw = _table("bot")
    interval: object | None = raw.get("interval")
    max_age: object | None = raw.get("max_age")
    retries: object | None = raw.get("retries")
    no_sync: object | None = raw.get("no_sync")

    if interval is not None and (not isinstance(interval, int) or interval <= 0):
        raise ValueError(f"bot.interval must be a positive integer, got {interval!r}")
    if max_age is not None and (not isinstance(max_age, int) or max_age <= 0):
        raise ValueError(f"bot.max_age must be a positive integer, got {max_age!r}")
    if retries is not None and (not isinstance(retries, int) or retries < 0):
        raise ValueError(f"bot.retries must be a non-negative integer, got {retries!r}")
    if no_sync is not None and not isinstance(no_sync, bool):
        raise ValueError(f"bot.no_sync must be true/false, got {no_sync!r}")

    return BotConfig(
        interval=interval,
        max_age=max_age,
        retries=retries,
        no_sync=no_sync,
    )


@dataclass(frozen=True)
class LiveConfig:
    """``[live]`` section values, all optional. ``None`` = absent from file.

    Entrypoint code merges this with CLI flags + code defaults
    (:func:`src.live.options.merge_live_options`). The adapter VALUE is not
    validated here — this module is a leaf with no access to the adapter
    vocabulary, and a wrong name must be refused by the layer that knows it.

    ``allow_unsafe`` is deliberately ABSENT: it disables the exit code a
    scheduler alerts on, so it stays a per-invocation flag and never a
    deployment-wide default (see §Live trading).
    """

    adapter: str | None = None
    allow_oth: bool | None = None
    #: Staleness gate in days. ``[live]`` owns this, not ``[bot]``: a one-shot
    #: cron cycle and a resident bot age differently.
    max_age: int | None = None
    #: Reconcile + report, place nothing. Safe direction only, so a file may set
    #: it; a flag still wins either way.
    dry_run: bool | None = None
    #: ``text`` | ``json`` report format.
    format: str | None = None


def read_live_config() -> LiveConfig:
    """Parse the ``[live]`` TOML table into a :class:`LiveConfig`.

    Missing file, missing table, and missing keys all produce the default
    (``None``-filled) config — the file is an override surface, not a
    requirement. A bad TYPE raises eagerly; whether a string names a real
    adapter (or a real format) is the merge layer's call, because this module
    is a leaf that cannot see either vocabulary.
    """
    raw = _table("live")
    adapter: object | None = raw.get("adapter")
    allow_oth: object | None = raw.get("allow_oth")
    max_age: object | None = raw.get("max_age")
    dry_run: object | None = raw.get("dry_run")
    fmt: object | None = raw.get("format")

    if adapter is not None and not isinstance(adapter, str):
        raise ValueError(f"live.adapter must be a string, got {adapter!r}")
    if allow_oth is not None and not isinstance(allow_oth, bool):
        raise ValueError(f"live.allow_oth must be true/false, got {allow_oth!r}")
    if max_age is not None and (not isinstance(max_age, int) or max_age <= 0):
        raise ValueError(f"live.max_age must be a positive integer, got {max_age!r}")
    if dry_run is not None and not isinstance(dry_run, bool):
        raise ValueError(f"live.dry_run must be true/false, got {dry_run!r}")
    if fmt is not None and not isinstance(fmt, str):
        raise ValueError(f"live.format must be a string, got {fmt!r}")

    return LiveConfig(
        adapter=adapter,
        allow_oth=allow_oth,
        max_age=max_age,
        dry_run=dry_run,
        format=fmt,
    )
