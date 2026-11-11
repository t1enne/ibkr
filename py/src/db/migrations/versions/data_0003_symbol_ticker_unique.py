"""``data_0003`` — one series key per listing, enforced by a unique index.

``symbol.ticker`` is the series key (see :mod:`src.data.venue`), so two rows
carrying one ticker are not two names for one thing — they are one key with two
possible calendars, currencies and bar sets, and nothing downstream can tell
which listing a bar belongs to. The unique index makes that state unrepresentable
instead of leaving it to every reader to notice.

The same migration drops ``symbol_symbol_idx``, a leftover index on a column the
table no longer has: it constrains nothing and every write pays for it.

PREFLIGHT FIRST. Both this index and the key rule rest on data, not on code, so
:func:`~src.db.migrations.preflight.require_clean` runs before any DDL and
refuses with the exact offending rows — an unmapped venue, a listing stored under
a name its venue does not derive, or two listings on one key. A refusal is
deliberate: re-keying one listing is a one-off data fix on one file (an operator
SQL update, or a re-download), not a step every future file must replay, so this
migration names the row and stops rather than doing it here.

The venue vocabulary below is FROZEN INSIDE THIS MODULE on purpose. A historical
migration must not change behaviour because ``src/data/venue.py`` learned a new
venue afterwards; editing this copy would silently change what has already run.

``down`` drops the index and nothing else, so no row is ever lost — the genuinely
lossless inverse this repo's policy asks for. It does NOT recreate
``symbol_symbol_idx``: that index names a column which no longer exists, so it
could only be rebuilt as a broken object.
"""

from __future__ import annotations

import peewee

from src.db.migrations.preflight import require_clean

#: The name recorded in ``peewee_migration``.
NAME = "data_0003_symbol_ticker_unique"

#: The index this migration owns.
_INDEX = "symbol_ticker_unique"

#: The stale index whose column no longer exists.
_ORPHAN_INDEX = "symbol_symbol_idx"

#: FROZEN venue vocabulary — see the module docstring for why it is duplicated.
_US_VENUES: frozenset[str] = frozenset(
    {
        "SMART",
        "XNYS",
        "NYSE",
        "NASDAQ",
        "ARCA",
        "AMEX",
        "BATS",
        "IEX",
        "BATY",
        "ARCAEDGE",
        "EDGEA",
        "NYSEAMERICAN",
        "NASDAQNM",
    }
)
_SYNTHETIC: frozenset[str] = frozenset({"GATE", "VALUE"})
_SUFFIXES: dict[str, str] = {
    "BVME": "MI",
    "MIL": "MI",
    "SBF": "PA",
    "IBIS": "DE",
    "FWB": "DE",
    "AEB": "AS",
    "LSE": "L",
    "BM": "MC",
    "VSE": "VI",
    "EBS": "SW",
    "MEXI": "MX",
}


def series_key(ticker: str, venue: str) -> str | None:
    """This migration's own key rule: the key, or ``None`` for an unmapped venue."""
    upper = ticker.strip().upper()
    code = (venue or "").strip().upper()
    if code == "" or code in _US_VENUES or code in _SYNTHETIC:
        return upper
    suffix = _SUFFIXES.get(code)
    if suffix is None:
        return None
    return upper if upper.endswith(f".{suffix}") else f"{upper}.{suffix}"


def up(db: peewee.SqliteDatabase) -> None:
    """Refuse unless every listing is key-clean, then own the unique index."""
    require_clean(db, series_key)
    db.execute_sql(f"CREATE UNIQUE INDEX IF NOT EXISTS {_INDEX} ON symbol(ticker)")
    db.execute_sql(f"DROP INDEX IF EXISTS {_ORPHAN_INDEX}")


def down(db: peewee.SqliteDatabase) -> None:
    """Drop the unique index (see the docstring: no row is touched)."""
    db.execute_sql(f"DROP INDEX IF EXISTS {_INDEX}")


__all__ = ["NAME", "down", "series_key", "up"]
