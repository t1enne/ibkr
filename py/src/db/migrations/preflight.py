"""Preflight: is the ``symbol`` table clean enough to carry the series-key rules?

Two invariants are enforced here, and both can fail on real data rather than in
code, so they are checked BEFORE a write and reported as the offending rows:

* every stored ticker must BE its own series key for the venue it records — a
  foreign listing stored under a US-looking name is the mix-up that puts the
  wrong bars, calendar and currency under one key;
* no two listings may derive the same key — that is the collision the unique
  index refuses, and an operator needs the conids, not an ``IntegrityError``.

The venue vocabulary is INJECTED, never imported. ``data_0003`` passes its own
frozen copy so a historical migration cannot change behaviour when
:mod:`src.data.venue` learns a new venue, while the operator-facing report passes
the live one. One algorithm, two vocabularies, no drift.

Everything here is read-only apart from the caller's own decision to refuse.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from peewee import SqliteDatabase

from src.db.introspect import table_exists
from src.db.migrations.types import MigrationRefused


class SeriesKeyFn(Protocol):
    """``(ticker, venue) -> series key``, or ``None`` when the venue is unmapped."""

    def __call__(self, ticker: str, venue: str) -> str | None: ...


@dataclass(frozen=True)
class Listing:
    """One ``symbol`` row, as far as the key rules care."""

    conid: int
    ticker: str
    market: str


@dataclass(frozen=True)
class KeyCollision:
    """Two or more listings deriving one series key."""

    key: str
    conids: tuple[int, ...]


@dataclass(frozen=True)
class UnmappedListing:
    """A listing whose venue has no series-key rule."""

    conid: int
    ticker: str
    market: str


@dataclass(frozen=True)
class MisKeyedListing:
    """A listing whose stored ticker is not the key its venue derives."""

    conid: int
    ticker: str
    market: str
    derived: str


@dataclass(frozen=True)
class PreflightReport:
    """The three ways a symbol table can fail the key rules (empty = clean)."""

    collisions: tuple[KeyCollision, ...]
    unmapped: tuple[UnmappedListing, ...]
    miskeyed: tuple[MisKeyedListing, ...]

    @property
    def ok(self) -> bool:
        """True when no check found anything."""
        return not (self.collisions or self.unmapped or self.miskeyed)

    def lines(self) -> tuple[str, ...]:
        """One actionable line per finding, naming the exact rows."""
        return (
            tuple(
                f"unmapped venue {item.market}: conid {item.conid} {item.ticker}"
                for item in self.unmapped
            )
            + tuple(
                f"mis-keyed: conid {item.conid} {item.ticker} on {item.market} "
                f"derives {item.derived}"
                for item in self.miskeyed
            )
            + tuple(
                f"key collision {item.key}: conids "
                f"{', '.join(str(c) for c in item.conids)}"
                for item in self.collisions
            )
        )


def inspect(listings: Sequence[Listing], key_for: SeriesKeyFn) -> PreflightReport:
    """Check *listings* against *key_for* (pure: rows in, findings out)."""
    unmapped: list[UnmappedListing] = []
    miskeyed: list[MisKeyedListing] = []
    groups: dict[str, list[int]] = {}
    for item in listings:
        derived = key_for(item.ticker, item.market)
        if derived is None:
            unmapped.append(UnmappedListing(item.conid, item.ticker, item.market))
            groups.setdefault(item.ticker.strip().upper(), []).append(item.conid)
            continue
        if derived != item.ticker:
            miskeyed.append(
                MisKeyedListing(item.conid, item.ticker, item.market, derived)
            )
        groups.setdefault(derived, []).append(item.conid)

    collisions = tuple(
        KeyCollision(key=key, conids=tuple(sorted(conids)))
        for key, conids in sorted(groups.items())
        if len(conids) > 1
    )
    return PreflightReport(
        collisions=collisions, unmapped=tuple(unmapped), miskeyed=tuple(miskeyed)
    )


def read_listings(db: SqliteDatabase) -> tuple[Listing, ...]:
    """Every ``symbol`` row, or none when the table does not exist yet."""
    if not table_exists(db, "symbol"):
        return ()
    rows = db.execute_sql("SELECT conid, ticker, market FROM symbol").fetchall()
    return tuple(Listing(int(r[0]), str(r[1]), str(r[2])) for r in rows)


def report(db: SqliteDatabase, key_for: SeriesKeyFn) -> PreflightReport:
    """The preflight findings for the file behind *db* (writes nothing)."""
    return inspect(read_listings(db), key_for)


def require_clean(db: SqliteDatabase, key_for: SeriesKeyFn) -> None:
    """Refuse with the offending rows unless the symbol table passes.

    Raises:
        MigrationRefused: at least one listing is unmapped, mis-keyed or
            collides with another. The message carries every finding, so the
            operator can fix the data without re-running under a debugger.
    """
    findings = report(db, key_for)
    if not findings.ok:
        raise MigrationRefused(
            "symbol table is not series-key clean:\n  " + "\n  ".join(findings.lines())
        )


__all__ = [
    "KeyCollision",
    "Listing",
    "MisKeyedListing",
    "PreflightReport",
    "SeriesKeyFn",
    "UnmappedListing",
    "inspect",
    "read_listings",
    "report",
    "require_clean",
]
