"""The candle file's data migrations: one series key per listing.

Exercised on a temp sqlite file seeded with the real file's shape (two ``EA``
listings, one stored under a key its venue does not derive), and the assertions
are the contract:

* the registry applies once and a second run applies nothing;
* the unique index exists, the orphan index is gone, and a duplicate ticker is
  refused by SQLite rather than by a reader's diligence;
* the preflight refuses a mis-keyed row / an unmapped venue with the offending
  row named, so the index can never be built over ambiguous data — re-keying one
  listing is operator work, not a migration;
* ``down`` of ``data_0003`` drops exactly its own index and no row;
* the migration carries its OWN frozen venue vocabulary, so history cannot drift
  with :mod:`src.data.venue`.
"""

from __future__ import annotations

from collections.abc import Iterator

import peewee
import pytest

from src.db.migrations.runner import run_down, run_pending
from src.db.migrations.types import Migration, MigrationRefused, Registry
from src.db.migrations.versions import data_0003_symbol_ticker_unique as m0003
from src.db.migrations.versions import DATA_MIGRATIONS
from src.db.models import CANDLE_MODELS

pytestmark = pytest.mark.db

_MEXI_CONID = 295338948
_US_CONID = 268995

#: The index migration alone, so a refusal/down can be exercised on its own.
_INDEX_ONLY: Registry = (Migration(name=m0003.NAME, up=m0003.up, down=m0003.down),)


@pytest.fixture()
def db(tmp_path) -> Iterator[peewee.SqliteDatabase]:
    """An isolated sqlite file with the real ``symbol``/``candle`` tables."""
    database = peewee.SqliteDatabase(str(tmp_path / "data.sqlite"))
    yield database


def _seed(db: peewee.SqliteDatabase, rows: tuple[tuple[int, str, str], ...]) -> None:
    """Create the tables and insert ``(conid, ticker, market)`` symbol rows."""
    with db.bind_ctx(CANDLE_MODELS):
        db.create_tables(CANDLE_MODELS)
    for conid, ticker, market in rows:
        db.execute_sql(
            "INSERT INTO symbol (conid, ticker, market, currency, name) "
            "VALUES (?, ?, ?, 'USD', NULL)",
            (conid, ticker, market),
        )


def _rows(db: peewee.SqliteDatabase, sql: str) -> list[tuple[object, ...]]:
    return [tuple(row) for row in db.execute_sql(sql).fetchall()]


def _indexes(db: peewee.SqliteDatabase) -> set[str]:
    return {str(row[0]) for row in db.execute_sql("SELECT name FROM sqlite_master")}


def test_a_fresh_file_applies_the_registry_in_tuple_order_once(db):
    names = run_pending(db, DATA_MIGRATIONS)

    assert names == (
        "data_0001_baseline",
        "data_0002_drop_migrated_live_tables",
        m0003.NAME,
    )
    assert run_pending(db, DATA_MIGRATIONS) == (), "a second run applies once"
    assert "symbol_ticker_unique" in _indexes(db)
    assert "symbol_symbol_idx" not in _indexes(db)


def test_the_index_refuses_a_duplicate_ticker_and_down_gives_it_back(db):
    """``down`` is the lossless inverse: the constraint goes, no row does."""
    _seed(db, ((_US_CONID, "EA", "SMART"),))
    assert run_pending(db, _INDEX_ONLY) == (m0003.NAME,)

    with pytest.raises(peewee.IntegrityError):
        db.execute_sql(
            "INSERT INTO symbol (conid, ticker, market, currency) "
            "VALUES (2, 'EA', 'MEXI', 'MXN')"
        )

    assert run_down(db, _INDEX_ONLY) == (m0003.NAME,)

    db.execute_sql(
        "INSERT INTO symbol (conid, ticker, market, currency) "
        "VALUES (2, 'EA', 'MEXI', 'MXN')"
    )
    assert _rows(db, "SELECT COUNT(*) FROM symbol") == [(2,)]
    assert "symbol_ticker_unique" not in _indexes(db)


def test_a_split_series_needs_no_migration(db):
    """A re-keyed listing satisfies the index: the one-off fix is data, not history."""
    _seed(db, ((_US_CONID, "EA", "SMART"), (_MEXI_CONID, "EA.MX", "MEXI")))

    assert run_pending(db, DATA_MIGRATIONS) == (
        "data_0001_baseline",
        "data_0002_drop_migrated_live_tables",
        m0003.NAME,
    )


def test_the_preflight_refuses_the_mis_keyed_row_and_names_it(db):
    """The index cannot be built over a listing stored under the wrong key."""
    _seed(db, ((_US_CONID, "EA", "SMART"), (_MEXI_CONID, "EA", "MEXI")))

    with pytest.raises(MigrationRefused, match=str(_MEXI_CONID)):
        run_pending(db, _INDEX_ONLY)

    assert "symbol_ticker_unique" not in _indexes(db), "a refusal writes nothing"


def test_the_preflight_refuses_an_unmapped_venue(db):
    """A venue with no key rule is data the operator must fix, not a suffix."""
    _seed(db, ((42, "SHOP", "TSX"),))

    with pytest.raises(MigrationRefused, match="unmapped venue TSX"):
        run_pending(db, _INDEX_ONLY)


def test_the_migration_carries_its_own_frozen_vocabulary():
    """A historical migration must not change because src/data/venue.py grew."""
    assert m0003.series_key("EA", "MEXI") == "EA.MX"
    assert m0003.series_key("SHOP", "TSX") is None
    assert m0003.series_key("AAPL", "SMART") == "AAPL"
