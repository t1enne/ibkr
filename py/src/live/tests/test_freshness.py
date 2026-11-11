"""The freshness gate matches SERIES KEYS, exactly as the candle table stores them.

``candle.ticker`` holds series keys (``ENI.MI``, ``AAPL``), and a live config's
symbols are the same strings, so the gate needs no translation: this pins that a
suffixed key is found, that case does not matter, and that a BARE ticker does not
silently match a suffixed series (which would let a foreign universe pass the gate
on another listing's bars).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from src.data.db import get_connection
from src.live.freshness import StaleDataError, assert_data_fresh

from src.live.tests.test_engine import make_candle_db

NOW = cast("pd.Timestamp", pd.Timestamp("2024-06-03T15:00:00Z"))
FRESH = cast("pd.Timestamp", pd.Timestamp("2024-06-03T14:00:00Z"))
OLD = cast("pd.Timestamp", NOW - pd.Timedelta(days=30))


@pytest.mark.db
def test_a_suffixed_series_key_passes_the_gate(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "ENI.MI", FRESH)

    assert_data_fresh(("ENI.MI",), 5, NOW, db)


@pytest.mark.db
def test_the_key_is_case_folded_before_matching(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "ENI.MI", FRESH)

    assert_data_fresh(("eni.mi",), 5, NOW, db)


@pytest.mark.db
def test_a_bare_ticker_does_not_match_a_suffixed_series(tmp_path: Path) -> None:
    # The suffix is part of the series identity: no rows for "ENI" means no data,
    # never the Milan listing's bars under a US name.
    db = tmp_path / "c.sqlite"
    make_candle_db(db, "ENI.MI", FRESH)

    with pytest.raises(StaleDataError):
        assert_data_fresh(("ENI",), 5, NOW, db)


@pytest.mark.db
def test_an_old_suffixed_series_is_stale(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    con = get_connection(db)
    con.execute("CREATE TABLE candle (ticker TEXT, timestamp INTEGER)")
    con.execute(
        "INSERT INTO candle VALUES (?,?)",
        ("ENI.MI", int(OLD.timestamp() * 1000)),
    )
    con.commit()
    con.close()

    with pytest.raises(StaleDataError, match="30 days"):
        assert_data_fresh(("ENI.MI",), 5, NOW, db)
