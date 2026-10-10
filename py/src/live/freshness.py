"""Freshness gate — refuse to trade on a stale universe tail.

A cron ``data dl`` failure must not let a cycle act on yesterday's intent, so the
cycle asserts the universe's newest bar is recent before it screens. The gate is
pure policy over the candle file: it reads ONE ``MAX(timestamp)`` (never the
callback-loop that would be fatal) and compares it to the cycle clock.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.db.connection import get_connection
from src.live.types import FeedError


class StaleDataError(RuntimeError):
    """The universe's newest bar is too old — refuse to trade on a stale tail."""

    def __init__(self, error: FeedError) -> None:
        super().__init__(error.message)
        self.error = error


def assert_data_fresh(
    symbols: tuple[str, ...],
    max_age_days: int,
    now: pd.Timestamp,
    db_path: str | Path | None = None,
) -> None:
    """Raise ``StaleDataError`` when the universe's newest bar is too old.

    ``max_age_days <= 0`` disables the gate; an empty universe is skipped. A
    universe with NO candle at all is stale by definition — it cannot be traded.
    """
    if max_age_days <= 0 or not symbols:
        return
    newest_ms = _newest_ms(symbols, db_path)
    if newest_ms is None:
        raise StaleDataError(
            FeedError(kind="stale_data", message="no data for universe")
        )
    # Compare like epochs: the DB stores ms-epoch UTC, so anchor both sides to
    # UTC. A naive local ``now`` would skew the age by the host's UTC offset.
    newest = pd.Timestamp(int(newest_ms), unit="ms", tz="UTC")
    age = _utc(now) - newest
    if age > pd.Timedelta(days=max_age_days):
        raise StaleDataError(
            FeedError(
                kind="stale_data",
                message=(
                    f"universe newest bar {newest} is {age} old (> {max_age_days}d)"
                ),
            )
        )


def _utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Normalise *ts* to UTC: a naive timestamp is taken as UTC, not local."""
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def _newest_ms(symbols: tuple[str, ...], db_path: str | Path | None) -> int | None:
    """Newest candle timestamp (ms epoch) across *symbols*, or ``None`` if none.

    ``candle.ticker`` is stored UPPERCASE, so upper-case the symbols before
    binding — a lowercase config symbol must match, not yield a spurious empty.
    """
    upper = tuple(s.upper() for s in symbols)
    placeholders = ",".join("?" * len(upper))
    con = get_connection(db_path)
    try:
        row = con.execute(
            f"SELECT MAX(timestamp) FROM candle WHERE ticker IN ({placeholders})",
            upper,
        ).fetchone()
    finally:
        con.close()
    return row[0] if row else None


__all__ = ["StaleDataError", "assert_data_fresh"]
