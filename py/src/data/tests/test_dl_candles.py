"""``data dl`` addresses SERIES, never conids.

The fetch is per conid, but its result is reported per series: a conid-keyed map
would collapse two listings of one contract into a single silent outcome, and the
series key is the only name a strategy, a report and a candle row all agree on.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from src.data import dl as dl_mod
from src.data.types import ISymbol


def _gaps() -> list[tuple[datetime, datetime]]:
    return [(datetime(2024, 1, 1), datetime(2024, 1, 2))]


@pytest.mark.asyncio
async def test_fetch_outcomes_are_keyed_by_series_key(monkeypatch):
    """One entry per series, and a failed fetch stays ``None`` under its key."""
    started: dict[str, object] = {}

    async def _batch(conids: list[int], **_kwargs):
        started["conids"] = conids
        return {1: _gaps(), 2: None}

    monkeypatch.setattr(dl_mod, "candles_batch", _batch)
    symbols = [
        ISymbol(conid=1, ticker="ENI.MI", market="BVME", currency="EUR"),
        ISymbol(conid=2, ticker="AAPL", market="SMART", currency="USD"),
    ]

    fetched = await dl_mod._get_candles(symbols, date(2024, 1, 1))

    assert fetched == {"ENI.MI": _gaps(), "AAPL": None}
    assert started["conids"] == [1, 2]
