"""Test the chunk loop's throttling branch in ``_fetch_candles_iterative``.

The gateway answers a burst of history calls with a 429 (pacing violation) or a
503 (server-side limiter) and no usable body. Both mean "retry the SAME window
after backing off" — treating either as fatal aborts a multi-year backfill that
would have completed a few seconds later. The retries are bounded so a gateway
that never lets up fails the symbol instead of sleeping forever.
"""

from __future__ import annotations

import importlib
from datetime import datetime
from typing import Any, cast

import pytest
from ib_rest_api_client.models.single_historical_bar_bid_ask import (
    SingleHistoricalBarBidAsk,
)

from src.data.types import CandleDict

_candles = importlib.import_module("src.data.ibkr.candles")

_FROM = datetime(2019, 11, 1)
_TO = datetime(2026, 11, 1)

_OLDEST_TS = 1569888000000  # before _FROM, so one successful chunk ends the walk


class _Bar:
    def __init__(self, ts: int) -> None:
        self.t = ts
        self.o = self.h = self.l = self.c = 1.0
        self.v = 10


class _Resp:
    def __init__(self, status_code: int, parsed: Any = None) -> None:
        self.status_code = status_code
        self.parsed = parsed


class _FakeEndpoint:
    """Plays a scripted list of responses, one per history call."""

    def __init__(self, script: list[_Resp]) -> None:
        self.script = script
        self.calls = 0

    async def asyncio_detailed(self, **_kwargs: Any) -> _Resp:
        self.calls += 1
        # The trailing response repeats once the script runs out.
        return self.script[min(self.calls - 1, len(self.script) - 1)]


@pytest.fixture
def throttle(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install a fake endpoint factory and no-op the sleeps the loop issues."""

    async def _no_sleep(_seconds: float) -> None:
        return None

    def _install(script: list[_Resp]) -> _FakeEndpoint:
        endpoint = _FakeEndpoint(script)
        monkeypatch.setattr(
            _candles, "get_iserver_marketdata_history", endpoint, raising=True
        )

        async def _asyncio_sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr(_candles.asyncio, "sleep", _asyncio_sleep)
        return endpoint

    _ = _no_sleep
    return _install


async def _run() -> list[CandleDict]:
    return await _candles._fetch_candles_iterative(
        conid=1, ticker="A2A.MI", bar="1h", from_datetime=_FROM, to_datetime=_TO
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 503])
async def test_throttled_status_retries_same_window_and_succeeds(
    throttle: Any, status: int
) -> None:
    """A throttle is transient: back off, retry, and return the candles."""
    endpoint = throttle(
        [
            _Resp(status),
            _Resp(status),
            _Resp(
                200,
                _candles.IserverHistoryBidAskResponse(
                    data=[cast(SingleHistoricalBarBidAsk, _Bar(_OLDEST_TS))]
                ),
            ),
        ]
    )

    candles = await _run()

    assert endpoint.calls == 3
    assert [c["timestamp"] for c in candles] == [_OLDEST_TS]


@pytest.mark.asyncio
async def test_persistent_throttle_fails_the_symbol(throttle: Any) -> None:
    """Bounded retries: a gateway that never lets up must not sleep forever."""
    endpoint = throttle([_Resp(429)])

    with pytest.raises(Exception, match="throttled"):
        await _run()

    assert endpoint.calls == _candles.CHUNK_THROTTLE_ATTEMPTS + 1
