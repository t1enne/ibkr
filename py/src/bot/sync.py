"""Bot-side candle sync — derive the download and bound it (plan §5.4).

The bot must not trade a stale tail, so before every cycle it refreshes the
config's candles through the SAME ``data dl`` primitive an operator uses. This
module owns the DERIVATION (which symbols/bar/start date the config implies)
and the TIMEOUT (``download`` has none, and an unbounded one would wedge a
cycle forever). The seam is :class:`SyncFn`.
"""

from __future__ import annotations

import asyncio
import math
from datetime import date, timedelta
from typing import Protocol

from src.bt.warmup import parse_warmup_days
from src.data.dl import download
from src.live.types import LiveConfig

#: How long one sync may take before the bot gives up and retries. Deliberately
#: generous: a full-history backfill of a large universe is slow but finite.
SYNC_TIMEOUT_S = 900.0


class SyncFn(Protocol):
    """Runs the candle sync for a config; ``loop.py`` injects this seam."""

    def __call__(self, cfg: LiveConfig, *, today: date) -> None:
        """Fetch the config's candles; raise (or time out) to fail the cycle."""
        ...


def derive_from_date(cfg: LiveConfig, today: date) -> date:
    """The download's start date: *today* less the config's warmup span.

    The warmup string is a CALENDAR span (``"300d"``/``"1y"``), so the head of
    the window is ``today - span``: a cycle needs the warmup bars LEADING UP TO
    now, not to some long-past backtest start.
    """
    days = parse_warmup_days(cfg.warmup)
    return today - timedelta(days=math.ceil(days))


async def _download(
    symbols: list[str], from_date: date, bar: str, timeout_s: float
) -> None:
    """Fetch candles under a bot-owned deadline (``download`` has no timeout)."""
    await asyncio.wait_for(
        download(symbols, from_date=from_date, bar=bar), timeout=timeout_s
    )


def sync_cfg(
    cfg: LiveConfig, *, today: date, timeout_s: float = SYNC_TIMEOUT_S
) -> None:
    """Sync the config's candles, bounded by *timeout_s*.

    ``bars[0]`` is the signal interval and what the cycle screens on, so it is
    also the bar to download. A raise or a timeout propagates: the loop maps it
    to exit 1 and never runs the cycle on the stale tail.
    """
    bar = cfg.bars[0]
    from_date = derive_from_date(cfg, today)
    asyncio.run(_download(list(cfg.symbols), from_date, bar, timeout_s))


__all__ = [
    "SYNC_TIMEOUT_S",
    "SyncFn",
    "derive_from_date",
    "sync_cfg",
]
