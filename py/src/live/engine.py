"""Live cycle orchestration — one batch pass: fetch pf → screen → reconcile → place → record.

Not a loop. The caller (CLI / cron) drives cadence. All I/O sits at the edges
(``PortfolioSource``, ``Broker``, the screen bridge); ``reconcile`` and
``build_report`` are pure. A cycle fails loudly on a stale feed or an
unfetchable portfolio rather than trading yesterday's intent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import pandas as pd

from src.bt.state import ActionType, PortfolioState
from src.data.db import get_connection
from src.live.broker import Broker, OrderResult
from src.live.ledger import Ledger, PositionRecord
from src.live.portfolio_source import PortfolioSource
from src.live.reconcile import reconcile
from src.live.result import Err
from src.live.signals import live_signals
from src.live.types import FeedError, LiveConfig, LiveSignal, OrderIntent


@dataclass(frozen=True)
class CycleReport:
    """The full outcome of one cycle: what we saw, decided, placed, book start."""

    as_of: pd.Timestamp
    signals: tuple[LiveSignal, ...]
    intents: tuple[OrderIntent, ...]
    results: tuple[OrderResult, ...]
    portfolio_before: PortfolioState


class StaleDataError(RuntimeError):
    """The universe's newest bar is too old — refuse to trade on a stale tail."""

    def __init__(self, error: FeedError) -> None:
        super().__init__(error.message)
        self.error = error


class PortfolioFetchError(RuntimeError):
    """The portfolio fetch failed; the cycle cannot reconcile without a book."""

    def __init__(self, error: FeedError) -> None:
        super().__init__(error.message)
        self.error = error


class SignalSource(Protocol):
    """The screen bridge: config path -> actionable signals (injectable)."""

    def __call__(
        self, config_path: str, max_age_days: int | None = None
    ) -> tuple[LiveSignal, ...]: ...


def build_report(
    portfolio: PortfolioState,
    signals: tuple[LiveSignal, ...],
    intents: tuple[OrderIntent, ...],
    results: tuple[OrderResult, ...],
    as_of: pd.Timestamp,
) -> CycleReport:
    """Pure: assemble the cycle report. No clock, no I/O."""
    return CycleReport(
        as_of=as_of,
        signals=signals,
        intents=intents,
        results=results,
        portfolio_before=portfolio,
    )


def assert_data_fresh(
    symbols: tuple[str, ...],
    max_age_days: int,
    now: pd.Timestamp,
    db_path: str | Path | None = None,
) -> None:
    """Raise ``StaleDataError`` when the universe's newest bar is too old.

    A cron ``data dl`` failure must not let the cycle trade on a stale tail.
    ``max_age_days <= 0`` disables the gate; an empty universe is skipped.
    """
    if max_age_days <= 0 or not symbols:
        return
    newest_ms = _newest_ms(symbols, db_path)
    if newest_ms is None:
        raise StaleDataError(
            FeedError(kind="stale_data", message="no data for universe")
        )
    newest = pd.Timestamp(int(newest_ms), unit="ms")
    age = now - newest
    if age > pd.Timedelta(days=max_age_days):
        raise StaleDataError(
            FeedError(
                kind="stale_data",
                message=(
                    f"universe newest bar {newest} is {age} old (> {max_age_days}d)"
                ),
            )
        )


def _newest_ms(symbols: tuple[str, ...], db_path: str | Path | None) -> int | None:
    """Newest candle timestamp (ms epoch) across *symbols*, or ``None`` if none."""
    placeholders = ",".join("?" * len(symbols))
    con = get_connection(db_path)
    try:
        row = con.execute(
            f"SELECT MAX(timestamp) FROM candle WHERE ticker IN ({placeholders})",
            tuple(symbols),
        ).fetchone()
    finally:
        con.close()
    return row[0] if row else None


async def run_cycle(
    config: LiveConfig,
    *,
    source: PortfolioSource,
    broker: Broker,
    ledger: Ledger,
    strategy_id: str,
    config_path: str | None = None,
    max_age_days: int = 5,
    dry_run: bool = False,
    db_path: str | Path | None = None,
    now: pd.Timestamp | None = None,
    signal_source: SignalSource = live_signals,
) -> CycleReport:
    """One full batch pass. Not a loop; the caller drives cadence.

    ``dry_run=True`` computes signals + intents but places NOTHING: the broker
    loop is skipped, ``results`` is empty, and no ``record_open`` / ``mark_closed``
    write happens (only ``touch_cycle``). Nothing is recorded, so the next cycle
    recomputes the same intents.
    """
    now_ts = now if now is not None else pd.Timestamp.now()
    fetched = await source.fetch()
    if isinstance(fetched, Err):
        raise PortfolioFetchError(cast("FeedError", fetched.error))
    snapshot = fetched.value
    assert_data_fresh(config.symbols, max_age_days, now_ts, db_path)
    assert config_path is not None, (
        "run_cycle requires config_path for the screen bridge"
    )
    signals = signal_source(config_path, max_age_days)
    # Align the simulated book with the fetched read so the SAME book is both
    # reconciled and settled (doc's Broker Protocol has no seed — adaptation).
    broker.seed(snapshot.portfolio)
    owned = _owned_ids(ledger, strategy_id)
    intents = reconcile(signals, snapshot.portfolio, config, owned)
    results: tuple[OrderResult, ...] = ()
    if not dry_run:
        results = await _place_all(broker, ledger, intents, strategy_id, now_ts)
    ledger.touch_cycle(strategy_id, now_ts)
    await broker.close()
    return build_report(snapshot.portfolio, signals, intents, results, now_ts)


def _owned_ids(ledger: Ledger, strategy_id: str) -> frozenset[str]:
    """Broker lot ids this strategy currently owns (scopes close intents)."""
    return frozenset(r.position_id for r in ledger.open_positions(strategy_id))


async def _place_all(
    broker: Broker,
    ledger: Ledger,
    intents: tuple[OrderIntent, ...],
    strategy_id: str,
    now_ts: pd.Timestamp,
) -> tuple[OrderResult, ...]:
    """Place every intent in order; a transport ``Err`` skips it and continues.

    An ``Err`` carries no ``OrderResult``, so nothing is recorded — the next
    cycle recomputes the same intent.
    """
    results: list[OrderResult] = []
    for intent in intents:
        placed = await broker.place(intent)
        if isinstance(placed, Err):
            continue
        results.append(placed.value)
        _record(ledger, strategy_id, intent, placed.value, now_ts)
    return tuple(results)


def _record(
    ledger: Ledger,
    strategy_id: str,
    intent: OrderIntent,
    result: OrderResult,
    now_ts: pd.Timestamp,
) -> None:
    """Ledger the write points. A failed/rejected result records NOTHING."""
    if not result.ok:
        return
    if intent.action is ActionType.close:
        if intent.position_id:
            ledger.mark_closed(strategy_id, intent.position_id, now_ts)
        return
    if intent.action in (ActionType.long, ActionType.short):
        # A lot the broker did not name can never be targeted by a close
        # (_close_position requires a position_id), so recording it would
        # create an unclosable phantom row. Record nothing.
        if not result.position_id:
            return
        ledger.record_open(_open_record(strategy_id, intent, result, now_ts))


def _open_record(
    strategy_id: str,
    intent: OrderIntent,
    result: OrderResult,
    now_ts: pd.Timestamp,
) -> PositionRecord:
    """Build the ledger row for a confirmed open, preferring the fill's numbers."""
    fill = result.fill
    return PositionRecord(
        strategy_id=strategy_id,
        position_id=result.position_id or "",
        symbol=intent.symbol,
        side="long" if intent.action is ActionType.long else "short",
        qty=fill.filled_qty if fill else intent.qty,
        entry_price=fill.executed_price if fill else intent.ref_price,
        entry_time=fill.timestamp if fill else now_ts,
        stop_loss=intent.stop_loss,
        take_profit=intent.take_profit,
        tag=intent.tag,
        status="open",
        opened_at=now_ts,
    )
