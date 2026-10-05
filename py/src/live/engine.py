"""Live cycle orchestration — one batch pass: fetch pf → screen → reconcile → place → record.

Not a loop. The caller (CLI / cron) drives cadence. All I/O sits at the edges
(``PortfolioSource``, ``LiveBroker``, the screen bridge); ``reconcile`` and
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
from src.live.broker import LiveBroker, OrderResult
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


class CycleLedger(Protocol):
    """The ledger capabilities ``run_cycle`` needs: cycle stamp (audit) + sim lots.

    ``sim_open_ids`` / ``record_sim_open`` / ``mark_sim_closed`` back the sim
    path's ownership scoping (a close may only target a lot the strategy opened).
    The IBKR path never calls them: its book is ours by construction.
    """

    def touch_cycle(self, strategy_id: str, at: pd.Timestamp) -> None: ...

    def sim_open_ids(self, strategy_id: str) -> frozenset[str]: ...

    def record_sim_open(self, strategy_id: str, position_id: str) -> None: ...

    def mark_sim_closed(
        self, strategy_id: str, position_id: str, closed_at: pd.Timestamp
    ) -> None: ...


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


async def run_cycle(
    config: LiveConfig,
    *,
    source: PortfolioSource,
    broker: LiveBroker,
    ledger: CycleLedger,
    strategy_id: str,
    config_path: str | None = None,
    max_age_days: int = 5,
    dry_run: bool = False,
    db_path: str | Path | None = None,
    now: pd.Timestamp | None = None,
    signal_source: SignalSource = live_signals,
) -> CycleReport:
    """One full batch pass. Not a loop; the caller drives cadence.

    ``dry_run=True`` computes signals + intents but places NOTHING and writes
    NOTHING: the broker loop is skipped, ``results`` is empty, and no
    ``touch_cycle`` write happens. Nothing is recorded, so the next cycle
    recomputes the same intents.
    """
    now_ts = now if now is not None else pd.Timestamp.now(tz="UTC")
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
    # reconciled and settled (the LiveBroker Protocol has no seed — adaptation).
    broker.seed(snapshot.portfolio)
    # Ownership scoping (plan rev 4.1 §3): the IBKR source's book ALREADY holds
    # only this scope's lots (``trades.reconcile`` filters by our cOID prefix), so
    # every lot in it is closable and no filter is applied (``owned=None``). The
    # sim/mock book may hold exogenous fixture lots, so its closes are scoped to
    # the lots the strategy is recorded as owning (``sim_open_ids``).
    owns_book = getattr(source, "owns_book", True)
    owned = _owned_ids(ledger, strategy_id) if owns_book else None
    intents = reconcile(signals, snapshot.portfolio, config, owned)
    results: tuple[OrderResult, ...] = ()
    if not dry_run:
        results = await _place_all(broker, intents)
        if owns_book:
            _record_owned(ledger, strategy_id, results, now_ts)
        ledger.touch_cycle(strategy_id, now_ts)
    await broker.close()
    return build_report(snapshot.portfolio, signals, intents, results, now_ts)


def _owned_ids(ledger: CycleLedger, strategy_id: str) -> frozenset[str]:
    """Broker lot ids this strategy currently owns (scopes sim close intents)."""
    return ledger.sim_open_ids(strategy_id)


def _record_owned(
    ledger: CycleLedger,
    strategy_id: str,
    results: tuple[OrderResult, ...],
    now_ts: pd.Timestamp,
) -> None:
    """Record the sim ownership write points. A failed/rejected result records nothing.

    Only the sim path calls this (the IBKR book advances from its own execution
    stream, never from a placement result). A lot the broker did not name can
    never be targeted by a close, so an unnamed open records nothing.
    """
    for result in results:
        if not result.ok:
            continue
        intent = result.intent
        if intent.action is ActionType.close:
            if intent.position_id:
                ledger.mark_sim_closed(strategy_id, intent.position_id, now_ts)
        elif result.position_id:
            ledger.record_sim_open(strategy_id, result.position_id)


async def _place_all(
    broker: LiveBroker,
    intents: tuple[OrderIntent, ...],
) -> tuple[OrderResult, ...]:
    """Place the cycle's intents as ONE cohort; a cohort-level ``Err`` skips all.

    The book is NOT written from the placement result: it is advanced from the
    broker's own execution stream by the next cycle's ``reconcile`` (plan §3).
    """
    if not intents:
        return ()
    placed = await broker.place_cohort(tuple(intents))
    if isinstance(placed, Err):
        return ()
    return tuple(placed.value)
