"""Type definitions for the sync module.

All domain types are immutable — following FP principles:
data over control flow, make invalid states unrepresentable.

Types defined here:
  CandleDict     — TypedDict for OHLCV candle data (structural product type)
  UniverseConf   — frozen config loaded from universe .json
  SyncResult     — frozen result of a sync operation
  FetchPlan      — frozen per-symbol gap plan for dry-run
  PreviewResult  — frozen result of a dry-run preview
  ProgressFn     — Protocol for progress callbacks (I/O boundary)
  ISymbol        — dataclass mirroring SymbolSchema peewee model
  ICandle        — dataclass mirroring CandleSchema peewee model

The peewee models and the process-global ``db`` handle now live in :mod:`src.db`
(``symbol``/``candle`` have exactly one concrete target, so they no longer belong
to the ``src.data`` package's own initializer). Both are RE-EXPORTED here so that
existing ``from src.data.types import SymbolSchema, db`` keeps working unchanged.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Protocol, TypedDict

from src.db.connection import db
from src.db.models.candles import CandleSchema, SymbolSchema

__all__ = [
    "CandleDict",
    "CandleSchema",
    "FetchPlan",
    "ICandle",
    "ISymbol",
    "PreviewResult",
    "ProgressFn",
    "SymbolSchema",
    "SyncResult",
    "UniverseConf",
    "db",
]


# ── Dataclass mirrors of ORM models ────────────────────────────


@dataclass
class ISymbol:
    """Dataclass matching SymbolSchema Peewee model."""

    conid: int
    ticker: str
    market: str
    currency: str
    name: Optional[str] = None


@dataclass
class ICandle:
    """Dataclass matching CandleSchema Peewee model."""

    conid: int
    ticker: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


# ── Candle data ──────────────────────────────────────────────────


class CandleDict(TypedDict, total=True):
    """A single OHLCV candle from IBKR, ready for DB insertion.

    total=True means every key is required — invalid states
    (missing fields) are structurally impossible.
    """

    conid: int
    ticker: str
    timestamp: int  # milliseconds since epoch
    open: float
    high: float
    low: float
    close: float
    volume: float


# ── Configuration ────────────────────────────────────────────────


@dataclass(frozen=True)
class UniverseConf:
    """Configuration loaded from universe .json.

    Frozen — configuration is immutable after loading.
    """

    symbols: list[str]
    from_date: Optional[date] = None
    to_date: Optional[date] = None
    bar: str = "1h"


# ── Results ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class SyncResult:
    """Result of a sync_data operation.

    Immutable — a snapshot of what happened during the sync.
    """

    resolved: int  # How many tickers resolved successfully
    fetched: list[int]  # Conids that had new data fetched
    gaps_found: int  # Total gap segments filled

    @property
    def total_fetched(self) -> int:
        """Number of symbols that received new data."""
        return len(self.fetched)


@dataclass(frozen=True)
class FetchPlan:
    """A plan for what gaps need to be fetched for a single symbol."""

    ticker: str
    conid: int
    gaps: list[tuple[datetime, datetime]]


@dataclass(frozen=True)
class PreviewResult:
    """Result of a dry-run preview — shows what would be fetched.

    Immutable — pure description, no side effects performed.
    """

    resolved: int  # How many tickers resolved
    total_gaps: int  # Total number of gap segments across all symbols
    plans: list[FetchPlan]  # Per-symbol gap plans


# ── Callbacks (I/O boundary) ─────────────────────────────────────


class ProgressFn(Protocol):
    """Callback for sync progress updates.

    Executed at the I/O boundary — the core logic remains pure.
    """

    def __call__(self, status: str, current: int, total: int) -> None: ...
