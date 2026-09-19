"""CandleStore — lazy DataFrame view over incremental numpy column arrays.

Single source of truth for OHLCV data during backtest. Wraps the mutable
``CandleRows`` accumulator by reference. Two access paths:

* ``.latest(sym, iv)`` / ``.count(sym, iv)`` — O(1) numpy reads, zero allocation
* ``store[(sym, iv)]`` / ``store.get(...)`` / iteration — lazy DataFrame build
  (``Mapping[(str, str), DataFrame]`` for drop-in strategy compatibility)

See ``src/bt/engine/backtest.py`` for the ``CandleRows`` layout.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pandas import DataFrame, Timestamp

# ---------------------------------------------------------------------------
# shared type
# ---------------------------------------------------------------------------

# Per-candle row accumulator — column-major numpy arrays keyed by (symbol, interval).
# Each value: {"timestamp": ndarray, "open": ndarray, …, "_len": array([N])}.
CandleRows = dict[tuple[str, str], dict[str, np.ndarray]]


# ---------------------------------------------------------------------------
# CandleStore
# ---------------------------------------------------------------------------


class CandleStore(Mapping[tuple[str, str], "DataFrame"]):
    """Lazy DataFrame view over incremental numpy column arrays.

    Wraps a ``CandleRows`` dict by reference — rows are appended in-place
    by the backtest loop.  Strategies read through this store.

    *Cursor* — a timestamp ceiling set via ``.advance(ts)`` by the engine
    before each strategy invocation.  DataFrames built via the Mapping
    interface are truncated to the cursor, so a strategy never sees data
    from future timestamps.  ``.latest()`` and ``.count()`` are not
    cursor-truncated (they return the absolute latest value known to the
    accumulator).

    ``Mapping`` interface: ``__getitem__``, ``get``, ``__contains__``,
    ``__len__``, ``__iter__``, ``keys``, ``items``, ``values``.
    """

    __slots__ = ("_rows", "_cursor", "_ta", "_strategy_state", "_fundamentals")

    def __init__(
        self,
        rows: CandleRows,
        cursor: Timestamp | None = None,
    ) -> None:
        self._rows: CandleRows = rows
        self._cursor: Timestamp | None = cursor
        self._ta: Any = None  # optional prefetched TaContext (DSL)
        self._strategy_state: dict | None = None  # optional per-run DSL holder
        self._fundamentals: Any = None  # optional Fundamentals store (DSL)

    # -- DSL support --------------------------------------------------------

    def attach_fundamentals(self, fundamentals: Any) -> None:
        """Bind a prefetched ``Fundamentals`` store to this store (DSL).

        Type is deliberately loose ``Any`` for the same reason as ``attach_ta``:
        the engine layer must not import the concrete strategy-layer context, so
        the DSL narrows it via isinstance at call time. Binding also hands the
        store's cursor to the fundamentals series (``bind_cursor``) so every
        as-first-stated read honors the same publication ceiling as the bars —
        one cursor, no second source of truth.
        """
        self._fundamentals = fundamentals
        if fundamentals is not None:
            fundamentals.bind_cursor(self.cursor_timestamp)

    @property
    def fundamentals(self) -> Any:
        """The prefetched Fundamentals store for DSL strategies, or None."""
        return self._fundamentals

    def cursor_timestamp(self) -> Timestamp | None:
        """The engine's current cursor (None before the first ``advance``).

        Read through by the fundamentals series so their visibility tracks the
        bar cursor instead of a snapshot taken at bind time.
        """
        return self._cursor

    def attach_ta(self, ta: Any) -> None:
        """Bind a prefetched TaContext to this store (set once by the engine).

        Strategies that opt into the DSL read indicators through ``store.ta``;
        the TaContext shares this store's cursor so it can never lookahead.
        Ta type is deliberately loose ``Any``: the store must not import the
        concrete DSL context (keeps the engine decoupled from the strategy
        layer); the decorated strategy narrows it via isinstance at call time.
        """
        self._ta = ta

    @property
    def ta(self) -> Any:
        """The prefetched TaContext for DSL strategies, or None if not enabled."""
        return self._ta

    def attach_strategy_state(self, holder: dict) -> None:
        """Bind a per-run cross-candle state holder (stateful DSL strategies).

        The engine mints a **fresh** holder per ``run_once``/window and stores
        it here, so concurrent runs never share strategy state — the adapter
        reads it from the ``state`` it is handed, never from a module singleton.
        """
        self._strategy_state = holder

    @property
    def strategy_state(self) -> dict | None:
        """The per-run cross-candle state holder for stateful DSL strategies.

        ``None`` when the strategy is stateless (or the engine didn't mint one).
        """
        return self._strategy_state

    @property
    def is_exhausted(self) -> bool:
        """True when the cursor sits at the final accumulated bar of every key.

        Read-only; used by post-run consumers (the plot DSL) to assert the run
        finished rather than being truncated mid-stream, which would silently
        under-count indicator series. False before the first ``advance`` (no
        cursor) and on an empty store.
        """
        if self._cursor is None or not self._rows:
            return False
        return all(
            self.cursor_count(sym, iv) == int(cols["_len"][0])
            for (sym, iv), cols in self._rows.items()
        )

    def cursor_count(self, sym: str, interval: str) -> int:
        """Number of accumulated bars for *sym*/*interval* up to the cursor.

        Like ``_build_df`` truncation, but O(log n) numpy ``searchsorted`` and
        zero DataFrame allocation. Falls back to the absolute count when no
        cursor is set.
        """
        cols = self._rows.get((sym, interval))
        if cols is None:
            return 0
        n = int(cols["_len"][0])
        if n == 0:
            return 0
        if self._cursor is None:
            return n
        cursor_ns = np.datetime64(self._cursor.to_datetime64())
        ts_arr = cols["timestamp"][:n]
        return int(np.searchsorted(ts_arr, cursor_ns, side="right"))

    # -- mutation (called by engine, not strategies) --------------------

    def advance(self, ts: Timestamp) -> None:
        self._cursor = ts

    # -- fast path: O(1) from numpy, no DataFrame build -----------------

    def latest(self, sym: str, interval: str) -> float | None:
        """Return the most recent close for *sym* at *interval*, or None."""
        cols = self._rows.get((sym, interval))
        if cols is None:
            return None
        n = int(cols["_len"][0])
        if n == 0:
            return None
        return float(cols["close"][n - 1])

    def count(self, sym: str, interval: str) -> int:
        """Return the number of accumulated bars for *sym* at *interval*."""
        cols = self._rows.get((sym, interval))
        if cols is None:
            return 0
        return int(cols["_len"][0])

    def prior_ohlcv(self, sym: str, interval: str) -> dict[str, float] | None:
        """OHLCV of the bar immediately *before* the cursor — O(1), no allocation.

        Returns ``{open, high, low, close, volume}`` for the bar one behind the
        current cursor bar, or ``None`` when fewer than two bars are visible.
        This is the fast-path replacement for
        ``store.get((sym, interval)).iloc[:-1].iloc[-1]`` (which builds a full
        DataFrame per call) — the pre-cursor read the online volume profile
        needs, so it no longer allocates a pandas frame per symbol per bar in
        the backtest hot path. Indexing matches ``_build_df`` truncation: with
        ``n = cursor_count`` visible rows, row ``n-1`` is the current cursor bar
        and row ``n-2`` is the prior one.
        """
        cols = self._rows.get((sym, interval))
        if cols is None:
            return None
        n = self.cursor_count(sym, interval)
        if n < 2:
            return None
        i = n - 2
        return {
            f: float(cols[f][i]) for f in ("open", "high", "low", "close", "volume")
        }

    # -- Mapping interface (full DataFrame, built on demand) ------------

    def _build_df(self, key: tuple[str, str]) -> DataFrame:
        cols = self._rows[key]
        n = int(cols["_len"][0])
        ts_arr = cols["timestamp"][:n]

        # Truncate to cursor
        if self._cursor is not None:
            cursor_ns = np.datetime64(self._cursor.to_datetime64())
            idx = np.searchsorted(ts_arr, cursor_ns, side="right")
            n = int(idx)
            ts_arr = ts_arr[:n]
            if n == 0:
                return pd.DataFrame(
                    {
                        "open": [],
                        "high": [],
                        "low": [],
                        "close": [],
                        "volume": [],
                    },
                    index=pd.DatetimeIndex([]),
                )

        return pd.DataFrame(
            {
                "open": cols["open"][:n],
                "high": cols["high"][:n],
                "low": cols["low"][:n],
                "close": cols["close"][:n],
                "volume": cols["volume"][:n],
            },
            index=pd.DatetimeIndex(ts_arr),
        )

    def __getitem__(self, key: tuple[str, str]) -> DataFrame:
        if key not in self._rows:
            raise KeyError(key)
        return self._build_df(key)

    def get(self, key: object, default: object = None) -> DataFrame | None:  # ty: ignore[invalid-method-override]
        if not isinstance(key, tuple) or key not in self._rows:
            return cast("DataFrame | None", default)
        return self._build_df(cast("tuple[str, str]", key))

    def __contains__(self, key: object) -> bool:
        return key in self._rows

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def keys(self):
        return self._rows.keys()

    def items(self):
        for k in self._rows:
            yield k, self._build_df(k)

    def values(self):
        for k in self._rows:
            yield self._build_df(k)
