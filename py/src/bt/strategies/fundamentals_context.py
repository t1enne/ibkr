"""Fundamentals DSL context — cursor-safe, as-first-stated fiscal series.

The read surface for strategies: ``ctx.fundamentals.income("AAPL").net_income``
returns a :class:`SeriesPIT` (a fiscal-period series, *not* a bar series), and
``[-1]`` / ``[-3:]`` slice it. ``.latest("AAPL", "income")`` is the separate
read-at-cursor view: the newest-filed (restated) statement, not part of the
curve.

Two disciplines this module exists to enforce:

* **as-first-stated** — the curve is built from
  :func:`~src.data.fundamentals.query.as_first_stated`, so a restatement never
  rewrites a prior period.
* **cursor safety** — a series is visible only up to the engine cursor. Every
  row whose ``filed`` is after the cursor is invisible, so no filing can leak
  from the strategy's future. Visibility is derived from the *bound cursor*
  at read time (as ``TaContext`` does), never cached.

Deliberately absent: ROC / TTM / ratios. The library serves the series and
``spans()``; a strategy computes its own metrics from them, because TTM on
quarter-vs-annual spans depends on the strategy's intent, not on the library's.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Generic, Mapping, TypeVar, cast

import pandas as pd

from src.data.fundamentals.query import (
    build_series,
    load_stated,
    rows_to_snapshot,
)
from src.data.fundamentals.schema import (
    BalanceSheet,
    CashFlow,
    FundamentalRow,
    Income,
    Statement,
    StatementSnapshot,
    field_names,
)

Snap = TypeVar("Snap", Income, BalanceSheet, CashFlow)

#: A cursor: ``pd.Timestamp``-normalizable, or None before the engine advances.
Cursor = pd.Timestamp | str | date | datetime | None

#: Per-statement snapshot classes, in accessor order.
_STATEMENT_CLASSES: dict[Statement, type[StatementSnapshot]] = {
    "income": Income,
    "balance": BalanceSheet,
    "cashflow": CashFlow,
}

#: Field names per statement, computed once (``__getattr__`` is a hot path).
_STATEMENT_FIELDS: dict[Statement, frozenset[str]] = {
    statement: field_names(statement) for statement in _STATEMENT_CLASSES
}


@dataclass(frozen=True)
class SeriesPIT:
    """Cursor-safe, as-first-stated fiscal series for one statement field.

    ``period`` and ``window`` share one ordering (``period_end`` ascending) and
    are built from as-first-stated rows, so index ``-1`` is the most recent
    fiscal period *whose filing the strategy has already seen*. Rows filed after
    the cursor are excluded from every read, not merely marked invisible, so a
    slice can never return a number the strategy could not have known.
    """

    period: tuple[pd.Timestamp, ...]
    window: tuple[FundamentalRow, ...]
    filed: tuple[pd.Timestamp, ...] = ()
    _cursor: Callable[[], Cursor] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # Derive ``filed`` from the window when not supplied, so a hand-built
        # series (tests, notebook) is cursor-safe without extra ceremony.
        if not self.filed:
            object.__setattr__(self, "filed", tuple(r.filed for r in self.window))

    def bind(self, cursor: Callable[[], Cursor]) -> "SeriesPIT":
        """Return a copy reading visibility from ``cursor`` (engine wiring).

        A copy, not a mutation: the cursor-free series built at load time stays
        valid as the template for other symbols/runs.
        """
        return SeriesPIT(
            period=self.period,
            window=self.window,
            filed=self.filed,
            _cursor=cursor,
        )

    @property
    def visible(self) -> int:
        """Count of periods whose filing is at/before the cursor (0..len)."""
        return self._visible_count()

    def _visible_count(self) -> int:
        if self._cursor is None:
            # Unbound series (no engine cursor reader at all). This is the
            # notebook/test path: the caller built the series directly and is
            # plainly not backtesting, so the whole curve is observable.
            return len(self.window)
        ts = self._cursor()
        if ts is None:
            # Bound to an engine cursor that has NOT been advanced yet. The run
            # has not started, so no filing is public — serving the full curve
            # here would be a lookahead. Empty is the only safe answer.
            return 0
        # ``filed`` is ascending (same order as ``period``), so the publication
        # boundary is a binary search, not a scan.
        return bisect.bisect_right(
            cast("Sequence[pd.Timestamp]", self.filed), pd.Timestamp(ts)
        )

    def __len__(self) -> int:
        return self._visible_count()

    def __getitem__(self, i: int | slice) -> float | list[float]:
        """Value at position ``i`` (negative counts back from the newest).

        A slice returns a plain list of floats in period order, like indexing a
        sequence of series values. Out-of-range reads raise ``IndexError``
        (there is no NaN padding: an absent period is absent).
        """
        n = self._visible_count()
        if isinstance(i, slice):
            start, stop, step = i.indices(n)
            return [self.window[j].value for j in range(start, stop, step)]
        idx = i if i >= 0 else n + i
        if idx < 0 or idx >= n:
            raise IndexError(f"index {i} out of range on {n}-period series")
        return self.window[idx].value

    def spans(self) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
        """``(period_start, period_end)`` per visible period, ascending.

        The fiscal-span axis a strategy needs to build its own TTM/annual
        arithmetic: the pair distinguishes a 13-week quarter from a 52-week
        year on cumulative-YTD SEC facts.
        """
        n = self._visible_count()
        return tuple((r.period_start, r.period_end) for r in self.window[:n])

    def forms(self) -> tuple[str, ...]:
        """Form type per visible period (``"10-Q"`` vs ``"10-K"``), ascending."""
        n = self._visible_count()
        return tuple(r.form for r in self.window[:n])

    def last(self) -> float | None:
        """Newest visible value, or None when nothing is visible yet."""
        n = self._visible_count()
        return None if n == 0 else self.window[n - 1].value

    def __repr__(self) -> str:
        return f"SeriesPIT(visible={self._visible_count()}, total={len(self.window)})"


class StatementSeries(Generic[Snap]):
    """Attribute-typed series holder for one statement + symbol.

    ``StatementSeries[Income].net_income`` is a :class:`SeriesPIT`; attribute
    lookup is routed through ``__getattr__`` against the snapshot dataclass's own
    fields, so the accessor surface and the dataclass can never drift, and an
    unknown name raises ``AttributeError`` (typos fail loudly, unlike a dict).
    """

    __slots__ = ("_statement", "_symbol", "_rows", "_series")

    def __init__(
        self,
        statement: Statement,
        symbol: str,
        rows: Sequence[FundamentalRow],
        series: Mapping[str, SeriesPIT],
    ) -> None:
        self._statement = statement
        self._symbol = symbol
        self._rows = rows
        self._series = series

    def __getattr__(self, name: str) -> SeriesPIT:
        if name not in _STATEMENT_FIELDS.get(self._statement, ()):
            raise AttributeError(
                f"{self._statement} statement has no field {name!r}; "
                f"available: {', '.join(sorted(_STATEMENT_FIELDS[self._statement]))}"
            )
        return self._series[name]

    def snapshot(self, period: pd.Timestamp | None = None) -> StatementSnapshot | None:
        """Reconstruct the statement snapshot for ``period`` (default: newest).

        As-first-stated, matching the series — pair with
        :meth:`Fundamentals.latest` for the restated view.
        """
        n = self._series[next(iter(self._series))].visible if self._series else 0
        if period is None:
            if n == 0:
                return None
            period = self._series[next(iter(self._series))].period[n - 1]
        return rows_to_snapshot(
            self._rows, self._symbol, period, statement=self._statement
        )

    def __repr__(self) -> str:
        return f"StatementSeries({self._statement!r}, {self._symbol!r})"


class Fundamentals:
    """Fundamentals accessor minted per run and surfaced as ``ctx.fundamentals``.

    Holds the as-first-stated rows per symbol (loaded once from the local DB at
    engine start) and the bound engine cursor. Series are memoised per
    ``(statement, symbol, field)`` on first access; the returned ``SeriesPIT``
    carries a *cursor callable*, so a memoised entry still reflects the current
    cursor on every read — caching the series cannot cache the cursor.

    Symbols are explicit at every call (``income("AAPL")``), consistent with
    ``ctx.ta.<indicator>(sym)``: fundamentals are per-issuer data, so an implicit
    "current symbol" would silently serve the wrong company in a multi-symbol
    strategy.
    """

    __slots__ = ("_rows", "_cursor", "_cache")

    def __init__(
        self,
        rows_by_symbol: Mapping[str, Sequence[FundamentalRow]],
        cursor: Callable[[], Cursor] | None = None,
    ) -> None:
        self._rows: dict[str, tuple[FundamentalRow, ...]] = {
            sym.upper(): tuple(rows) for sym, rows in rows_by_symbol.items()
        }
        self._cursor = cursor
        self._cache: dict[tuple[Statement, str, str], SeriesPIT] = {}

    @classmethod
    def build(
        cls,
        rows_by_symbol: Mapping[str, Sequence[FundamentalRow]],
        cursor: Callable[[], Cursor] | None = None,
    ) -> "Fundamentals":
        """Mint the store from raw rows, oldest-filing-first per symbol.

        **All** filings are kept (sorted by ``filed``), never pre-deduped: the
        restatements are the data. The series applies as-first-stated when it is
        assembled (:func:`build_series`), while ``latest`` reads the same rows
        newest-filed — one row set, two PIT views. Deduping here would make
        ``latest`` structurally unable to ever show a restatement.
        """
        return cls(
            {
                sym: sorted(rows, key=lambda r: (r.period_end, r.filed))
                for sym, rows in rows_by_symbol.items()
            },
            cursor=cursor,
        )

    @property
    def symbols(self) -> tuple[str, ...]:
        """Symbols with fundamentals loaded (insertion order, deduped)."""
        return tuple(self._rows)

    def bind_cursor(self, cursor: Callable[[], Cursor]) -> None:
        """Bind the engine cursor reader (set once by the engine at run start).

        The callable is read on every access, so advancing the engine cursor is
        immediately reflected in series visibility; nothing is snapshotted.
        """
        self._cursor = cursor

    def income(self, symbol: str) -> StatementSeries[Income]:
        """Income-statement series for ``symbol``."""
        return cast(StatementSeries[Income], self._statement_series("income", symbol))

    def balance(self, symbol: str) -> StatementSeries[BalanceSheet]:
        """Balance-sheet series for ``symbol``."""
        return cast(
            StatementSeries[BalanceSheet], self._statement_series("balance", symbol)
        )

    def cashflow(self, symbol: str) -> StatementSeries[CashFlow]:
        """Cash-flow series for ``symbol``."""
        return cast(
            StatementSeries[CashFlow], self._statement_series("cashflow", symbol)
        )

    def latest(self, symbol: str, statement: Statement) -> StatementSnapshot | None:
        """Newest-FILED (restated) snapshot for ``symbol`` at the cursor.

        Read-at-cursor, not part of the series: this is the statement as it
        stands *now* (every field from its latest filing), so a restated figure
        appears here while the curve keeps the as-first-stated value. Rows filed
        after the cursor are excluded, so ``latest`` leaks nothing either.
        """
        sym = symbol.upper()
        visible = [r for r in self._rows.get(sym, ()) if self._filed_by(r)]
        if not visible:
            return None
        periods = [r.period_end for r in visible if r.statement == statement]
        if not periods:
            return None
        newest = max(periods)
        return rows_to_snapshot(
            visible, sym, newest, statement=statement, as_first=False
        )

    def _filed_by(self, row: FundamentalRow) -> bool:
        """Is ``row`` published at the current cursor?"""
        if self._cursor is None:
            return True
        cursor = self._cursor()
        if cursor is None:
            return False
        return row.filed <= pd.Timestamp(cursor)

    def _statement_series(self, statement: Statement, symbol: str) -> StatementSeries:
        sym = symbol.upper()
        rows = self._rows.get(sym, ())
        series = {
            name: self._series(statement, sym, name) for name in _field_names(statement)
        }
        return StatementSeries(statement, sym, rows, series)

    def _series(self, statement: Statement, symbol: str, field: str) -> SeriesPIT:
        key = (statement, symbol, field)
        cached = self._cache.get(key)
        if cached is None:
            cached = build_series(self._rows.get(symbol, ()), statement, field)
            self._cache[key] = cached
        # Always (re)bind the live cursor: the cached entry holds the series data,
        # but which periods are visible must be read from the current engine
        # cursor on every access — never snapshotted into the cache.
        return cached if self._cursor is None else cached.bind(self._cursor)


def _field_names(statement: Statement) -> tuple[str, ...]:
    return tuple(sorted(_STATEMENT_FIELDS[statement]))


def init_fundamentals(
    rows_by_symbol: Mapping[str, Sequence[FundamentalRow]],
    cursor: Callable[[], Cursor] | None = None,
) -> Fundamentals:
    """Build the per-run ``Fundamentals`` store (kindred to ``init_ta``)."""
    return Fundamentals.build(rows_by_symbol, cursor=cursor)


def load_fundamentals(symbols: Sequence[str]) -> Fundamentals:
    """Load and assemble the as-first-stated store for ``symbols``.

    The one I/O call in the fundamentals read path: one local query per symbol,
    once per run. Symbols without stored fundamentals simply have empty series
    (a strategy reading them sees ``len(...) == 0``), so a partially covered
    universe still runs.
    """
    return Fundamentals.build({sym: load_stated(sym) for sym in symbols})


__all__ = [
    "Fundamentals",
    "StatementSeries",
    "SeriesPIT",
    "init_fundamentals",
    "load_fundamentals",
]
