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
    #: ``filed`` in ascending order — the bisect axis for the visible *count*.
    #: Separate from ``filed`` (which is period-ordered) because the two orders
    #: disagree on real SEC data: a cumulative annual fact for an older period
    #: is often filed *after* interim facts for newer periods. See
    #: ``_visible_positions`` for why that forces a mask rather than a prefix.
    _filed_sorted: tuple[pd.Timestamp, ...] = field(
        default=(), repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # Derive ``filed`` from the window when not supplied, so a hand-built
        # series (tests, notebook) is cursor-safe without extra ceremony.
        if not self.filed:
            object.__setattr__(self, "filed", tuple(r.filed for r in self.window))
        object.__setattr__(self, "_filed_sorted", tuple(sorted(self.filed)))

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
        return len(self._visible_positions())

    def _cursor_timestamp(self) -> pd.Timestamp | None:
        """The publication boundary, or None when nothing is observable yet.

        ``None`` means either no cursor reader at all (notebook/test path — the
        caller is plainly not backtesting, so the whole curve is fair game) or a
        bound-but-unadvanced cursor (the run has not started, so no filing is
        public). The two cases are distinguished by the caller, not here.
        """
        if self._cursor is None:
            return None
        ts = self._cursor()
        if ts is None:
            return None
        # ``parse_timestamp`` only accepts ``str | pd.Timestamp``, while a cursor
        # may be a ``date``/``datetime`` (see ``Cursor``), so normalise through
        # ``pd.Timestamp`` here. ``pd.Timestamp(...)`` is typed ``Timestamp |
        # NaTType`` (an unparseable value yields NaT), and NaT is the same
        # "nothing to observe" case as a ``None`` cursor.
        stamp = ts if isinstance(ts, pd.Timestamp) else pd.Timestamp(ts)
        # ``pd.Timestamp(...)`` is typed ``Timestamp | NaTType`` and neither
        # ``pd.isna`` nor an identity check narrows that arm away for the
        # checker, so the cast records the guaranteed-by-inspection fact: NaT was
        # returned as ``None`` just above.
        return None if stamp is pd.NaT else cast(pd.Timestamp, stamp)

    def _visible_positions(self) -> tuple[int, ...]:
        """Period-order indices of rows published at/before the cursor.

        A **mask**, not a prefix: visibility is decided per row by ``filed``,
        while the series is ordered by ``period_end``, and real SEC data has the
        two disagree — a cumulative annual fact for an older period is routinely
        filed after interim facts for newer periods (a 10-K restating FY2008 was
        filed 2010-03-18, while the FY2009 Q2 10-Q was filed 2009-08-20). So the
        visible periods are a subsequence of ``window``, and slicing a prefix
        would both drop legitimate periods and admit filing-future ones.

        The common case (``filed`` already ascending with ``period_end``) is
        short-circuited to a prefix, keeping the hot path allocation-free.
        """
        if self._cursor is None:
            return tuple(range(len(self.window)))
        ts = self._cursor_timestamp()
        if ts is None:
            return ()
        if self.filed == self._filed_sorted:
            # Monotonic: the boundary is a count, so a single bisect suffices.
            return tuple(range(bisect.bisect_right(self._filed_sorted, ts)))
        return tuple(i for i, r in enumerate(self.window) if r.filed <= ts)

    def __len__(self) -> int:
        return len(self._visible_positions())

    def __getitem__(self, i: int | slice) -> float | list[float]:
        """Value at position ``i`` (negative counts back from the newest).

        A slice returns a plain list of floats in period order, like indexing a
        sequence of series values. Out-of-range reads raise ``IndexError``
        (there is no NaN padding: an absent period is absent).
        """
        idxs = self._visible_positions()
        n = len(idxs)
        if isinstance(i, slice):
            start, stop, step = i.indices(n)
            return [self.window[idxs[j]].value for j in range(start, stop, step)]
        idx = i if i >= 0 else n + i
        if idx < 0 or idx >= n:
            raise IndexError(f"index {i} out of range on {n}-period series")
        return self.window[idxs[idx]].value

    def spans(self) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
        """``(period_start, period_end)`` per visible period, ascending.

        The fiscal-span axis a strategy needs to build its own TTM/annual
        arithmetic: the pair distinguishes a 13-week quarter from a 52-week
        year on cumulative-YTD SEC facts.
        """
        return tuple(
            (self.window[i].period_start, self.window[i].period_end)
            for i in self._visible_positions()
        )

    def forms(self) -> tuple[str, ...]:
        """Form type per visible period (``"10-Q"`` vs ``"10-K"``), ascending."""
        return tuple(self.window[i].form for i in self._visible_positions())

    def last(self) -> float | None:
        """Newest visible value, or None when nothing is visible yet."""
        idxs = self._visible_positions()
        return None if not idxs else self.window[idxs[-1]].value

    def values(self) -> list[float]:
        """Every visible value in period order, as a list of floats.

        The typed counterpart of ``series[:]``: slicing returns
        ``float | list[float]`` (a scalar and a slice share one ``__getitem__``),
        so a caller wanting the whole curve must narrow a union. Fiscal
        arithmetic wants the list, so it is spelled once here rather than
        re-derived at each call site.
        """
        return [self.window[i].value for i in self._visible_positions()]

    def __repr__(self) -> str:
        return f"SeriesPIT(visible={len(self)}, total={len(self.window)})"


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

        The default period is the newest visible ``period_end`` **across every
        field of the statement**, not the newest visible period of some one
        field: a statement's fields are independently sparse (a filer may tag
        ``net_income`` and never ``gross_profit``), so anchoring on an arbitrary
        single field would return ``None`` for a period whose rows plainly
        exist.
        """
        if not self._series:
            return None
        if period is None:
            # Newest *published* period — never ``period[-1]`` blindly, whose
            # filing may still be in the strategy's future (_visible_positions).
            newest: pd.Timestamp | None = None
            for series in self._series.values():
                idxs = series._visible_positions()
                if not idxs:
                    continue
                candidate = series.period[idxs[-1]]
                if newest is None or candidate > newest:
                    newest = candidate
            if newest is None:
                return None
            period = newest
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

    __slots__ = ("_rows", "_cursor", "_cache", "_names", "_loader")

    def __init__(
        self,
        rows_by_symbol: Mapping[str, Sequence[FundamentalRow]] | None = None,
        cursor: Callable[[], Cursor] | None = None,
        names: Sequence[str] | None = None,
        loader: Callable[[str], Sequence[FundamentalRow]] | None = None,
    ) -> None:
        self._rows: dict[str, tuple[FundamentalRow, ...]] = {
            sym.upper(): tuple(rows) for sym, rows in (rows_by_symbol or {}).items()
        }
        self._cursor = cursor
        self._cache: dict[tuple[Statement, str, str], SeriesPIT] = {}
        # Lazy mode: `names` is the configured universe, `loader` fetches one
        # symbol's rows on first read. Rows are memoised in `_rows`, so a symbol
        # is queried at most once per run -- a run that never reads fundamentals
        # never touches the DB.
        self._names: tuple[str, ...] = tuple(s.upper() for s in (names or ()))
        self._loader = loader

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
        """Configured symbols in lazy mode, else symbols already loaded."""
        return self._names if self._names else tuple(self._rows)

    def _rows_for(self, symbol: str) -> tuple[FundamentalRow, ...]:
        """Rows for ``symbol``, loading and sorting them on first read."""
        sym = symbol.upper()
        rows = self._rows.get(sym)
        if rows is None:
            if self._loader is None:
                return ()
            rows = tuple(
                sorted(self._loader(sym), key=lambda r: (r.period_end, r.filed))
            )
            self._rows[sym] = rows
        return rows

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
        visible = [r for r in self._rows_for(sym) if self._filed_by(r)]
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
        rows = self._rows_for(sym)
        series = {
            name: self._series(statement, sym, name) for name in _field_names(statement)
        }
        return StatementSeries(statement, sym, rows, series)

    def _series(self, statement: Statement, symbol: str, field: str) -> SeriesPIT:
        key = (statement, symbol, field)
        cached = self._cache.get(key)
        if cached is None:
            cached = build_series(self._rows_for(symbol), statement, field)
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
    """Lazy as-first-stated store for ``symbols`` -- nothing is read here.

    Rows for a symbol are fetched on first read of that symbol and memoised for
    the rest of the run, so the DB cost is paid only by runs that actually read
    fundamentals (and only for the symbols they read). Symbols without stored
    fundamentals simply have empty series (a strategy reading them sees
    ``len(...) == 0``), so a partially covered universe still runs.
    """
    return Fundamentals(names=symbols, loader=load_stated)


__all__ = [
    "Fundamentals",
    "StatementSeries",
    "SeriesPIT",
    "init_fundamentals",
    "load_fundamentals",
]
