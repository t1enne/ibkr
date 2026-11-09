"""Macro DSL context — cursor-safe FRED print series (``ctx.macro``).

The read surface for strategies: ``ctx.macro.vix`` is a :class:`MacroSeries`
bound to the engine cursor, and ``ctx.macro.dgs10`` resolves ``assets/dgs10.csv``
verbatim — attribute names are the on-disk FRED stems, so ``dgs10`` /
``t10y2y`` / ``hyspread`` need no Python-identifier gymnastics (a stem that is
not a valid identifier is reachable through ``getattr(ctx.macro, name)``). An
unknown stem raises naming the available stems; it never degrades to an empty
series or a silent fallback. Stems are derived from the filesystem once per
run, not from a hard-coded registry.

**The lookahead rule.** A FRED daily print dated ``D`` is that day's CLOSE.
So a strategy on an INTRADAY bar dated ``D`` cannot know it: visible prints are
strictly ``date < D`` (one full session of lag). A base bar of one calendar day
or longer is labelled by its bucket date and carries that day's close, so at
such a bar a print dated ``D`` is legitimately in the past and is visible.
:func:`visible_prints` is that rule, in one testable place. The daily arm is
therefore only sound on a COMPLETED bar: a run whose last bar is still forming
(the live/screen path) must not treat ``bars: ["1d"]`` as live-safe. Macro is
also a static file, so a live strategy gates freshness itself by reading
``ctx.macro.<stem>.date`` (see :meth:`MacroSeries.date` for what the date is).

Deliberately absent: z-scores, yield-curve spreads, TTM-style derived ratios.
The library serves the series plus ``[-1]``/``len``/``last``/``date``/``rank``;
a strategy computes anything else from those, because the transformation
depends on the strategy's intent, not on the library's.

Every store is minted per run (never module state), so sweep/split/optimize
workers can neither share a cache nor a cursor.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import cast

import pandas as pd

from src.indicators.macro import DEFAULT_ASSETS_DIR, load_macro_series

#: A cursor: ``pd.Timestamp``-normalizable, or None before the engine advances.
Cursor = pd.Timestamp | str | date | datetime | None

#: On-disk directory holding the FRED assets (``assets/<stem>.csv``), resolved
#: repo-relative so a read never depends on the process cwd.
_DEFAULT_DIR = DEFAULT_ASSETS_DIR

#: Where a missing FRED stem comes from — quoted in the error so the fix is
#: obvious. ``cpi`` is World Bank data, not FRED, so it gets its own text.
_FETCH_HINT = "uv run python scripts/fetch_macro_fred.py"
_CPI_HINT = "the World Bank CPI file (annual inflation chained into a level)"

# Interval grammar: ``<number><unit>`` with an optional space (``"1d"``, ``"1 day"``).
_INTERVAL_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([a-z]+)\s*$", re.IGNORECASE)
#: Units that are exactly one calendar day.
_DAY_UNITS = frozenset({"d", "day", "days"})
#: Units that are one day or longer.
_MULTI_DAY_UNITS = frozenset(
    {"w", "wk", "week", "weeks", "mo", "mon", "month", "months", "y", "yr", "year"}
)


def is_daily_or_larger(interval: str | None) -> bool:
    """True when a bar of ``interval`` covers at least one calendar day.

    ``None`` (no interval known) is treated as intraday — the conservative
    arm of :func:`visible_prints`. Unrecognised units are intraday too: losing
    a same-day print is safe, showing one is not.
    """
    if interval is None:
        return False
    text = interval.strip().lower()
    if text in {"d", "1d"}:
        return True
    match = _INTERVAL_RE.match(text)
    if match is None:
        return False
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit in _DAY_UNITS:
        return value >= 1.0
    return unit in _MULTI_DAY_UNITS


def visible_prints(
    dates: Sequence[pd.Timestamp],
    cursor: Cursor,
    interval: str | None,
) -> int:
    """Count of prints visible at ``cursor`` for a base bar of ``interval``.

    ``dates`` must be ascending (they are the FF'd daily grid). The rule:

    * intraday bar dated ``D`` — returns the prints dated before ``D`` (strict),
      because ``D``'s print *is* ``D``'s close and is not public intraday;
    * bar of one calendar day or longer dated ``D`` — returns the prints dated
      at/before ``D``, because a *completed* such bar carries ``D``'s close.
      Only sound on a completed bar; a forming daily bar is not covered.

    ``None`` cursor (no cursor reader bound) means "not backtesting": every
    print is visible, mirroring ``SeriesPIT``. A bound-but-unadvanced cursor
    (the reader returns ``None``) sees nothing — :class:`MacroSeries` separates
    those two cases before calling in; passing ``None`` here always means
    "unbound".
    """
    assert all(a <= b for a, b in zip(dates, dates[1:], strict=False)), (
        "visible_prints needs ascending dates"
    )
    if cursor is None:
        return len(dates)
    stamp = pd.Timestamp(cursor)
    if pd.isna(stamp):
        return 0
    # ``pd.Timestamp(...)`` is typed ``Timestamp | NaTType`` and ``pd.isna`` does
    # not narrow that for the checker; the cast records the guaranteed fact (NaT
    # was returned above).
    day = cast(pd.Timestamp, stamp).normalize()
    # One nanosecond before midnight excludes a print dated ``D``; using the
    # day itself (side="right") includes it for a completed daily bucket.
    cutoff = day if is_daily_or_larger(interval) else day - pd.Timedelta(1, "ns")
    return bisect.bisect_right(dates, cutoff)


class MacroSeries:
    """Cursor-safe view of one macro print series (``ctx.macro.vix``).

    ``dates``/``values`` are the daily forward-filled grid loaded once per run;
    visibility is re-derived from the *bound cursor* at every read
    (:meth:`_visible`), never snapshotted, so a series object captured on an
    earlier bar cannot serve a stale or future window.
    """

    __slots__ = ("_name", "_dates", "_values", "_cursor", "_interval")

    def __init__(
        self,
        name: str,
        dates: tuple[pd.Timestamp, ...],
        values: tuple[float, ...],
        cursor: Callable[[], Cursor] | None,
        interval: str | None,
    ) -> None:
        self._name = name
        self._dates = dates
        self._values = values
        self._cursor = cursor
        self._interval = interval

    @property
    def name(self) -> str:
        """On-disk stem this series was loaded from (e.g. ``"vix"``)."""
        return self._name

    def _visible(self) -> int:
        """Prints visible right now.

        Two distinct cases, kept apart on purpose (as ``SeriesPIT`` does): no
        cursor reader bound at all means "not backtesting" and serves the whole
        series; a reader that answers ``None`` means the engine has not advanced
        yet, and nothing is visible. Collapsing them would expose the file's tail
        before the first bar.
        """
        if self._cursor is None:
            return len(self._values)
        stamp = self._cursor()
        if stamp is None:
            return 0
        return visible_prints(self._dates, stamp, self._interval)

    def __len__(self) -> int:
        return self._visible()

    def _at(self, i: int) -> float:
        n = self._visible()
        idx = i if i >= 0 else n + i
        if idx < 0 or idx >= n:
            raise IndexError(
                f"index {i} out of range on {n}-print macro series {self._name!r}"
            )
        return self._values[idx]

    def __getitem__(self, i: int | slice) -> float | tuple[float, ...]:
        """Value at ``i`` (negative counts back from the latest visible print).

        A slice returns a tuple of floats in date order. Out-of-range reads
        raise ``IndexError`` — there is no NaN padding: an absent print is
        absent.
        """
        if isinstance(i, slice):
            n = self._visible()
            start, stop, step = i.indices(n)
            return tuple(self._values[j] for j in range(start, stop, step))
        return self._at(i)

    def last(self) -> float:
        """Latest visible print (``IndexError`` when nothing is visible)."""
        return self._at(-1)

    @property
    def date(self) -> pd.Timestamp:
        """Latest visible GRID row (``IndexError`` when none).

        The grid is forward-filled, so this may be a weekend/holiday carrying
        the prior print. For a print's own release date compare across two
        cursors, or gate freshness with a tolerance wider than a weekend.
        """
        n = self._visible()
        if n == 0:
            raise IndexError(f"no visible print yet for macro series {self._name!r}")
        return self._dates[n - 1]

    def rank(self, window: int) -> float:
        """Fraction of the prior visible prints below the latest.

        The trailing-percentile pattern strategies use for a stress regime.
        ``window`` counts visible daily GRID rows, not FRED releases, so a
        window of 250 calendar rows spans roughly 172 prints.

        Short history is CLIPPED, not padded: with fewer than ``window`` prior
        rows the fraction uses the last ``min(window, len - 1)`` visible rows,
        so the caller sees the statistic the data supports and can gate on
        ``len(series)`` for a full window. ``IndexError`` only when nothing (or
        nothing prior) is visible — the fraction is undefined then, and a
        neutral default would silently switch a regime filter off.
        """
        if window < 1:
            raise ValueError(f"rank window must be >= 1, got {window}")
        n = self._visible()
        if n == 0:
            raise IndexError(f"no visible print yet for macro series {self._name!r}")
        start = max(0, n - 1 - window)
        history = self._values[start : n - 1]
        if not history:
            raise IndexError(
                f"rank needs a prior visible print; {self._name!r} has only one"
            )
        latest = self._values[n - 1]
        return sum(1 for value in history if value < latest) / len(history)

    def __repr__(self) -> str:
        return f"MacroSeries({self._name!r}, visible={self._visible()})"


class Macro:
    """Macro accessor minted per run and surfaced as ``ctx.macro``.

    Series are loaded from ``<source_dir>/<stem>.csv`` on first attribute
    access and memoised for the run, so a strategy never pays disk per read and
    a run that reads no macro never touches the filesystem. The available stems
    are listed from the directory once (never a hard-coded registry).

    ``getattr(ctx.macro, stem, None)`` is the one idiom to avoid: it turns a
    missing asset into ``None``, i.e. a silently disabled filter. An unknown
    stem raises on plain attribute access by design.
    """

    __slots__ = ("_dir", "_cursor", "_interval", "_cache", "_stems")

    def __init__(
        self,
        source_dir: str | Path | None = None,
        cursor: Callable[[], Cursor] | None = None,
        interval: str | None = None,
    ) -> None:
        """``interval=None`` is the intraday (conservative) arm of the rule."""
        self._dir = Path(source_dir) if source_dir is not None else _DEFAULT_DIR
        self._cursor = cursor
        self._interval = interval
        self._cache: dict[str, tuple[tuple[pd.Timestamp, ...], tuple[float, ...]]] = {}
        self._stems: tuple[str, ...] | None = None

    @property
    def stems(self) -> tuple[str, ...]:
        """Available macro stems (``<source_dir>/*.csv``), listed once per run."""
        if self._stems is None:
            self._stems = tuple(
                sorted(p.stem for p in self._dir.glob("*.csv") if p.is_file())
            )
        return self._stems

    def bind_cursor(
        self,
        cursor: Callable[[], Cursor],
        interval: str,
    ) -> None:
        """Bind the engine cursor reader and the base bar's interval (engine).

        Both are read on every access, so advancing the cursor is immediately
        reflected in visibility. ``interval`` decides the lookahead rule (see
        :func:`visible_prints`), so it must be the signal bar's own interval —
        the engine always passes ``config.bars[0]``.
        """
        assert interval, "a macro store needs the base bar's interval"
        self._cursor = cursor
        self._interval = interval

    def __getattr__(self, name: str) -> MacroSeries:
        # Guard dunder/private probes (copy, pickle, hasattr on internals).
        if name.startswith("_"):
            raise AttributeError(name)
        stems = self.stems
        if name not in stems:
            hint = (
                _CPI_HINT
                if name == "cpi"
                else f"`{_FETCH_HINT}` so `{self._dir / (name + '.csv')}` exists"
            )
            raise AttributeError(
                f"unknown macro series {name!r}; available stems: "
                f"{', '.join(stems) or '(none)'}. To add one, provide {hint}."
            )
        cached = self._cache.get(name)
        if cached is None:
            series = load_macro_series(name, source_dir=self._dir)
            cached = (
                tuple(cast(pd.Timestamp, pd.Timestamp(ts)) for ts in series.index),
                tuple(float(v) for v in series.to_numpy(dtype=float)),
            )
            self._cache[name] = cached
        dates, values = cached
        return MacroSeries(name, dates, values, self._cursor, self._interval)

    def __dir__(self) -> list[str]:
        """Expose the filesystem-derived stems to introspection."""
        return sorted({*super().__dir__(), *self.stems})

    def __repr__(self) -> str:
        return f"Macro(dir={str(self._dir)!r}, loaded={sorted(self._cache)})"


def load_macro(source_dir: str | Path | None = None) -> Macro:
    """Per-run ``Macro`` store — nothing is read here (lazy, like fundamentals)."""
    return Macro(source_dir=source_dir)


__all__ = [
    "Cursor",
    "Macro",
    "MacroSeries",
    "is_daily_or_larger",
    "load_macro",
    "visible_prints",
]
