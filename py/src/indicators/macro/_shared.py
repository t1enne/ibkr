"""Macro asset indicators — data loaders and the one public factory.

This module is the single implementation for the ``src.indicators.macro``
package. The public APIs are the factory :func:`init_macro_indicator` (a
lookahead-free ``f(ts)`` cursor reader) and :func:`load_macro_series` (the one
loader, shared with the DSL's ``ctx.macro``):

    from src.indicators.macro import init_macro_indicator

    cpi = init_macro_indicator("cpi")                 # loads assets/cpi.csv once
    v   = cpi(ts)                                     # latest CPI level at/before ts

    payems = init_macro_indicator("payems",           # load once, reuse cheaply
                                  series=my_preloaded_series)
    n  = payems(ts)

:func:`init_macro_indicator` returns a lookahead-free callable ``f(ts) -> float
| None``. The asset file is read **once** at init (or a pre-loaded Series is
bound directly); every later ``f(ts)`` call is a single ``Series.asof`` — no
re-reading the disk per call.

Two kinds of source files are supported transparently by name:

- **FRED two-column** (``assets/<name>.csv``: ``date``, ``<name>`` with blank
  no-print cells) — the default for any name other than ``"cpi"``. Handled by
  :func:`load_daily`, which forward-fills blanks onto a daily grid.
- **World Bank CPI** (``assets/cpi.csv``: ``Country,Country Code,Year,CPI``
  annual inflation rates) — handled by :func:`load_cpi_price_index`, which
  chains the rates into a 1.0-based daily price-index *level*.

All series are forward-filled on load and selected at the cursor with
:func:`at_cursor` (``Series.asof``), so a call can never leak a future print
into the current ``ts``.

:func:`deflated_log_prices` is retained as a **pure vectorised helper** (not a
macro indicator): ``ln(nominal) - ln(cpi)`` real-price series, consumed by the
RDD strategy. It is not part of the factory cursor API.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

#: Repo root, resolved file-relative so a macro read never depends on the
#: process cwd (``src/db/path.py``'s rule: never read the wrong file because
#: the process started elsewhere). ``parents[3]`` = the ``py/`` package root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
#: Directory holding the asset CSVs, exported so the DSL's ``ctx.macro`` reads
#: the same place this module does.
DEFAULT_ASSETS_DIR = _REPO_ROOT / "assets"

_DEFAULT_DIR = DEFAULT_ASSETS_DIR
_DEFAULT_CPI_PATH = DEFAULT_ASSETS_DIR / "cpi.csv"
_USA = "USA"

# Cursor timestamp type accepted by :func:`init_macro_indicator` accessors.
_TsT = pd.Timestamp | str | int

# A macro indicator: bind a series once, then read it lookahead-free at a cursor.
MacroIndicator = Callable[[_TsT], float | None]

# ---------------------------------------------------------------------------
# loaders (internal)
# ---------------------------------------------------------------------------


def load_daily(
    name: str,
    source_dir: str | Path | None = None,
) -> pd.Series:
    """Load ``<name>`` FRED asset file and return a daily forward-filled Series.

    Contract matches ``assets/dgs10.csv``: two columns (``date`` YYYY-MM-DD,
    ``<name>``), with FRED no-print dates as blank cells. Blanks coerce to NaN,
    are dropped, then the parsed series is reindexed to a daily grid with
    ``method="ffill"`` so every element is the latest known print.

    Args:
        name: on-disk series stem, e.g. ``"payems"``.
        source_dir: directory holding ``<name>.csv`` (default
            :data:`DEFAULT_ASSETS_DIR` — repo-relative, never cwd-relative).

    Returns:
        Daily ``pd.Series`` named ``name``, forward-filled over its span.

    Raises:
        FileNotFoundError: the asset file does not exist (the message names the
            fetch script).
        ValueError: the file is not the two-column FRED shape.
    """
    src = source_dir if source_dir is not None else _DEFAULT_DIR
    path = Path(src) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"macro asset file not found at {path}. "
            "Run `uv run python scripts/fetch_macro_fred.py` to download it."
        )
    df = pd.read_csv(
        path,
        names=["date", name],
        skiprows=0,
        parse_dates=["date"],
        date_format="%Y-%m-%d",
    )
    if df.shape[1] > 2:
        # The FRED contract is two columns. A wider file read positionally would
        # silently truncate to the first two, turning an unrelated CSV into a
        # plausible-looking series -- refuse instead.
        raise ValueError(
            f"macro asset {path} has {df.shape[1]} columns; expected 2 (date,{name})"
        )
    df = df.dropna(subset=["date"])
    s = pd.to_numeric(df[name], errors="coerce").dropna()
    s.index = pd.DatetimeIndex(df.loc[s.index, "date"])
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if s.empty:
        return pd.Series(dtype=float, name=name)
    daily: pd.Series = s.reindex(
        pd.date_range(s.index.min(), s.index.max(), freq="D"), method="ffill"
    )
    return daily.rename(name)


def load_cpi_price_index(
    source: str | Path | BinaryIO | None = None,
    country: str = _USA,
) -> pd.Series:
    """Return a daily cumulative CPI price-index level series (base 1.0).

    Reads the World Bank annual-inflation file (one row per country-year),
    keeps ``country``, chains the annual percent rates into a 1.0-based index,
    and steps it onto a daily grid with ``method="ffill"`` (each year's rate
    applies across that year, and the final rate forward-fills past the file's
    last year).
    """
    df = _read_frame(source)
    sub = df[df["country_code"].astype(str).str.upper() == country.upper()]
    if sub.empty:
        raise ValueError(f"No CPI rows for country code {country!r} in source")
    sub = sub.sort_values("year")

    years = sub["year"].astype(int).to_numpy()
    rates = sub["cpi"].astype(float).to_numpy() / 100.0
    levels = np.concatenate(([1.0], np.cumprod(1.0 + rates)))
    idx_years = np.concatenate(([years[0]], years + 1))

    annual = pd.Series(
        levels, index=[pd.Timestamp(int(y), 1, 1) for y in idx_years], name="cpi"
    )
    end = annual.index.max()
    daily: pd.Series = annual.reindex(
        pd.date_range(annual.index.min(), end + pd.Timedelta(days=366), freq="D"),
        method="ffill",
    )
    return daily


def _read_frame(source: str | Path | BinaryIO | None) -> pd.DataFrame:
    if source is None:
        if not Path(_DEFAULT_CPI_PATH).exists():
            raise FileNotFoundError(
                f"Default CPI file not found at {_DEFAULT_CPI_PATH}. "
                "Pass source=... to load_cpi_price_index."
            )
        df = pd.read_csv(_DEFAULT_CPI_PATH)
    else:
        df = pd.read_csv(source)  # type: ignore[arg-type]
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# pure helpers (not macro indicators)
# ---------------------------------------------------------------------------


def deflated_log_prices(
    nominal: pd.Series,
    cpi_index: pd.Series,
) -> pd.Series:
    """real_log = ln(nominal) - ln(cpi), reindexed to ``nominal`` and FF'd.

    Equals the log of real (inflation-adjusted) prices. No re-normalization to
    the nominal level is applied here because the consumer (e.g. the RDD ``real``
    feature) only uses the *z-score* of the series, which is invariant to a
    constant level shift (a constant in log space shifts mean and series
    equally, leaving z unchanged).

    Args:
        nominal: price Series.
        cpi_index: CPI price-index level Series (e.g. from
            :func:`load_cpi_price_index`).

    Returns:
        ``log(nominal) - log(cpi)`` aligned to ``nominal``'s index, forward-filled.
    """
    cpi = cpi_index.reindex(nominal.index, method="ffill")
    log_cpi = pd.Series(
        np.log(cpi.replace(0.0, np.nan)), index=nominal.index, name="lc"
    )
    log_px = pd.Series(
        np.log(nominal.replace(0.0, np.nan)), index=nominal.index, name="lp"
    )
    aligned = pd.concat([log_px, log_cpi], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float, name="real_log")
    return (aligned["lp"] - aligned["lc"]).rename("real_log")


# ---------------------------------------------------------------------------
# the one public factory
# ---------------------------------------------------------------------------


def init_macro_indicator(
    name: str,
    series: pd.Series | None = None,
    source_dir: str | Path | None = None,
    country: str = _USA,
) -> MacroIndicator:
    """Return a lookahead-free callable ``f(ts) -> float | None`` for ``name``.

    The returned closure retains a **single pre-loaded daily Series**, so the
    asset file is read at most once (at init) and every ``f(ts)`` call is just a
    ``Series.asof`` — it never re-reads disk per call.

    Usage::

        cpi = init_macro_indicator("cpi")     # loads assets/cpi.csv once
        level = cpi(ts)                       # latest CPI level at/before ts

        payems = init_macro_indicator("payems", series=load_daily("payems"))
        n = payems(ts)

    Args:
        name: macro asset name. ``"cpi"`` uses the World Bank chained-index
            loader; any other name reads ``assets/<name>.csv`` (FRED two-column
            format) via :func:`load_daily`.
        series: optional already-loaded daily Series to bind directly, skipping
            all file I/O. This is the fast path for backtest loops that load a
            bundle once.
        source_dir: directory holding the asset CSVs (default ``assets/``). For
            ``"cpi"`` the file ``<source_dir>/cpi.csv`` is used.
        country: country code for the CPI load (default ``"USA"``); ignored for
            FRED series.

    Returns:
        Callable ``f(ts)`` returning the latest ``float`` known at or before
        ``ts``, or ``None`` when ``ts`` precedes the series coverage.
    """
    s = series if series is not None else load_macro_series(name, source_dir, country)
    return MacroIndicatorLambda(s)


def load_macro_series(
    name: str,
    source_dir: str | Path | None = None,
    country: str = _USA,
) -> pd.Series:
    """Load one macro asset as a daily forward-filled Series (dispatch by name).

    The single loader behind :func:`init_macro_indicator` and the DSL's
    ``ctx.macro``, so both read the same series. ``"cpi"`` uses the World Bank
    chained-index loader; any other name reads ``<source_dir>/<name>.csv``
    (FRED two-column format) via :func:`load_daily`.

    Raises:
        FileNotFoundError: the named asset file does not exist (the message
            names the fetch script).
    """
    if name == "cpi":
        # World Bank chained-index loader takes a file path (or None for default).
        src = Path(source_dir) / "cpi.csv" if source_dir is not None else None
        return load_cpi_price_index(source=src, country=country)
    return load_daily(name, source_dir=source_dir)


class MacroIndicatorLambda:
    """Callable wrapper retaining a pre-loaded daily series + cursor lookup.

    Backs :func:`init_macro_indicator`. Calling ``inds(ts)`` delegates to
    :func:`at_cursor`, the lookahead-free selector. Kept as a tiny class (rather
    than a bare ``lambda``) so the wrapped series and name are introspectable
    and the callable repr is meaningful.
    """

    __slots__ = ("_series", "name")

    def __init__(self, series: pd.Series) -> None:
        self._series = series
        self.name: str = str(series.name)

    def __call__(self, ts: _TsT) -> float | None:
        return at_cursor(self._series, ts)


def at_cursor(
    series: pd.Series,
    ts: _TsT,
) -> float | None:
    """Latest value of ``series`` with a release date at/before ``ts``.

    Lookahead-free by construction: uses ``Series.asof(ts)``, which selects the
    value at the greatest index ``<= ts`` and ignores every later observation —
    so a call cannot leak a future macro print into the present cursor.

    Args:
        series: daily (or coarser) forward-filled macro Series.
        ts: cursor timestamp. Accepts anything ``pd.Timestamp`` normalises
            (datetime, ISO string, or epoch-seconds int).

    Returns:
        The latest ``float`` known at ``ts``, or ``None`` when ``ts`` precedes
        the series' coverage (warmup) or the series is empty.
    """
    if series.empty:
        return None
    # Integers are treated as epoch **seconds** (unambiguous); timestamps,
    # datetimes and ISO strings normalize via pd.Timestamp directly.
    cursor = (
        pd.Timestamp(ts, unit="s")
        if isinstance(ts, (int, np.integer))
        else pd.Timestamp(ts)
    )
    val = series.asof(cursor)
    return None if pd.isna(val) else float(val)


__all__ = ["init_macro_indicator", "deflated_log_prices", "load_macro_series"]
