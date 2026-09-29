"""TaContext — prefetched, cursor-safe indicator context for the strategy DSL.

Built once at engine start from the complete OHLCV feed, then handed to a
decorated strategy. Every indicator is computed **once** over the full series
(memoised per ``(symbol, indicator, period, *args)``), cached as a numpy array,
and exposed as a cursor-truncated :class:`~src.bt.strategies.series.SeriesView`
bounded by the engine's ``CandleStore`` cursor.

This structurally fixes the hot path in the old ``on_candle`` style, where
``ta.ema(closes, period).iloc[-1]`` recomputed the whole EMA *per symbol per
candle* (O(N²) total) and ``state.candles[(s, iv)]`` rebuilt a DataFrame per
access. Here a per-candle ``fast[-1]`` costs O(1) after a single full-series
compute.

Example from a DSL strategy::

    fast = ctx.ta.ema(\"AAPL\", 9)    # one EMA compute per engine run
    if ctx.cross_over(fast, slow):    # reads cursor-truncated O(1)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, TYPE_CHECKING

import numpy as np
import pandas as pd

from src.bt.strategies.series import SeriesView

if TYPE_CHECKING:
    from src.bt.engine.candle_store import CandleStore

_FIELDS = ("open", "high", "low", "close", "volume")

# A single indicator + its args identifies a cache entry. Hashable because all
# members are ints/strings.
CacheKey = tuple[str, str, tuple]

# Guard for the volume-weighted denominator: a window whose volume sums to
# (near) zero has no meaningful VWATR and must yield NaN, not inf.
_EPS = 1e-12


@dataclass(frozen=True)
class OhlcvView:
    """Cursor-truncated OHLCV for one symbol (``ctx.ohlcv``)."""

    open: SeriesView
    high: SeriesView
    low: SeriesView
    close: SeriesView
    volume: SeriesView

    @property
    def count(self) -> int:
        return len(self.close)


class TaContext:
    """Prefetched, cursor-safe indicator context over a complete OHLCV feed.

    ``data`` is a DataFrame whose columns are a MultiIndex of ``(symbol, field)``
    (the same shape ``candle_generator`` consumes). All per-symbol arrays are
    extracted up front; indicators are computed lazily and memoised. The cursor
    is read from a bound ``CandleStore`` at access time, never cached, so reads
    are always safe against lookahead.
    """

    __slots__ = (
        "_complete",
        "_cache",
        "_symbols",
        "_store",
        "_base_interval",
        "_compute_count",
    )

    def __init__(
        self,
        complete: dict[str, dict[str, np.ndarray]],
        symbols: tuple[str, ...],
        base_interval: str,
    ) -> None:
        self._complete = complete
        self._symbols = symbols
        self._base_interval = base_interval
        self._cache: dict[CacheKey, np.ndarray] = {}
        self._store: CandleStore | None = None
        self._compute_count = 0  # number of full-series indicator computes

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_data(
        cls,
        data: pd.DataFrame,
        symbols: Iterable[str],
        base_interval: str,
    ) -> "TaContext":
        """Build a TaContext from a MultiIndex-column OHLCV DataFrame.

        Extracts per-symbol numpy arrays once (a cheap copy of the base arrays,
        shared by every indicator via cached compute).
        """
        if hasattr(data.columns, "levels") and isinstance(data.columns, pd.MultiIndex):
            cols = data.columns
        else:
            cols = None

        complete: dict[str, dict[str, np.ndarray]] = {}
        for sym in symbols:
            arrays: dict[str, np.ndarray] = {}
            if cols is not None and (sym, "close") in cols:
                # Align the TaContext feed with the engine's CandleStore 1:1.
                # ``candle_generator`` skips a symbol's candle whenever its close
                # is non-finite (data gap / closed market), so the store only ever
                # accumulates rows with a finite close. We must drop the same rows
                # here, otherwise ``cursor_count`` (store-based) and ``values[idx]``
                # (this feed) point at different timestamps -> silent misalignment.
                mask = np.isfinite(data[(sym, "close")].to_numpy(dtype=np.float64))
                for f in _FIELDS:
                    if (sym, f) in cols:
                        arrays[f] = data[(sym, f)].to_numpy(dtype=np.float64)[mask]
                    else:
                        arrays[f] = np.array([], dtype=np.float64)
            else:
                for f in _FIELDS:
                    arrays[f] = np.array([], dtype=np.float64)
            complete[sym] = arrays
        return cls(complete, tuple(symbols), base_interval)

    def bind(self, store: "CandleStore") -> None:
        """Bind the engine's CandleStore to share its cursor."""
        self._store = store

    @property
    def compute_count(self) -> int:
        """Number of full-series indicator computations performed so far.

        A strategy should pay exactly one compute per ``(sym, indicator, period)"
        access; a later read of the same key is a cache hit (count unchanged).
        """
        return self._compute_count

    # -- internal helpers -----------------------------------------------------

    def _series(self, sym: str, interval: str | None) -> dict[str, np.ndarray]:
        """Per-symbol field arrays; ``interval`` must match the signal bar."""
        iv = interval or self._base_interval
        if iv != self._base_interval:
            raise ValueError(
                f"TaContext only serves the signal interval '{self._base_interval}' "
                f"(got '{iv}'). Indicators over higher timeframes are a follow-up."
            )
        try:
            return self._complete[sym]
        except KeyError:
            raise KeyError(f"symbol {sym!r} not in TaContext feed") from None

    def _lengther(self, sym: str, interval: str | None) -> Callable[[], int]:
        """Return a lengther reading the cursor-truncated count from the store."""
        iv = interval or self._base_interval
        store = self._store

        def _len() -> int:
            if store is None:
                arr = self._complete[sym].get("close")
                return len(arr) if arr is not None else 0
            n = store.cursor_count(sym, iv)
            close_arr = self._complete[sym].get("close")
            feed_n = len(close_arr) if close_arr is not None else 0
            if n > feed_n:
                # The store accumulated more bars than this symbol's aligned feed
                # holds - the feed was gapped/NaN-filtered inconsistently, so
                # ``values[idx]`` would read a future/stale bar. Fail loudly rather
                # than serve misaligned data.
                raise RuntimeError(
                    f"TaContext feed for {sym!r} has {feed_n} rows but CandleStore "
                    f"accumulated {n} up to the cursor; feed and store are "
                    "misaligned (NaN/data-gap rows were not filtered consistently)."
                )
            return n

        return _len

    def _view(self, sym: str, values: np.ndarray, interval: str | None) -> SeriesView:
        return SeriesView(values, self._lengther(sym, interval))

    def _compute(
        self,
        key: CacheKey,
        fn,
    ) -> np.ndarray:
        """Memoised full-series indicator compute, keyed by ``key``."""
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        self._compute_count += 1
        result = fn()
        arr = np.asarray(result, dtype=np.float64)
        self._cache[key] = arr
        return arr

    # -- raw OHLCV ---------------------------------------------------------------

    def ohlcv(self, sym: str, interval: str | None = None) -> OhlcvView:
        """Cursor-truncated OHLCV for ``sym`` (fields as SeriesViews)."""
        arrays = self._series(sym, interval)
        iv = interval or self._base_interval
        return OhlcvView(
            open=self._view(sym, arrays["open"], iv),
            high=self._view(sym, arrays["high"], iv),
            low=self._view(sym, arrays["low"], iv),
            close=self._view(sym, arrays["close"], iv),
            volume=self._view(sym, arrays["volume"], iv),
        )

    def field(self, sym: str, field: str, interval: str | None = None) -> SeriesView:
        """Cursor-truncated raw field (``'open'|'high'|'low'|'close'|'volume'``)."""
        if field not in _FIELDS:
            raise ValueError(f"unknown field {field!r}; choose from {_FIELDS}")
        return self._view(sym, self._series(sym, interval)[field], interval)

    # -- indicator accessors (single full-series compute, then O(1) reads) -------

    def ema(self, sym: str, period: int, interval: str | None = None) -> SeriesView:
        """Pine ``ema`` — exponential moving average over ``period`` bars."""
        from src.indicators.ta import ema as _ema

        key: CacheKey = ("ema", sym, (period,))
        arr = self._compute(
            key, lambda: _pnl(self._series(sym, interval)["close"], period, _ema)
        )
        return self._view(sym, arr, interval)

    def sma(self, sym: str, period: int, interval: str | None = None) -> SeriesView:
        """Pine ``sma`` — simple moving average over ``period`` bars."""
        from src.indicators.ta import sma as _sma

        key: CacheKey = ("sma", sym, (period,))
        arr = self._compute(
            key, lambda: _pnl(self._series(sym, interval)["close"], period, _sma)
        )
        return self._view(sym, arr, interval)

    def atr(
        self, sym: str, period: int = 14, interval: str | None = None
    ) -> SeriesView:
        """Pine ``atr`` — Wilder average true range over ``period`` bars."""
        from src.indicators.ta import atr as _atr

        key: CacheKey = ("atr", sym, (period,))
        a = self._series(sym, interval)

        def _calc() -> np.ndarray:
            return _atr(
                _ser(a["high"]), _ser(a["low"]), _ser(a["close"]), period
            ).to_numpy()

        return self._view(sym, self._compute(key, _calc), interval)

    def adx(
        self, sym: str, period: int = 14, interval: str | None = None
    ) -> SeriesView:
        """Pine ``adx`` — average directional index over ``period`` bars."""
        from src.indicators.ta import adx as _adx

        key: CacheKey = ("adx", sym, (period,))
        a = self._series(sym, interval)

        def _calc() -> np.ndarray:
            return _adx(
                _ser(a["high"]), _ser(a["low"]), _ser(a["close"]), period
            ).to_numpy()

        return self._view(sym, self._compute(key, _calc), interval)

    def rsi(
        self, sym: str, period: int = 14, interval: str | None = None
    ) -> SeriesView:
        """Pine ``rsi`` — relative strength index over ``period`` bars."""
        from src.indicators.ta import rsi as _rsi

        key: CacheKey = ("rsi", sym, (period,))
        arr = self._compute(
            key, lambda: _pnl(self._series(sym, interval)["close"], period, _rsi)
        )
        return self._view(sym, arr, interval)

    def mfi(
        self, sym: str, period: int = 14, interval: str | None = None
    ) -> SeriesView:
        """Pine ``mfi`` — money flow index over ``period`` bars (0..100)."""
        from src.indicators.ta import mfi as _mfi

        key: CacheKey = ("mfi", sym, (period,))
        a = self._series(sym, interval)

        def _calc() -> np.ndarray:
            return _mfi(
                _ser(a["high"]),
                _ser(a["low"]),
                _ser(a["close"]),
                _ser(a["volume"]),
                period,
            ).to_numpy()

        return self._view(sym, self._compute(key, _calc), interval)

    def obv_z(
        self, sym: str, window: int = 40, interval: str | None = None
    ) -> SeriesView:
        """OBV level z-score over ``window`` bars — the direct cumulative-flow
        read (sign = flow direction, magnitude = strength off the names' own
        noise floor).
        """
        from src.indicators.ta import obv_z as _obv_z

        key: CacheKey = ("obv_z", sym, (window,))
        a = self._series(sym, interval)

        def _calc() -> np.ndarray:
            return _obv_z(_ser(a["close"]), _ser(a["volume"]), window).to_numpy()

        return self._view(sym, self._compute(key, _calc), interval)

    def vwatr(
        self, sym: str, period: int = 14, interval: str | None = None
    ) -> SeriesView:
        """Volume-weighted ATR: ``smooth(TR * volume) / smooth(volume)``.

        ``smooth`` is the plain trailing rolling mean over ``period`` bars (NOT
        Wilder-smoothed like :meth:`atr`) and ``TR`` is the standard true range.
        Same units as price, so a stop distance in VWATR is a dollar distance.
        """
        key: CacheKey = ("vwatr", sym, (period,))
        a = self._series(sym, interval)

        def _calc() -> np.ndarray:
            return _vwatr_np(a["high"], a["low"], a["close"], a["volume"], period)

        return self._view(sym, self._compute(key, _calc), interval)

    def true_range(self, sym: str, interval: str | None = None) -> SeriesView:
        """Standard true range ``max(high-low, |high-prev_close|, |low-prev_close|)``.

        First bar is NaN (no prior close). The Wilder-*smoothed* form is
        :meth:`atr`; this accessor exposes the raw bar ranges so a caller can
        smooth them its own way.
        """
        key: CacheKey = ("true_range", sym, ())
        a = self._series(sym, interval)
        arr = self._compute(
            key, lambda: _true_range_np(a["high"], a["low"], a["close"])
        )
        return self._view(sym, arr, interval)

    def plain_atr(
        self, sym: str, period: int = 14, interval: str | None = None
    ) -> SeriesView:
        """**Plain** rolling-mean average true range over ``period`` bars.

        Distinct from :meth:`atr`, which is Wilder-smoothed (``ewm(alpha=1/n)``).
        Same units and same true range, different smoothing — the divergence
        exits that normalize by range want this one.
        """
        key: CacheKey = ("plain_atr", sym, (period,))
        a = self._series(sym, interval)
        arr = self._compute(
            key,
            lambda: _rolling_mean_np(
                _true_range_np(a["high"], a["low"], a["close"]), period
            ),
        )
        return self._view(sym, arr, interval)

    def vwatr_baseline(
        self, sym: str, period: int = 14, window: int = 60, interval: str | None = None
    ) -> SeriesView:
        """Plain trailing rolling mean of :meth:`vwatr` over ``window`` bars.

        The strategy's ``vwatr_base_win`` baseline: an expansion reference, not
        a signal. Computed from the VWATR series itself, so the head is NaN
        until both the VWATR warm-up (``period`` bars) and the baseline window
        are satisfied (``period + window - 1`` bars).
        """
        key: CacheKey = ("vwatr_baseline", sym, (period, window))
        a = self._series(sym, interval)

        def _calc() -> np.ndarray:
            return _rolling_mean_np(
                _vwatr_np(a["high"], a["low"], a["close"], a["volume"], period),
                window,
            )

        return self._view(sym, self._compute(key, _calc), interval)

    def highest(self, sym: str, period: int, interval: str | None = None) -> SeriesView:
        """Pine ``highest`` — rolling maximum over ``period`` bars (incl. current)."""
        return self._rolling_extreme(sym, "highest", period, interval)

    def lowest(self, sym: str, period: int, interval: str | None = None) -> SeriesView:
        """Pine ``lowest`` — rolling minimum over ``period`` bars (incl. current)."""
        return self._rolling_extreme(sym, "lowest", period, interval)

    def sum(self, sym: str, period: int, interval: str | None = None) -> SeriesView:
        """Pine ``sum`` — rolling sum (e.g. volume) over ``period`` bars."""
        key: CacheKey = ("sum", sym, (period,))
        values = self._series(sym, interval)["close"]
        arr = self._compute(key, lambda: _rolling_sum_np(values, period))
        return self._view(sym, arr, interval)

    def _rolling_extreme(
        self,
        sym: str,
        kind: str,
        period: int,
        interval: str | None,
    ) -> SeriesView:
        key: CacheKey = (kind, sym, (period,))
        arr = self._compute(
            key,
            lambda: _rolling_extreme_np(
                self._series(sym, interval)["close"], kind, period
            ),
        )
        return self._view(sym, arr, interval)

    def close(self, sym: str, interval: str | None = None) -> SeriesView:
        """Cursor-truncated close series (shorthand for ``ctx.ta.field(...)``)."""
        return self.field(sym, "close", interval)


# ---------------------------------------------------------------------------
# small pandas/numpy helpers (indicator compute is once-per-series, so pandas
# overhead here is amortised across every per-candle read)
# ---------------------------------------------------------------------------


def _ser(arr: np.ndarray) -> pd.Series:
    return pd.Series(arr, dtype=np.float64)


def _pnl(values: np.ndarray, period: int, fn) -> np.ndarray:
    """Run a pandas indicator over one series, returning the numpy result."""
    return fn(_ser(values), period).to_numpy()


def _rolling_extreme_np(values: np.ndarray, kind: str, period: int) -> np.ndarray:
    """Rolling ``max``/``min`` (window includes current bar), NaN head preserved."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if period < 1 or len(values) == 0:
        return out
    from numpy.lib.stride_tricks import sliding_window_view

    out[period - 1 :] = (
        sliding_window_view(values, period).max(axis=1)
        if kind == "highest"
        else sliding_window_view(values, period).min(axis=1)
    )
    return out


def _rolling_sum_np(values: np.ndarray, period: int) -> np.ndarray:
    """Rolling sum over ``period`` bars, NaN head preserved (Pine ``sum``)."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if period < 1 or len(values) == 0:
        return out
    from numpy.lib.stride_tricks import sliding_window_view

    out[period - 1 :] = sliding_window_view(values, period).sum(axis=1)
    return out


def _rolling_mean_np(values: np.ndarray, window: int) -> np.ndarray:
    """Plain trailing rolling mean (NaN propagates, ``skipna`` is NOT used)."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if window < 1 or len(values) < window:
        return out
    from numpy.lib.stride_tricks import sliding_window_view

    out[window - 1 :] = sliding_window_view(values, window).mean(axis=1)
    return out


def _true_range_np(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True range; the first bar is NaN (no previous close)."""
    n = len(close)
    tr = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return tr
    prev_close = close[:-1]
    hl = high[1:] - low[1:]
    hc = np.abs(high[1:] - prev_close)
    lc = np.abs(low[1:] - prev_close)
    tr[1:] = np.maximum(np.maximum(hl, hc), lc)
    return tr


def _vwatr_np(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    period: int,
) -> np.ndarray:
    """Volume-weighted ATR: ``smooth(TR * volume) / smooth(volume)``.

    ``smooth`` is a plain trailing rolling mean, and true range is itself NaN on
    bar 0, so the first ``period`` bars are NaN (NOT ``period - 1``: the NaN
    propagates through the window rather than being skip-filled). A window whose
    volume sum is ~0 yields NaN rather than inf.
    """
    tr = _true_range_np(high, low, close)
    num = _rolling_mean_np(tr * volume, period)
    den = _rolling_mean_np(volume, period)
    return _safe_div_np(num, den)


def _safe_div_np(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Elementwise ``num / den``; non-finite or ~zero denominator -> NaN."""
    out = np.full(num.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > _EPS)
    out[ok] = num[ok] / den[ok]
    return out


__all__ = ["TaContext", "OhlcvView", "init_ta"]


def init_ta(
    data: pd.DataFrame,
    symbols: Iterable[str],
    base_interval: str,
) -> TaContext:
    """Convenience factory: build a TaContext from the base OHLCV feed.

    ``data`` must be the same MultiIndex-column DataFrame underlying the candle
    generator so indicator arrays line up 1:1 with the engine's cursor.
    """
    return TaContext.from_data(data, symbols, base_interval)
