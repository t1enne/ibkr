"""Online (incremental) adaptive entropy trend indicator.

Strategy-owned counterpart to :func:`src.indicators.adaptive_entropy.pure.adaptive_entropy`.
Feeds one bar at a time through :meth:`observe` and returns the current
:class:`AdaptiveEntropyResult`.  State lives entirely on the instance so a
fresh instance per run/window (e.g. keyed in ``ctx.shared``) is safe across
split/sweep workers.

The online path recomputes only the *active* window on each call (O(lookback ×
bins) for the entropy histogram plus O(1) amortised for ATR) rather than the
whole series, and keeps the adaptive EMA and trend as running scalars.  It
numerically matches the batch function bar-for-bar.
"""

from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np

from src.indicators.adaptive_entropy.types import (
    AdaptiveEntropyConfig,
    AdaptiveEntropyResult,
)
from src.indicators.adaptive_entropy.validate import require_finite_bar


class OnlineAdaptiveEntropy:
    """Incremental adaptive entropy trend, owned by a strategy via ``ctx.shared``.

    Parameters
    ----------
    config:
        Indicator hyper-parameters.  Defaults to ``AdaptiveEntropyConfig()``.
    """

    def __init__(self, config: AdaptiveEntropyConfig | None = None) -> None:
        cfg = config or AdaptiveEntropyConfig()
        self._cfg = cfg
        self._w = cfg.lookback

        # Rolling log-return window (for Shannon entropy).
        self._returns: Deque[float] = deque()

        # Wilder RMA ATR is a running scalar.
        self._atr: float | None = None
        self._alpha_atr = 1.0 / self._w

        # Adaptive EMA + trend are running scalars.
        self._ema: float | None = None
        self._trend: int = 0

        self._prev_close: float | None = None
        self._n: int = 0

    # ------------------------------------------------------------------
    # read-only state
    # ------------------------------------------------------------------

    @property
    def n_bars(self) -> int:
        """Total bars passed to ``observe``."""
        return self._n

    @property
    def ready(self) -> bool:
        """True once the entropy window / ATR / EMA are all defined."""
        return self._ema is not None and self._atr is not None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all accumulated state.

        Reconstructing the instance per run also achieves this; ``reset()`` is
        provided for the runner's defensive ``reset_global()`` protocol.
        """
        self._returns.clear()
        self._atr = None
        self._ema = None
        self._trend = 0
        self._prev_close = None
        self._n = 0

    def observe(self, close: float, high: float, low: float) -> AdaptiveEntropyResult:
        """Feed one OHLC bar and return the current indicator snapshot.

        Parameters
        ----------
        close, high, low:
            Current bar's close / high / low.  ``open`` is unused (the Pine
            script only reads close[1] / close, high, low).

        Returns
        -------
        AdaptiveEntropyResult
            Snapshot for this bar.  Until the lookback window has filled the
            `inner_*`/`outer_*` bands are collapsed onto the EMA and ``trend``
            is the previous value, so callers should gate signals on
            :attr:`ready`.

        Raises
        ------
        NonFiniteBarError
            If any of ``close``/``high``/``low`` is not finite.  The model's
            EMA and ATR are running scalars, so a junk bar would poison them
            for the rest of the run and freeze ``trend`` without any signal to
            the caller.  State is untouched when this raises.
        """
        cfg = self._cfg
        close = float(close)
        high = float(high)
        low = float(low)
        require_finite_bar(close, high, low, bar=self._n)

        # ---- seed (first bar) ----
        if self._n == 0:
            self._ema = close
            self._prev_close = close
            self._n = 1
            return self._emit(close=close, entropy=0.5, strength=0.5)

        prev_close = float(self._prev_close) if self._prev_close is not None else close

        # ---- log return + Shannon-entropy window ----
        if prev_close > 0:
            log_ret = float(np.log(close / prev_close))
            self._returns.append(log_ret)
            if len(self._returns) > self._w:
                self._returns.popleft()
        self._prev_close = close

        # ---- ATR (Wilder RMA seeded at first finite TR) ----
        tr = float(max(high - low, abs(high - prev_close), abs(low - prev_close)))
        if self._atr is None:
            self._atr = tr
        else:
            self._atr = self._alpha_atr * tr + (1.0 - self._alpha_atr) * self._atr
        atr = self._atr

        # ---- rolling normalized entropy over the active window ----
        if len(self._returns) < self._w:
            # Window not full: entropy undefined.  Matches the batch path where
            # the EMA holds and no bands/trend update happen yet.
            self._n += 1
            return self._emit(close=close, entropy=0.5, strength=0.5)

        arr = np.asarray(self._returns, dtype=np.float64)
        lo = float(arr.min())
        hi = float(arr.max())
        rng = hi - lo
        if rng <= 0:
            entropy = 0.5
        else:
            bin_idx = np.floor((arr - lo) / rng * (cfg.num_bins - 1)).astype(np.int64)
            np.clip(bin_idx, 0, cfg.num_bins - 1, out=bin_idx)
            counts = np.bincount(bin_idx, minlength=cfg.num_bins).astype(np.float64)
            probs = counts / self._w
            mask = probs > 0
            entropy = -float(np.sum(probs[mask] * np.log2(probs[mask])))
            max_entropy = np.log2(cfg.num_bins)
            entropy = entropy / max_entropy if max_entropy > 0 else 0.5

        trend_strength = 1.0 - entropy

        # ---- adaptive EMA ----
        adaptive_alpha = 2.0 / (self._w * (0.3 + entropy * 1.4) + 1.0)
        assert self._ema is not None
        self._ema = self._ema + adaptive_alpha * (close - self._ema)

        # ---- bands / trend ----
        fast_width = atr * cfg.fast_multiplier * (0.5 + trend_strength)
        inner_upper = float(self._ema) + fast_width
        inner_lower = float(self._ema) - fast_width

        if close > inner_upper:
            self._trend = 1
        elif close < inner_lower:
            self._trend = -1
        # else: hold previous trend (bar inside the bands)

        self._n += 1
        return self._emit(
            close=close, entropy=entropy, strength=trend_strength, atr=atr
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _emit(
        self,
        close: float,
        entropy: float,
        strength: float,
        atr: float | None = None,
    ) -> AdaptiveEntropyResult:
        """Package the current running state into a snapshot."""
        ema = self._ema if self._ema is not None else 0.0
        atr_value = atr if atr is not None else (self._atr or 0.0)
        fast_width = atr_value * self._cfg.fast_multiplier * (0.5 + strength)
        slow_width = atr_value * self._cfg.slow_multiplier * (0.5 + strength)
        return AdaptiveEntropyResult(
            close=close,
            entropy=entropy,
            normalized_entropy=entropy,
            trend_strength=strength,
            adaptive_ema=ema,
            atr=atr_value,
            fast_band_width=fast_width,
            slow_band_width=slow_width,
            inner_upper=ema + fast_width,
            inner_lower=ema - fast_width,
            outer_upper=ema + slow_width,
            outer_lower=ema - slow_width,
            trend=self._trend,
        )
