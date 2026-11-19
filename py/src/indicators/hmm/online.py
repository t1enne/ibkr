"""Online (step-by-step) HMM regime detector for real-time / backtest use.

Since hmmlearn has no partial_fit, we simulate online behaviour:
- Refit the full HMM every retrain_interval bars on a rolling window (or exactly
  once, with ``fit_once``).
- Between refits, label the bar with the MAP state of the **accumulated prefix**
  (Viterbi carried forward), which is what the reference article does.
- The ``update()`` method returns the current regime label (0/1/2).

Usage (strategy-owned; via :class:`src.indicators.hmm.strategy.OnlineRegime`):

    hmm = MarketRegimeHMMOnline(retrain_interval=50, window_size=500)
    ...
    for each tick:
        regime = hmm.update(price)   # 0/1/2, held by the strategy itself
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
import warnings

from hmmlearn import hmm

from src.indicators.hmm.hmm import rank_states_by_vol

_LOG_2PI = float(np.log(2.0 * np.pi))


def _log_probs(p: np.ndarray) -> np.ndarray:
    """Natural log of a probability vector, floored so an exact zero is finite.

    ``hmmlearn`` leaves exact zeros in ``startprob_``/``transmat_`` (a state can
    be unreachable in a fitted model). A raw ``np.log`` would emit a
    RuntimeWarning and return ``-inf``; the floor is float64's smallest normal,
    so the state stays effectively unreachable (log ≈ -708) without poisoning
    the max-product recursion with NaN.
    """
    return np.log(np.maximum(p, np.finfo(np.float64).tiny))


class MarketRegimeHMMOnline:
    """Online HMM regime detector — periodic batch refit on rolling window.

    Maintains a rolling window of prices internally. Every retrain_interval
    bars (or once, if ``fit_once``), refits the full HMM on the window. Between
    refits the label is the last state of a Viterbi pass over the accumulated
    prefix — carried forward incrementally, so it costs O(n_regimes^2) per bar
    instead of re-decoding the prefix.

    Why the prefix and not the row: a one-row ``predict()`` drops the
    transition matrix entirely (it degenerates to ``argmax(log emission +
    log startprob)``), so the label has no regime persistence and disagrees
    with the reference article's ``predict(prefix)[-1]``.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        window_size: int = 500,
        vol_window: int = 20,
        momentum_window: int = 10,
        retrain_interval: int = 50,
        random_state: int = 42,
        bars_per_year: int = 252,
        fit_once: bool = False,
    ) -> None:
        self.n_regimes = n_regimes
        self.window_size = window_size
        self.vol_window = vol_window
        self.momentum_window = momentum_window
        self.retrain_interval = retrain_interval
        self.random_state = random_state
        self.bars_per_year = bars_per_year
        #: Freeze the model at its first fit (the reference's "fit once, never
        #: retrain"). OFF keeps the counter-driven rolling refit untouched.
        self.fit_once = fit_once

        # Rolling window of prices
        self._prices: deque[float] = deque(maxlen=window_size)

        # Fitted model (None until first refit)
        self._model: hmm.GaussianHMM | None = None
        self._state_to_regime: dict[int, int] = {}
        self._fitted: bool = False
        self._n_refits: int = 0

        # Step counter
        self._n: int = 0
        self._last_refit: int = -window_size  # force refit at bar 0

        # Running Viterbi (log-domain) score per state since the last fit; None
        # until the first labelled bar after it.
        self._delta: np.ndarray | None = None

        # Current regime
        self._current_regime: int = -1

        # Regime labels (filled after first fit)
        self.regime_labels: dict[int, str] = {
            0: "Low Vol",
            1: "Medium Vol",
            2: "High Vol",
        }

    # ------------------------------------------------------------------
    # read-only properties
    # ------------------------------------------------------------------

    @property
    def current_regime(self) -> int:
        """Current regime label (0=low vol, 1=med, 2=high). -1 if not yet fitted."""
        return self._current_regime

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def n_steps(self) -> int:
        return self._n

    @property
    def n_refits(self) -> int:
        """How many times the model has been fitted (0 or 1 with ``fit_once``)."""
        return self._n_refits

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def update(self, price: float) -> int:
        """Process one price observation, return current regime.

        Returns -1 until enough data for first fit.
        """
        self._prices.append(price)
        self._n += 1

        # Need enough data for features before anything
        needed = max(self.vol_window, self.momentum_window) + 1
        if len(self._prices) < needed:
            return -1

        if self._should_refit():
            self._refit()

        if not self._fitted:
            return -1

        raw = self._decode(self._latest_features())
        self._current_regime = self._state_to_regime.get(raw, raw)
        return self._current_regime

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _should_refit(self) -> bool:
        """Whether this bar triggers a fit.

        ``fit_once`` makes the freeze explicit instead of relying on the
        counter coincidence ``retrain_interval == window_size`` (with
        ``_last_refit`` seeded at ``-window_size``): the single fit lands the
        moment the window is full and never repeats. Without it a huge
        ``retrain_interval`` never fires at all and the model stays unfitted
        forever, so no regime is ever reported.
        """
        if len(self._prices) < self.window_size:
            return False
        if self.fit_once:
            return not self._fitted
        return self._n - self._last_refit >= self.retrain_interval

    def _to_series(self) -> pd.Series:
        return pd.Series(list(self._prices))

    def _compute_features(self) -> pd.DataFrame:
        """Build feature DataFrame from the rolling window."""
        prices = self._to_series()
        returns = np.log(prices / prices.shift(1))
        volatility = returns.rolling(self.vol_window).std() * np.sqrt(
            self.bars_per_year
        )
        momentum = returns.rolling(self.momentum_window).mean() * self.bars_per_year
        return pd.DataFrame(
            {
                "returns": returns,
                "volatility": volatility,
                "momentum": momentum,
            }
        ).dropna()

    def _latest_features(self) -> np.ndarray:
        """Compute feature vector for the latest bar only.

        Feature definitions must match ``_compute_features`` (what the HMM was
        trained on): log returns, not simple returns. Mixing log-return
        volatility/momentum during training with simple-return features at
        prediction shifts the distribution the classifier sees and degrades
        regime labels.
        """
        prices = self._to_series()
        if len(prices) < 2:
            return np.zeros(3)
        returns = np.log(prices / prices.shift(1))
        ret = float(returns.iloc[-1])
        vol = returns.rolling(self.vol_window).std().iloc[-1] * np.sqrt(
            self.bars_per_year
        )
        mom = returns.rolling(self.momentum_window).mean().iloc[-1] * self.bars_per_year
        return np.array(
            [ret, vol if not pd.isna(vol) else 0.0, mom if not pd.isna(mom) else 0.0]
        )

    def _refit(self) -> None:
        """Fit HMM on current rolling window and build vol-ranked remap."""
        import sys
        import io

        features = self._compute_features()
        if len(features) < 50:
            return

        self._model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="diag",
            random_state=self.random_state,
            n_iter=200,
            tol=1e-3,
        )

        # hmmlearn prints convergence warnings to stderr from Cython.
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model.fit(features.values)
        finally:
            sys.stderr = old_stderr

        # Build vol-ranked remap: 0=lowest return-var, N-1=highest. Anchored to
        # the fitted return-variance emission parameter (rank_states_by_vol),
        # deterministic across refits — not the noisy empirical mean of
        # predict() assignments, which could reorder and make current_vol
        # flip spuriously between refits.
        self._state_to_regime, _ = rank_states_by_vol(self._model, self.n_regimes)

        self._fitted = True
        self._n_refits += 1
        self._last_refit = self._n
        # A new fit invalidates the running Viterbi: the prefix restarts here,
        # because every per-state score was accumulated under the old model.
        self._delta = None

        # Update labels
        label_map = {0: "Low Vol", 1: "Medium Vol", 2: "High Vol"}
        self.regime_labels = {
            i: label_map.get(i, f"Regime {i}") for i in range(self.n_regimes)
        }

    def _log_emission(self, model: hmm.GaussianHMM, features: np.ndarray) -> np.ndarray:
        """Per-state ``log N(features | means_, covars_)`` for ``diag`` covariance.

        Written out rather than calling hmmlearn's private
        ``_compute_log_likelihood`` so the emission the state scores are built
        from is explicit and version-independent.
        """
        assert model.covars_ is not None
        assert model.means_ is not None
        tiny = np.finfo(np.float64).tiny
        covars = np.asarray(model.covars_, dtype=float)
        # hmmlearn stores `diag` covariance as an (n_components, n_features,
        # n_features) stack of diagonal matrices; a 2-D (n_components,
        # n_features) profile is accepted too, as in ``rank_states_by_vol``.
        var = np.diagonal(covars, axis1=1, axis2=2) if covars.ndim == 3 else covars
        var = np.maximum(var, tiny)
        maha = (((features - np.asarray(model.means_, dtype=float)) ** 2) / var).sum(
            axis=1
        )
        log_norm = -0.5 * (features.shape[0] * _LOG_2PI + np.log(var).sum(axis=1))
        out = log_norm - 0.5 * maha
        assert out.shape == (model.n_components,), "one score per state"
        return out

    def _decode(self, features: np.ndarray) -> int:
        """MAP state of the accumulated prefix (Viterbi, carried forward).

        The reference runs ``predict()`` over every bar from the start of the
        trading window to bar t and takes ``[-1]``. Carrying Viterbi's
        max-product recursion forward one row at a time returns exactly that
        last column at O(n_regimes^2) per bar instead of O(prefix length); the
        prefix is never re-decoded. The recursion starts at the last fit, which
        is where the reference's prefix starts.
        """
        assert self._model is not None, "_decode requires a fitted model"
        logp = self._log_emission(self._model, features)
        if self._delta is None:
            self._delta = _log_probs(np.asarray(self._model.startprob_)) + logp
        else:
            trans = _log_probs(np.asarray(self._model.transmat_))
            assert trans.shape == (self.n_regimes, self.n_regimes)
            # max over the previous state: the same step a batch Viterbi takes.
            self._delta = (self._delta[:, None] + trans).max(axis=0) + logp
        return int(np.argmax(self._delta))
