"""Online HMM label must be the prefix-Viterbi state, not a single-row argmax.

Goal: ``MarketRegimeHMMOnline.update`` labels bar t with the MAP state of the
**accumulated prefix** — the reference article's
``predict(returns_from_the_start_through_t)[-1]``. Before the fix it called
``predict()`` on a single feature row, which degenerates to
``argmax(log emission + log startprob)``: the transition matrix and the whole
state history are discarded, so the label carries no regime persistence and can
disagree with the reference on the same bar.

Method: a synthetic two-regime price path (calm, then turbulent) is fed one bar
at a time with ``fit_once=True``, so the model is fitted exactly once — at the
bar where the window first fills — and frozen. The oracle is hmmlearn's own
batch ``predict()`` over the feature rows from that fit bar through bar t,
remapped through ``rank_states_by_vol`` so both sides speak the same labels. The
pre-fix answer (single-row argmax) is recomputed inline, and the test asserts the
two genuinely disagree on this path — otherwise the equality assertion would be
vacuous.

The second test pins ``fit_once``: one fit, then no more, and the pre-existing
trap (a huge ``retrain_interval`` never firing, so the model is never fitted) is
unchanged for the default path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.hmm.hmm import rank_states_by_vol
from src.indicators.hmm.online import MarketRegimeHMMOnline

VOL_WINDOW = 5
MOMENTUM_WINDOW = 3
WINDOW_SIZE = 120
#: momentum is annualised LINEARLY (*bars_per_year), volatility by sqrt — as in
#: ``MarketRegimeHMMOnline._compute_features``; the features must match exactly.
BARS_PER_YEAR = 252


def _prices(n_calm: int, n_turbulent: int, seed: int) -> np.ndarray:
    """Two well-separated return regimes: calm bars, then turbulent bars."""
    rng = np.random.default_rng(seed)
    rets = np.concatenate(
        [rng.normal(0.0, 0.004, n_calm), rng.normal(0.0, 0.04, n_turbulent)]
    )
    return 100.0 * np.exp(np.cumsum(rets))


def _features(prices: np.ndarray) -> np.ndarray:
    """Feature matrix exactly as the class builds it (returns, vol, momentum)."""
    closes = pd.Series(prices)
    returns = np.log(closes / closes.shift(1))
    vol = returns.rolling(VOL_WINDOW).std() * np.sqrt(BARS_PER_YEAR)
    mom = returns.rolling(MOMENTUM_WINDOW).mean() * BARS_PER_YEAR
    frame = pd.DataFrame(
        {"returns": returns, "volatility": vol, "momentum": mom}
    ).dropna()
    return frame.values


def _model() -> MarketRegimeHMMOnline:
    return MarketRegimeHMMOnline(
        n_regimes=2,
        window_size=WINDOW_SIZE,
        vol_window=VOL_WINDOW,
        momentum_window=MOMENTUM_WINDOW,
        # A huge retrain_interval would never fire on a rolling path; fit_once
        # is what makes the single frozen fit happen.
        retrain_interval=10**9,
        random_state=42,
        fit_once=True,
    )


def test_online_label_is_prefix_viterbi_not_single_row_argmax() -> None:
    prices = _prices(160, 40, seed=1)
    model = _model()
    labels = [model.update(float(p)) for p in prices]

    assert model.n_refits == 1, "fit_once must fit exactly once"
    assert model.fitted
    first_labelled = next(i for i, v in enumerate(labels) if v >= 0)

    features = _features(prices)
    # The fit bar's feature row is where the prefix starts (the reference starts
    # its prefix at the trading window's first bar, i.e. after the fit).
    fit_row = first_labelled - (len(prices) - len(features))
    raw_model = model._model
    assert raw_model is not None
    remap, _ = rank_states_by_vol(raw_model, 2)
    log_prior = np.log(
        np.maximum(np.asarray(raw_model.startprob_), np.finfo(float).tiny)
    )

    single_row_disagreements = 0
    for bar in range(first_labelled, len(prices)):
        row = bar - (len(prices) - len(features))
        prefix_raw = int(raw_model.predict(features[fit_row : row + 1])[-1])
        assert labels[bar] == remap[prefix_raw], (
            f"bar {bar}: online {labels[bar]} != prefix Viterbi {remap[prefix_raw]}"
        )
        old_raw = int(
            np.argmax(log_prior + model._log_emission(raw_model, features[row]))
        )
        if remap[old_raw] != labels[bar]:
            single_row_disagreements += 1

    assert single_row_disagreements > 0, (
        "this path does not exercise the fix: single-row argmax agrees with "
        "prefix Viterbi at every bar"
    )


def test_fit_once_freezes_the_model() -> None:
    prices = _prices(240, 60, seed=3)  # 300 bars, window fills at bar 120

    frozen = _model()
    for price in prices:
        frozen.update(float(price))
    assert frozen.n_refits == 1
    assert frozen.fitted
    assert frozen.current_regime >= 0

    # Default path unchanged: the counter coincidence still refits every
    # retrain_interval bars (two fits over 300 bars at interval == window).
    rolling = MarketRegimeHMMOnline(
        n_regimes=2,
        window_size=WINDOW_SIZE,
        vol_window=VOL_WINDOW,
        momentum_window=MOMENTUM_WINDOW,
        retrain_interval=WINDOW_SIZE,
        random_state=42,
    )
    for price in prices:
        rolling.update(float(price))
    assert rolling.n_refits == 2

    # ...and the pre-existing trap survives for the default path: a huge
    # retrain_interval never fires, so the model is never fitted at all.
    never = MarketRegimeHMMOnline(
        n_regimes=2,
        window_size=WINDOW_SIZE,
        vol_window=VOL_WINDOW,
        momentum_window=MOMENTUM_WINDOW,
        retrain_interval=10**9,
        random_state=42,
    )
    for price in prices:
        never.update(float(price))
    assert never.n_refits == 0
    assert not never.fitted
    assert never.current_regime == -1
