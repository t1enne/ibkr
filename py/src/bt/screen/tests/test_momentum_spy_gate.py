"""Tests for the self-contained momentum_spy_gate screen (SPY AE-gate momentum).

The module absorbed the former ``momentum`` screen (removed with the no-alpha
divergent family), so these tests target it as the sole momentum home:
registration/discovery, the validated-pass default parameterization, param
resolution, and well-formed/stable scoring on synthetic frames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.bt.screen.screens import init_screen, resolve_screen_params
from src.bt.screen.screens import momentum_spy_gate
from src.bt.screen.screens.momentum_spy_gate import Params as SpyParams
from src.bt.screen.types import ScreenResult, ScreenState


def _ts(v: str) -> pd.Timestamp:
    ts = pd.Timestamp(v)
    assert isinstance(ts, pd.Timestamp)
    return ts


def _uprun_df(n: int = 200) -> pd.DataFrame:
    """Steady 60%-gain uptrend daily frame — long enough to pass warmup.

    Won't necessarily form a valid compression coil, but exercises the full
    scoring path (coil detection → box replay → setup/flat scoring) over many
    bars and must not raise.
    """
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    closes = 100.0 * (1.0 + np.linspace(0, 0.6, n))
    opens = np.concatenate([[100.0], closes[:-1]])
    high = np.maximum(opens, closes) * 1.004
    low = np.minimum(opens, closes) * 0.996
    return pd.DataFrame(
        {
            "open": opens,
            "high": high,
            "low": low,
            "close": closes,
            "volume": 1_000_000.0,
        },
        index=idx,
    )


def _params() -> SpyParams:
    """Resolve the screen's own typed Params (the resolver returns a broad
    union; the target type is SpyParams which carries all the screen knobs)."""
    resolved = resolve_screen_params("momentum_spy_gate", {})
    assert isinstance(resolved, SpyParams)
    return resolved


def _state(symbols: tuple[str, ...] = ("ABC",)) -> ScreenState:
    """ScreenState over synthetic frames, no benchmark embedded (un-gated)."""
    frames = tuple((s, _uprun_df()) for s in symbols)
    return ScreenState(
        ts=frames[0][1]["close"].index[-1],
        frames=frames,
        trend={s: "BULL" for s in symbols},
        vol={s: "MED_VOL" for s in symbols},
    )


def test_discovery_registers_spy_gate_screen():
    mod = init_screen("momentum_spy_gate")
    assert mod.SCREEN_TYPE == "momentum_spy_gate"
    assert callable(mod.on_state)
    assert getattr(mod, "Params", None) is not None


def test_resolve_params_uses_spy_gate_pass_defaults():
    p = resolve_screen_params("momentum_spy_gate", {})
    assert isinstance(p, SpyParams)
    assert p.benchmark == "SPY"
    assert p.min_gain == 0.35
    assert p.body_atr_ratio == 0.30
    assert p.min_hover_bars == 4
    assert p.regime_min_strength == 0.12
    assert p.ae_lookback == 40
    # Regime observer alias reflects the SPY benchmark.
    assert p.regime_symbol == "SPY"


def test_resolve_params_accepts_call_overrides():
    p = resolve_screen_params("momentum_spy_gate", {"min_gain": 0.5, "bogus": 1})
    assert isinstance(p, SpyParams)
    assert p.min_gain == 0.5
    assert p.benchmark == "SPY"  # untouched default retained


def test_resolve_params_ignores_backtest_trading_knobs():
    # The validated pass dict carries sizing/cooldown fields the screen never
    # uses; resolve_screen_params must load it cleanly (extras ignored) while
    # keeping the gate on SPY.
    pass_params = {
        "big_lookback": 63,
        "min_gain": 0.35,
        "max_gain": 1.0,
        "ma_fast": 10,
        "ma_slow": 20,
        "comp_window": 15,
        "body_atr_ratio": 0.3,
        "vol_period": 20,
        "vol_mult": 0.8,
        "min_hover_bars": 4,
        "hover_tol": 0.002,
        "decay_bars": 10,
        "atr_period": 14,
        "atr_mult": 1.5,  # backtest-only
        "risk_pct": 0.02,  # backtest-only
        "warmup_bars": 80,
        "cooldown_bars": 5,  # backtest-only
        "regime_symbol": "SPY",
        "regime_trend_min": 1,
        "regime_min_strength": 0.12,
        "ae_lookback": 40,
        "ae_num_bins": 10,
        "max_positions": 10,  # backtest-only
        "max_position_notional": 0.25,  # backtest-only
    }
    p = resolve_screen_params("momentum_spy_gate", pass_params)
    assert isinstance(p, SpyParams)
    # Every screen-relevant knob matched, trading-only extras dropped.
    assert p.big_lookback == 63
    assert p.min_gain == 0.35
    assert p.max_gain == 1.0
    assert p.ma_fast == 10
    assert p.ma_slow == 20
    assert p.comp_window == 15
    assert p.body_atr_ratio == 0.30
    assert p.vol_period == 20
    assert p.vol_mult == 0.8
    assert p.min_hover_bars == 4
    assert p.hover_tol == 0.002
    assert p.decay_bars == 10
    assert p.atr_period == 14
    assert p.warmup_bars == 80
    assert p.regime_trend_min == 1
    assert p.regime_min_strength == 0.12
    assert p.ae_lookback == 40
    assert p.ae_num_bins == 10
    assert p.benchmark == "SPY"  # regime_symbol is a derived alias, not a field
    # No unknown-field crash and no backtest-only leakage onto the dataclass.
    assert not hasattr(p, "risk_pct")
    assert not hasattr(p, "max_positions")


def test_on_state_returns_one_well_formed_result_per_symbol():
    state = _state(("ABC", "XYZ"))
    results = momentum_spy_gate.on_state(state, _params())
    assert len(results) == 2
    for r in results:
        assert isinstance(r, ScreenResult)
        assert r.symbol in ("ABC", "XYZ")
        assert r.timestamp == state.ts
        assert 0.0 <= r.score <= 1.0
        assert r.action in ("long", "flat")
        assert r.signals
        assert isinstance(r.model_features, dict)


def test_on_state_is_deterministic():
    state = _state()
    params = _params()
    assert momentum_spy_gate.on_state(state, params) == momentum_spy_gate.on_state(
        state, params
    )


def test_observer_symbol_never_scored():
    # SPY embedded as the regime observer must not be surfaced as a candidate.
    spy_df = _uprun_df()
    sym_df = _uprun_df()
    frames = (("SPY", spy_df), ("ABC", sym_df))
    state = ScreenState(
        ts=sym_df["close"].index[-1],
        frames=frames,
        trend={"SPY": "BULL", "ABC": "BULL"},
        vol={"SPY": "MED_VOL", "ABC": "MED_VOL"},
    )
    results = momentum_spy_gate.on_state(state, _params())
    syms = {r.symbol for r in results}
    assert "ABC" in syms
    assert "SPY" not in syms


def test_absorbs_momentum_regime_label_helpers():
    # runner.build_state imports these from this module — they must exist.
    assert callable(momentum_spy_gate._trend_label)
    assert callable(momentum_spy_gate._vol_label)
    closes = pd.Series(np.linspace(100.0, 200.0, 250))
    assert momentum_spy_gate._trend_label(closes, 50, 200, 0.005) == "BULL"
