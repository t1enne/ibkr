"""Screen driver — score a universe by running its real strategy through the
engine and capturing fresh current-bar intent.

Replaces the bespoke stateless screen-scoring layer. A screen does NOT trade:
it runs the strategy's own ``on_candle`` over the configured feed via the real
``Backtest.run``, but plumbs a signal observer into the engine so every fresh
``TradeSignal`` the strategy emits is captured *before* ``_finalize`` flattens
the book (that flatten — plus close-at-final-bar deferral — is what discards
the manual-trade signal a stock backtest swallows). The captured feed is pure
posture/intent, never real fills.

The engine-side DSL wiring (cursor-safe ``TaContext`` for decorated strategies
plus a fresh per-run state holder for stateful ones) lives in ``Backtest.run``;
this driver only hands it an observer so intent survives ``_finalize``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd

from src.bt import Backtest, init_strat, load_strategy, run
from src.bt.data_feed import load_candles
from src.bt.state import ActionType, BacktestState, TradeSignal
from src.bt.types import StrategyConfig

#: Common metrics shown as extra table columns, uniform with the old screen.
COMMON_COLS = ["ema_50", "ema_100", "atr_14", "rsi_14", "hi_52w", "lo_52w"]

Action = Literal["long", "short", "flat"]


@dataclass(frozen=True)
class ScreenRow:
    symbol: str
    action: Action
    score: float  # >0 iff actionable; 1.0 fresh open, <1.0 retained
    signals: tuple[str, ...]  # human reasons from TradeSignal.reason strings
    ts: pd.Timestamp


@dataclass(frozen=True)
class Posture:
    """Scoring knobs for deriving a per-symbol row from its observer feed."""

    base_score_open: float = 1.0  # fresh open signal on the newest bar
    base_score_held: float = 0.8  # prior intent, none newer this window
    include_flat: bool = True  # emit explicit flat rows for the universe


class SignalCollector:
    """Collect observed signals; present each symbol's final posture.

    ``on_signal`` is the engine observer target. Signals are kept grouped per
    symbol in emission (chronological) order so ``_project`` replays each
    symbol's intent ledger latest-wins without the driver touching the engine.
    """

    def __init__(self, symbols: tuple[str, ...]) -> None:
        self._feed: dict[str, tuple[TradeSignal, ...]] = {s: () for s in symbols}

    def on_signal(self, sig: TradeSignal) -> None:
        """Observer target — record one freshly-generated signal."""
        self._feed[sig.symbol] = self._feed.get(sig.symbol, ()) + (sig,)


def run_screen_from_strategy(
    config_path: str,
    posture: Posture = Posture(),
) -> tuple[tuple[ScreenRow, ...], BacktestState]:
    """Score a universe by running its strategy through the real engine.

    Loads the config exactly as ``bt run`` does, drives ``Backtest.run`` with a
    ``SignalCollector`` observer, and projects each symbol's captured intent
    into ranked ``ScreenRow``s.

    Returns ``(rows, final_state)``: rows carry the per-symbol posture, and
    ``final_state`` is the engine's post-``_finalize`` state (book flattened).
    """
    config = load_strategy(config_path)
    bt = Backtest(config)
    # Feed equals what a stock backtest loads — train start -> test end so the
    # DSL's TaContext/indicators warm up over full history, scored over test.
    df = _load_feed(config)
    strat_mod = init_strat(config.strategy_type)

    collector = SignalCollector(tuple(config.symbols))
    results = run(
        bt,
        df,
        strat_mod=strat_mod,
        signal_observer=collector.on_signal,
    )
    rows = _project(collector, tuple(config.symbols), posture, _latest_ts(df))
    return rows, results.final_state


def _load_feed(config: StrategyConfig) -> pd.DataFrame:
    """Load the MultiIndex-column OHLCV frame the engine generator consumes."""
    bt = Backtest(config)
    return load_candles(
        config.symbols,
        bt.window.train_start,
        bt.window.test_end,
        config.bars[0],
    )


def _latest_ts(df: pd.DataFrame) -> pd.Timestamp:
    """Timestamp of the newest scored (decision) bar in the feed."""
    idx = df.index
    if len(idx):
        return cast(pd.Timestamp, pd.Timestamp(idx[-1]))
    return pd.Timestamp.now()


def _project(
    collector: SignalCollector,
    symbols: tuple[str, ...],
    posture: Posture,
    latest_ts: pd.Timestamp,
) -> tuple[ScreenRow, ...]:
    """Derive one ranked row per symbol from its collected intent feed."""
    rows: list[ScreenRow] = []
    for sym in symbols:
        action, score, reasons = _resolve_posture(
            collector._feed.get(sym, ()), posture, latest_ts
        )
        if action == "flat" and not posture.include_flat:
            continue
        rows.append(
            ScreenRow(
                symbol=sym,
                action=action,
                score=score,
                signals=reasons,
                ts=latest_ts,
            )
        )
    # Ranked by score desc (actionable first), then symbol — deterministic.
    rows.sort(key=lambda r: (-r.score, r.symbol))
    return tuple(rows)


def _resolve_posture(
    feed: tuple[TradeSignal, ...],
    posture: Posture,
    latest_ts: pd.Timestamp,
) -> tuple[Action, float, tuple[str, ...]]:
    """Replay a symbol's chronological intents -> (action, score, signals).

    Latest-wins over the run (posture, never fills): a fresh ``long``/``short``
    orients the symbol to that side; a ``close`` reverts to flat; a rebalance
    leaves the incumbent side. A side (re)established on the newest scored bar
    is a fresh open (``base_score_open``); a side decided earlier (nothing
    newer this window) is a retained setup (``base_score_held``). Flat -> 0.0.
    """
    action: Action | None = None
    reasons: tuple[str, ...] = ()
    decided_ts: pd.Timestamp | None = None  # ts of the signal that set ``action``

    for sig in feed:
        side = _side_of(sig)
        if side is None:  # rebalance keeps the incumbent side
            continue
        action = side
        reasons = _reasons(sig)
        decided_ts = sig.timestamp

    if action in (None, "flat"):
        return ("flat", 0.0, reasons)

    fresh = decided_ts is not None and decided_ts == latest_ts
    score = posture.base_score_open if fresh else posture.base_score_held
    return (action, score, reasons)


def _side_of(sig: TradeSignal) -> Action | None:
    """Posture a signal imposes: open side, flat on a close, None on rebalance."""
    if sig.action == ActionType.long:
        return "long"
    if sig.action == ActionType.short:
        return "short"
    if sig.action == ActionType.close:
        return "flat"
    return None  # rebalance


def _reasons(sig: TradeSignal) -> tuple[str, ...]:
    """Human reason strings from a signal (``reason`` may be scalar or list)."""
    r = sig.reason
    if r is None:
        return ()
    if isinstance(r, (list, tuple)):
        return tuple(str(x) for x in r)
    return (str(r),)


# ---------------------------------------------------------------------------
# common per-symbol metrics (re-homed from the deleted screen scoring layer);
# diagnostics/context for the printed table only, never scoring inputs
# ---------------------------------------------------------------------------

COMMON_METRIC_KEYS = tuple(COMMON_COLS)


def _ta_ema(closes: pd.Series, span: int) -> float:
    """Lazy ``ta.ema`` — avoid pulling ``src.indicators`` at bt bootstrap."""
    from src.indicators.ta import ema

    try:
        v = float(ema(closes, span).iloc[-1])
    except IndexError, ValueError:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _ta_atr(frame: pd.DataFrame) -> float:
    """ATR(14) over a frame's OHLC — closest to price diagonal."""
    closes = frame["close"]
    high = frame["high"] if "high" in frame.columns else closes
    low = frame["low"] if "low" in frame.columns else closes
    if len(high) < 1:
        return float("nan")
    from src.indicators.ta import atr

    try:
        v = float(atr(high, low, closes, window=14).iloc[-1])
    except IndexError, ValueError:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _ta_rsi(closes: pd.Series) -> float:
    """RSI(14)."""
    from src.indicators.ta import rsi

    try:
        v = float(rsi(closes, window=14).iloc[-1])
    except IndexError, ValueError:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def common_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Compute the common metric set for a symbol frame (flat dict, float values,
    ``nan`` on missing data). Calendar-aware 52-week high/low over the index."""
    closes = frame["close"] if "close" in frame.columns else None
    if closes is None or len(closes) == 0:
        return {k: float("nan") for k in COMMON_METRIC_KEYS}

    hi_52w = lo_52w = float("nan")
    if isinstance(closes.index, pd.DatetimeIndex) and len(closes) > 0:
        hi = closes.rolling("365D", min_periods=1).max()
        lo = closes.rolling("365D", min_periods=1).min()
        hi_52w = float(hi.iloc[-1]) if np.isfinite(hi.iloc[-1]) else float("nan")
        lo_52w = float(lo.iloc[-1]) if np.isfinite(lo.iloc[-1]) else float("nan")

    return {
        "ema_50": _ta_ema(closes, 50),
        "ema_100": _ta_ema(closes, 100),
        "atr_14": _ta_atr(frame),
        "rsi_14": _ta_rsi(closes),
        "hi_52w": hi_52w,
        "lo_52w": lo_52w,
    }
