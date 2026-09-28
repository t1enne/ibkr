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
from src.bt.warmup import parse_warmup_bars

from dataclasses import dataclass, replace
from typing import Literal, Mapping, cast

import numpy as np
import pandas as pd

from src.bt import Backtest, init_strat, load_strategy, run
from src.bt.data_feed import load_candles
from src.bt.state import ActionType, BacktestState, TradeSignal
from src.bt.types import StrategyConfig

#: Common metrics shown as extra table columns, uniform with the old screen.
#: ``rsi_14`` was replaced by ``mfi_14`` (money-flow index — blends volume into
#: its value), and the two raw volume stats were collapsed into one ``obv_z``:
#: the direct cumulative-flow channel a trader can actually weight (sign =
#: flow direction, magnitude = strength vs the name's own recent pattern).
COMMON_COLS = [
    "ema_50",
    "ema_100",
    "atr_14",
    "mfi_14",
    "obv_z",
    "hi_52w",
    "lo_52w",
]

Action = Literal["long", "short", "flat"]


@dataclass(frozen=True)
class ScreenRow:
    symbol: str
    action: Action
    score: float  # >0 iff actionable; 1.0 fresh open, <1.0 retained
    signals: tuple[str, ...]  # human reasons from TradeSignal.reason strings
    ts: pd.Timestamp  # newest loaded data bar for this symbol
    sig_ts: pd.Timestamp | None = None  # bar on which the live posture was set


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
    max_age_days: int | None = None,
) -> tuple[tuple[ScreenRow, ...], BacktestState]:
    """Score a universe by running its strategy through the real engine.

    Runs over a **trailing warm-up window ending at the newest available data**
    (not the config's multi-year backtest span — a screen only needs enough
    history to compute the latest-bar decision). ``warmup_days`` bounds the
    lookback; the decision bar is always the last bar present in the feed, so
    the reported posture is current tape, never the config's (possibly stale)
    ``trading_end``.

    Returns ``(rows, final_state)``: rows carry the per-symbol posture, and
    ``final_state`` is the engine's post-``_finalize`` state (book flattened).
    """
    cfg = load_strategy(config_path)
    data_end = _data_end(cfg)
    warmup_days = parse_warmup_bars(cfg.warmup, "1d")
    warm_start = cast(pd.Timestamp, data_end - pd.Timedelta(days=warmup_days))

    # A screen does not trade and has no IS/OOS split: it runs over a trailing
    # warm-up window ending at the newest data bar and treats the WHOLE tail as
    # the trading window, so signals fire through the final bar (whose timestamp
    # is data_end) and fresh-vs-held attribution stays correct. ``warmup`` is 0:
    # the loaded tail already IS the span the strategy sees, so there is no
    # second warmup collapse — every loaded bar is tradable tape.
    config = replace(
        cfg,
        warmup="0d",
        trading_start=_iso(warm_start),
        trading_end=_iso(data_end),
    )
    bt = Backtest(config)
    df = _load_feed(config)
    strat_mod = init_strat(config.strategy_type)

    collector = SignalCollector(tuple(config.symbols))
    results = run(
        bt,
        df,
        strat_mod=strat_mod,
        signal_observer=collector.on_signal,
    )
    final = results.final_state
    # Freshness bar is each symbol's OWN last loaded bar (stale/delisted names
    # end early; a shared feed-last would misrank shorter-calendar winners).
    latest = _last_bar_by_symbol(final, tuple(config.symbols), config.bars[0]) or {
        s: _latest_ts(df) for s in config.symbols
    }
    rows = _project(collector, tuple(config.symbols), posture, latest)
    rows = _filter_recent(rows, max_age_days)
    return rows, final


def _filter_recent(
    rows: tuple[ScreenRow, ...],
    max_age_days: int | None,
) -> tuple[ScreenRow, ...]:
    """Drop flat rows and postures whose setting bar is older than ``max_age_days``.

    Age is measured against each row's OWN latest data bar (``r.ts``), never
    wall-clock, so a stale/delisted symbol's fresh intent is not misjudged
    against today. ``None`` disables the filter (current behaviour).
    """
    if max_age_days is None:
        return rows
    keep: list[ScreenRow] = []
    for r in rows:
        if r.action == "flat" or r.sig_ts is None:
            continue
        if (r.ts - r.sig_ts) <= pd.Timedelta(days=max_age_days):
            keep.append(r)
    return tuple(keep)


def _load_feed(config: StrategyConfig) -> pd.DataFrame:
    """Load the MultiIndex-column OHLCV frame the engine generator consumes."""
    from src.bt import warmup_load_start

    bt = Backtest(config)
    return load_candles(
        config.symbols,
        warmup_load_start(config, bt.window.test_start),
        bt.window.test_end,
        config.bars[0],
    )


def _screen_symbols(config: StrategyConfig) -> tuple[str, ...]:
    """Traded universe plus benchmark references that must also be loaded."""
    bms = list(getattr(config, "benchmark_symbols", None) or [])
    return tuple(dict.fromkeys(list(config.symbols) + bms))


def _data_end(config: StrategyConfig) -> pd.Timestamp:
    """Newest bar timestamp available in the local DB for this config's universe.

    Probes the candle table directly (one aggregate query, no OHLCV materialised)
    so the screen anchors to *current tape* regardless of a stale ``trading_end``
    in the config. Falls back to ``trading_end`` if the universe has no rows
    (never crashes the command on an empty probe). Returns a **naive** local
    timestamp (the clock the engine/DB share).
    """
    from src.data.db import get_connection  # lazy: avoid pkg-init cycle
    from src.utils import parse_timestamp

    symbols = _screen_symbols(config)
    con = get_connection()
    try:
        ph = ",".join("?" * len(symbols))
        row = con.execute(
            f"SELECT MAX(timestamp) FROM candle WHERE ticker IN ({ph})", symbols
        ).fetchone()
    finally:
        con.close()
    ms = row[0] if row else None
    if ms is None:
        return parse_timestamp(config.trading_end)
    # Guarded non-null above; ms is an epoch-millis int (candle.timestamp is ms).
    return cast(pd.Timestamp, pd.Timestamp(int(ms), unit="ms"))


def _iso(ts: pd.Timestamp) -> str:
    """Encode a Timestamp as the naive-ISO string ``StrategyConfig`` parses back."""
    return str(ts)


def _latest_ts(df: pd.DataFrame) -> pd.Timestamp:
    """Timestamp of the newest scored (decision) bar in the feed."""
    idx = df.index
    if len(idx):
        return cast(pd.Timestamp, pd.Timestamp(idx[-1]))
    return pd.Timestamp.now()


#: Signal-freshness bar: per symbol, or a single shared one (simple tests).
LatestTs = pd.Timestamp | Mapping[str, pd.Timestamp]


def _last_bar_by_symbol(
    state: BacktestState,
    symbols: tuple[str, ...],
    base_iv: str,
) -> dict[str, pd.Timestamp]:
    """Each symbol's own newest loaded bar (its signal-freshness bar).

    Per-symbol, never global: after re-anchoring to current data (decision B)
    a universe's listings end on *different* days (a stale/delisted name can
    stop months early). Judging freshness off one shared feed-last timestamp
    would mislabel every same-day winner on a shorter calendar as ``held``, so
    each symbol's own last bar is its true decision bar.
    """
    out: dict[str, pd.Timestamp] = {}
    for sym in symbols:
        frame = state.candles.get((sym, base_iv))
        if frame is not None and len(frame):
            out[sym] = cast(pd.Timestamp, pd.Timestamp(frame.index[-1]))
    return out


def _resolve_latest(latest: LatestTs, sym: str) -> pd.Timestamp:
    """Per-symbol freshness bar, with a shared fallback when given one ts."""
    if isinstance(latest, pd.Timestamp):
        return latest
    ts = latest.get(sym)
    if ts is not None:
        return ts
    return max(latest.values(), default=pd.Timestamp.now())


def _project(
    collector: SignalCollector,
    symbols: tuple[str, ...],
    posture: Posture,
    latest_ts: LatestTs,
) -> tuple[ScreenRow, ...]:
    """Derive one ranked row per symbol from its collected intent feed.

    ``latest_ts`` is a per-symbol freshness bar (preferred — listings end on
    different days) or a single shared timestamp (kept for simple callers).
    Each row is timestamped by its own symbol's latest bar.
    """
    rows: list[ScreenRow] = []
    for sym in symbols:
        own = _resolve_latest(latest_ts, sym)
        action, score, reasons, sig_ts = _resolve_posture(
            collector._feed.get(sym, ()), posture, own
        )
        if action == "flat" and not posture.include_flat:
            continue
        rows.append(
            ScreenRow(
                symbol=sym,
                action=action,
                score=score,
                signals=reasons,
                ts=own,
                sig_ts=sig_ts,
            )
        )
    # Ranked by score desc (actionable first), then symbol — deterministic.
    rows.sort(key=lambda r: (-r.score, r.symbol))
    return tuple(rows)


def _resolve_posture(
    feed: tuple[TradeSignal, ...],
    posture: Posture,
    latest_ts: pd.Timestamp,
) -> tuple[Action, float, tuple[str, ...], pd.Timestamp | None]:
    """Replay a symbol's chronological intents -> (action, score, signals, sig_ts).

    Latest-wins over the run (posture, never fills): a fresh ``long``/``short``
    orients the symbol to that side; a ``close`` reverts to flat; a rebalance
    leaves the incumbent side. A side (re)established on the newest scored bar
    is a fresh open (``base_score_open``); a side decided earlier (nothing
    newer this window) is a retained setup (``base_score_held``). Flat -> 0.0.
    ``sig_ts`` is the bar that set the live posture; a 0.8-retained row whose
    ``sig_ts`` trails ``latest_ts`` by weeks is a stale setup, not a fresh
    signal — the reason string is emission-time, not current tape.
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
        return ("flat", 0.0, reasons, decided_ts)

    fresh = decided_ts is not None and decided_ts == latest_ts
    score = posture.base_score_open if fresh else posture.base_score_held
    return (action, score, reasons, decided_ts)


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


def _ta_mfi(frame: pd.DataFrame) -> float:
    """MFI(14) — money-flow index; volume-weighted, distinct from the plain
    volume reads below."""

    def _col(name: str) -> pd.Series:
        return frame[name] if name in frame.columns else frame["close"]

    from src.indicators.ta import mfi

    try:
        v = float(
            mfi(_col("high"), _col("low"), _col("close"), _col("volume")).iloc[-1]
        )
    except IndexError, ValueError:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _obv_z(frame: pd.DataFrame) -> float:
    """Cumulative-flow channel, single column. OBV over the frame, then its
    current level expressed as a z-score against its own trailing-40 pattern
    (rolling mean/std of the OBV level). Sign = money-flow direction; |value|
    = strength off the name's own noise floor (~+1.5/-1.5 starts to be
    notable). A long printing while this sits near/below zero is unconfirmed
    flow — the price/OBV divergence tell. Nan without enough bars to warm the
    OBV z-window."""
    if "volume" not in frame.columns or len(frame) < 41:
        return float("nan")
    from src.indicators.ta import obv

    try:
        obv_series = obv(frame["close"], frame["volume"])
    except IndexError, ValueError:
        return float("nan")
    if len(obv_series) < 41:
        return float("nan")
    win = obv_series.rolling(window=40)
    m = float(win.mean().iloc[-1])
    s = float(win.std().iloc[-1])
    if not np.isfinite(m) or not np.isfinite(s) or s == 0:
        return float("nan")
    last = float(obv_series.iloc[-1])
    return (last - m) / s if np.isfinite(last) else float("nan")


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
        "mfi_14": _ta_mfi(frame),
        "obv_z": _obv_z(frame),
        "hi_52w": hi_52w,
        "lo_52w": lo_52w,
    }
