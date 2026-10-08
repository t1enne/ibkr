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
from src.bt.warmup import parse_warmup

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Mapping, TypedDict, cast

if TYPE_CHECKING:
    from src.bt.state.types import Trade

import pandas as pd

from src.bt import Backtest, init_strat, load_strategy, run
from src.bt.data_feed import load_candles
from src.bt.state import ActionType, BacktestState, TradeSignal
from src.bt.types import StrategyConfig

#: ``long``/``short`` open or reorient a side, ``close`` is an explicit exit
#: directive, ``flat`` means no signal / no position (context only, never an
#: order — a live consumer must not read it as "flatten").
Action = Literal["long", "short", "close", "flat"]


@dataclass(frozen=True)
class ScreenRow:
    symbol: str
    action: Action
    score: float  # >0 iff actionable; 1.0 fresh, <1.0 retained
    signals: tuple[str, ...]  # human reasons from TradeSignal.reason strings
    ts: pd.Timestamp  # newest loaded data bar for this symbol
    sig_ts: pd.Timestamp | None = None  # bar on which the live posture was set
    # Executable fields lifted from the posture-setting TradeSignal so a live
    # consumer can act on the row without replaying the engine (see
    # ``render_screen_json``). Defaults keep non-signal (flat) rows inert.
    price: float = 0.0
    qty: float = 0.0  # absolute share count (0.0 = engine-sized)
    stop_loss: float | None = None
    take_profit: float | None = None
    position_id: str | None = None
    tag: str = ""


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


@dataclass(frozen=True)
class ResolvedPosture:
    """One symbol's replayed intent: action + score + the setting signal."""

    action: Action
    score: float
    reasons: tuple[str, ...]
    sig_ts: pd.Timestamp | None
    signal: TradeSignal | None  # the latest posture-setting signal (None = flat)


@dataclass(frozen=True)
class ScreenRun:
    """Full screen run result: ranked rows, post-finalize state, source config."""

    rows: tuple[ScreenRow, ...]
    state: BacktestState
    config: StrategyConfig
    # The run's executed trades (open -> close). A screen runs the real engine,
    # so its own ``on_candle`` fills are recorded here — INCLUDING the fills
    # ``_finalize`` uses to flatten the book at the final bar, which are not
    # strategy intent. Empty by default so row-only callers/tests stay inert.
    trades: tuple[Trade, ...] = ()


def run_screen_from_strategy(
    config_path: str,
    posture: Posture = Posture(),
    max_age_days: int | None = None,
) -> ScreenRun:
    """Score a universe by running its strategy through the real engine.

    Runs over a **trailing warm-up window ending at the newest available data**
    (not the config's multi-year backtest span — a screen only needs enough
    history to compute the latest-bar decision). ``warmup_days`` bounds the
    lookback; the decision bar is always the last bar present in the feed, so
    the reported posture is current tape, never the config's (possibly stale)
    ``trading_end``.

    Returns a ``ScreenRun``: ranked rows carry the per-symbol posture plus the
    executable fields of its setting signal, ``state`` is the engine's
    post-``_finalize`` state (book flattened), and ``config`` is the resolved
    source config.
    """
    cfg = load_strategy(config_path)
    data_end = _data_end(cfg)
    # Calendar span, not bar count: the trailing window is a real date range on
    # the constant ``1d`` grid, so ``parse_warmup``'s ``days`` is the unit for
    # date subtraction (``bars`` would be 5/7 of it and under-load history).
    warmup_span = parse_warmup(cfg.warmup, "1d")
    warm_start = cast(pd.Timestamp, data_end - pd.Timedelta(days=warmup_span.days))

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
        # A screen's trailing window starts trading exactly at the first loaded
        # bar, so the tail-symbol clock-coverage assertion (a truncation guard
        # for full-history backtests) is a false positive here. Generate signals
        # through the final bar instead of tripping on the leading bar-grid snap.
        evaluation_clock_check=False,
    )
    final = results.final_state
    # Freshness bar is each symbol's OWN last loaded bar (stale/delisted names
    # end early; a shared feed-last would misrank shorter-calendar winners).
    latest = _last_bar_by_symbol(final, tuple(config.symbols), config.bars[0]) or {
        s: _latest_ts(df) for s in config.symbols
    }
    rows = _project(collector, tuple(config.symbols), posture, latest)
    rows = _filter_recent(rows, max_age_days)
    # The observer feed is pure intent; the engine's own book is the trade log.
    # Flat after ``_finalize`` (see module docstring) — carried as executed fills.
    return ScreenRun(
        rows=rows, state=final, config=config, trades=tuple(results.pf.trades)
    )


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
    from src.db.connection import get_connection  # leaf pkg: no import cycle
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
        rp = _resolve_posture(collector._feed.get(sym, ()), posture, own)
        if rp.action == "flat" and not posture.include_flat:
            continue
        rows.append(_row_from_posture(sym, rp, own))
    # Ranked by score desc (actionable first), then symbol — deterministic.
    rows.sort(key=lambda r: (-r.score, r.symbol))
    return tuple(rows)


def _row_from_posture(
    sym: str,
    rp: ResolvedPosture,
    own: pd.Timestamp,
) -> ScreenRow:
    """Lift a resolved posture into a row, carrying its signal's executable fields."""
    s = rp.signal
    return ScreenRow(
        symbol=sym,
        action=rp.action,
        score=rp.score,
        signals=rp.reasons,
        ts=own,
        sig_ts=rp.sig_ts,
        price=float(s.price) if s is not None else 0.0,
        qty=float(s.qty) if s is not None else 0.0,
        stop_loss=_opt_float(s.stop_loss) if s is not None else None,
        take_profit=_opt_float(s.take_profit) if s is not None else None,
        position_id=s.position_id if s is not None else None,
        tag=s.tag if s is not None else "",
    )


def _opt_float(v: float | None) -> float | None:
    """Pass an optional float through (``None`` stays ``None``)."""
    return None if v is None else float(v)


def _resolve_posture(
    feed: tuple[TradeSignal, ...],
    posture: Posture,
    latest_ts: pd.Timestamp,
) -> ResolvedPosture:
    """Replay a symbol's chronological intents -> its latest resolved posture.

    Latest-wins over the run (posture, never fills): a fresh ``long``/``short``
    orients the symbol to that side; a ``close`` is an explicit exit directive;
    a rebalance leaves the incumbent side. A side (re)established on the newest
    scored bar is a fresh action (``base_score_open``); a side decided earlier
    (nothing newer this window) is retained (``base_score_held``). No
    position-setting signal -> flat, score 0.0. ``sig_ts`` is the bar that set
    the live posture, and ``signal`` is the setting signal itself (for its
    executable fields); a 0.8-retained row whose ``sig_ts`` trails ``latest_ts``
    by weeks is a stale setup, not fresh — the reason string is emission-time.
    """
    action: Action | None = None
    setting: TradeSignal | None = None  # latest signal that set ``action``

    for sig in feed:
        side = _side_of(sig)
        if side is None:  # rebalance keeps the incumbent side
            continue
        action = side
        setting = sig

    if action is None or setting is None:
        return ResolvedPosture("flat", 0.0, (), None, None)

    fresh = setting.timestamp == latest_ts
    score = posture.base_score_open if fresh else posture.base_score_held
    return ResolvedPosture(action, score, _reasons(setting), setting.timestamp, setting)


def _side_of(sig: TradeSignal) -> Action | None:
    """Posture a signal imposes: open side, explicit close, None on rebalance."""
    if sig.action == ActionType.long:
        return "long"
    if sig.action == ActionType.short:
        return "short"
    if sig.action == ActionType.close:
        return "close"
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
# machine-readable intent (pluggable into a live trading system)
# ---------------------------------------------------------------------------

#: Actions a live consumer may execute. ``flat`` is deliberately excluded: it
#: means "no signal / no position", never "flatten the book".
ACTIONABLE: tuple[Action, ...] = ("long", "short", "close")


class ScreenSignalJson(TypedDict):
    """One actionable per-symbol intent as a JSON-ready dict."""

    symbol: str
    action: Action
    score: float
    reasons: list[str]
    signal_ts: str | None  # bar the intent fired on; None only for flat rows
    data_ts: str  # symbol's newest loaded bar (freshness anchor)
    price: float
    qty: float  # absolute shares (0.0 = engine-sized, live layer decides)
    stop_loss: float | None
    take_profit: float | None
    position_id: str | None
    tag: str


class ScreenJson(TypedDict):
    """Top-level screen payload — stable shape for live-trading consumption."""

    command: str
    strategy: str
    strategy_type: str
    bars: str
    generated_at: str
    signals: list[ScreenSignalJson]


def render_screen_json(
    run: ScreenRun,
    *,
    strategy: str,
    generated_at: pd.Timestamp | None = None,
) -> ScreenJson:
    """ScreenRun -> one JSON-ready payload of actionable per-symbol intent.

    Only ``long``/``short``/``close`` rows are emitted (see ``ACTIONABLE``):
    a live consumer must never mistake a no-signal ``flat`` row for an order
    to flatten. Rows are already ranked (fresh before retained) by ``_project``.
    """
    signals: list[ScreenSignalJson] = [
        _signal_json(r) for r in run.rows if r.action in ACTIONABLE
    ]
    return {
        "command": "screen",
        "strategy": strategy,
        "strategy_type": run.config.strategy_type,
        "bars": run.config.bars[0] if run.config.bars else "",
        "generated_at": str(
            generated_at if generated_at is not None else pd.Timestamp.now()
        ),
        "signals": signals,
    }


def trade_table_row(t: Trade) -> dict[str, str]:
    """One executed ``Trade`` -> its printed trade-table row (all strings).

    ``exit_time``/``exit_price``/``close_reason`` render blank when ``None``
    (an unclosed trade) rather than the literal ``"None"``; every other field
    is non-optional on ``Trade``. Kept pure so the formatting is unit-testable
    without a live run (empty trades, None exits).
    """
    return {
        "symbol": t.symbol,
        "position": str(getattr(t.position, "value", t.position)),
        "qty": f"{t.qty:.2f}",
        "entry_time": str(t.entry_time),
        "entry_price": f"{t.entry_price:.2f}",
        "exit_time": "" if t.exit_time is None else str(t.exit_time),
        "exit_price": "" if t.exit_price is None else f"{t.exit_price:.2f}",
        "pnl": f"{t.pnl:.2f}",
        "close_reason": _reason_str(t.close_reason),
    }


def _reason_str(v: object) -> str:
    """Close reason (enum or scalar or ``None``) -> printable string."""
    if v is None:
        return ""
    return str(getattr(v, "value", v))


def _signal_json(r: ScreenRow) -> ScreenSignalJson:
    """Lift one actionable row into its JSON representation."""
    return {
        "symbol": r.symbol,
        "action": r.action,
        "score": r.score,
        "reasons": list(r.signals),
        "signal_ts": str(r.sig_ts) if r.sig_ts is not None else None,
        "data_ts": str(r.ts),
        "price": r.price,
        "qty": r.qty,
        "stop_loss": r.stop_loss,
        "take_profit": r.take_profit,
        "position_id": r.position_id,
        "tag": r.tag,
    }
