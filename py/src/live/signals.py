"""Signals bridge — project a strategy-as-screen run into actionable ``LiveSignal``s.

No separate signal vocabulary: the live layer consumes the same driver
``ibkr bt screen`` uses (``run_screen_from_strategy``) and re-shapes each
actionable ``ScreenRow`` into a ``LiveSignal``. ``flat`` rows are dropped —
absence of a signal means HOLD downstream, never flatten.
"""

from __future__ import annotations

import math
from typing import cast

from src.bt import load_strategy
from src.bt.screen import ScreenRow, run_screen_from_strategy
from src.bt.state import BacktestState
from src.live.types import LiveSignal, SignalAction

#: Actions that carry an order decision (never emitted as ``flat``).
ACTIONABLE: frozenset[SignalAction] = frozenset({"long", "short", "close"})


def live_signals(
    config_path: str,
    max_age_days: int | None = None,
) -> tuple[LiveSignal, ...]:
    """Run the strategy-as-screen and project its actionable rows.

    Calls ``run_screen_from_strategy`` (the same driver ``ibkr bt screen`` uses)
    and maps every ``long``/``short`` row to a ``LiveSignal`` priced at the
    symbol's last close of the base interval. ``flat`` rows are dropped, as is
    any row whose base-interval frame is missing/empty or whose last close is
    non-finite or ``<= 0``. Pure mapping; the only I/O is the screen run.
    """
    rows, state = run_screen_from_strategy(config_path, max_age_days=max_age_days)
    base_iv = load_strategy(config_path).bars[0]
    out: list[LiveSignal] = []
    for row in rows:
        if row.action == "flat":
            continue
        price = _ref_price(state, row.symbol, base_iv)
        if price is None:
            continue
        out.append(_to_live_signal(row, price))
    return tuple(out)


def _ref_price(state: BacktestState, symbol: str, base_iv: str) -> float | None:
    """Last close of *symbol*'s base-interval frame, or ``None`` when unusable.

    ``None`` (missing frame, empty frame, non-finite or non-positive close)
    means the row has no tradable reference price and must be dropped — never
    emit a ``LiveSignal`` with ``price <= 0``.
    """
    frame = state.candles.get((symbol, base_iv))
    if frame is None or len(frame) == 0:
        return None
    price = float(frame["close"].iloc[-1])
    if not math.isfinite(price) or price <= 0:
        return None
    return price


def _to_live_signal(row: ScreenRow, price: float) -> LiveSignal:
    """Map an actionable ``ScreenRow`` + ref price to a ``LiveSignal`` (unsized)."""
    return LiveSignal(
        symbol=row.symbol,
        action=cast("SignalAction", row.action),  # caller dropped the "flat" rows
        score=row.score,
        reasons=row.signals,
        signal_ts=row.sig_ts,
        price=price,
        qty=0.0,
    )
