"""Signals bridge — project a strategy-as-screen run into actionable ``LiveSignal``s.

No separate signal vocabulary: the live layer consumes the same driver
``ibkr bt screen`` uses (``run_screen_from_strategy``) and re-shapes each row
into a ``LiveSignal``.

**Close reconstruction.** The screen vocabulary is ``long|short|flat`` — there
is no ``"close"`` row. A close is reconstructed from a ``flat`` row that carries
a ``sig_ts``: ``_resolve_posture`` sets ``flat`` ONLY from a close and stamps
``sig_ts``/``signals`` from that close, so a ``flat`` row with a ``sig_ts`` IS a
close. A ``flat`` row with no ``sig_ts`` was never signalled -> HOLD (drop). A
``close`` intent is idempotent downstream: reconciling a flat target against a
flat book is a no-op.

**Why the driver's ``max_age_days`` is not used.** Its filter DROPS every flat
row, which deletes exactly the rows a close is derived from. We call the driver
with ``max_age_days=None`` (all rows) and apply the age filter locally, on the
``long``/``short``/``close`` actions, so a stale close is still filtered.
"""

from __future__ import annotations

import math

import pandas as pd

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

    Calls ``run_screen_from_strategy`` with ``max_age_days=None`` (all rows,
    including flat) and maps every actionable row to a ``LiveSignal`` priced at
    the symbol's last close of the base interval. ``long``/``short`` rows pass
    through; a ``flat`` row with a ``sig_ts`` becomes a ``close`` (see module
    docstring). The local age filter drops a stale posture. A row whose
    base-interval frame is missing/empty or whose last close is non-finite or
    ``<= 0`` is dropped. Pure mapping; the only I/O is the screen run.
    """
    screen = run_screen_from_strategy(config_path, max_age_days=None)
    rows, state = screen.rows, screen.state
    base_iv = load_strategy(config_path).bars[0]
    out: list[LiveSignal] = []
    for row in rows:
        if not _is_fresh(row, max_age_days):
            continue
        action = _action_of(row)
        if action is None or action not in ACTIONABLE:
            continue
        price = _ref_price(state, row.symbol, base_iv)
        if price is None:
            continue
        out.append(_to_live_signal(row, action, price))
    return tuple(out)


def _action_of(row: ScreenRow) -> SignalAction | None:
    """Live action for a row, or ``None`` to HOLD.

    ``long``/``short`` pass through, and the driver's explicit ``close`` action
    (a ``ctx.close`` signal, ``_side_of`` in the screen driver) maps to ``close``
    too. The older shape survives: a ``flat`` row with a ``sig_ts`` came from a
    close -> ``close``. A ``flat`` row with no ``sig_ts`` never signalled ->
    ``None`` (HOLD, drop).
    """
    if row.action in ("long", "short", "close"):
        return row.action
    if row.action == "flat" and row.sig_ts is not None:
        return "close"
    return None


def _is_fresh(row: ScreenRow, max_age_days: int | None) -> bool:
    """Local posture-age filter (the driver's is unusable — it drops flat rows).

    Drop when ``max_age_days > 0`` and the row's posture-setting bar trails its
    own data bar by more than ``max_age_days``. ``None`` or ``<= 0`` disables.
    """
    if max_age_days is None or max_age_days <= 0 or row.sig_ts is None:
        return True
    return (row.ts - row.sig_ts) <= pd.Timedelta(days=max_age_days)


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


def _to_live_signal(row: ScreenRow, action: SignalAction, price: float) -> LiveSignal:
    """Map an actionable ``ScreenRow`` + resolved action + ref price to a signal."""
    return LiveSignal(
        symbol=row.symbol,
        action=action,
        score=row.score,
        reasons=row.signals,
        signal_ts=row.sig_ts,
        price=price,
        qty=0.0,
    )
