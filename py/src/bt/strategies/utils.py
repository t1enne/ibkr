from typing import List, Optional, Any, Tuple
from collections.abc import Mapping

from src.bt.state import (
    TradeSignal,
    ActionType,
    Candle,
    Position,
)

from src.bt.size.pure import SizingParams, compute_qty


def close(
    tick: Candle,
    position: Position,
    reason: Any,
    z: Optional[float] = None,
) -> List[TradeSignal]:
    return [
        TradeSignal(
            action=ActionType.close,
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            price=tick.close,
            z_score=z,
            qty=abs(position.qty),
            position_id=position.position_id,
            reason=reason,
        )
    ]


def open(
    tick: Candle,
    dir: ActionType,
    qty: float,
    reason: Optional[str] = "",
    hedge: Optional[float] = None,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
) -> List[TradeSignal]:
    return [
        TradeSignal(
            action=dir,
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            price=tick.close,
            qty=qty,
            reason=reason,
            hedge_beta=hedge,
            stop_loss=stop_loss,
            take_profit=take_profit,
        ),
    ]


# ---------------------------------------------------------------------------
# shared sizing + SL/TP helpers — per-trade sizing is strategy-owned and
# driven by ``strategy_params`` (position_size / stop_loss / take_profit).
# ---------------------------------------------------------------------------


def sized_qty(cash: float, position_size: float, price: float) -> float:
    """Share quantity for ``position_size`` fraction of ``cash`` at ``price``.

    ``position_size`` is a 0-1 fraction of cash deployed per position/trade.
    Delegates to the shared sizing layer (cash-based). Risk-targeted sizing
    (``risk_pct`` + stop/ATR) is strategy-owned via :func:`size.risk_sized_qty`.
    """
    return compute_qty(
        equity=cash,
        cash=cash,
        price=price,
        params=SizingParams(sizing_mode="cash", size=position_size),
    )


def sl_tp_from_pct(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    is_long: bool,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute absolute per-trade SL/TP prices from percentage params.

    ``stop_loss`` / ``take_profit`` are fractional pcts (e.g. 0.05 = 5%). A
    pct <= 0 disables that leg (returns None so the position has no SL/TP).
    Returns ``(stop_loss_price, take_profit_price)``.
    """
    if is_long:
        sl = entry_price * (1 - stop_loss) if stop_loss > 0 else None
        tp = entry_price * (1 + take_profit) if take_profit > 0 else None
    else:
        sl = entry_price * (1 + stop_loss) if stop_loss > 0 else None
        tp = entry_price * (1 - take_profit) if take_profit > 0 else None
    return sl, tp


# ---------------------------------------------------------------------------
# portfolio-level gross-exposure cap
# ---------------------------------------------------------------------------


def gross_exposure(
    base_equity: float,
    positions: Mapping[str, Tuple[Position, ...]],
) -> float:
    """Aggregate |gross notional| across all open lots / ``base_equity``.

    Sums ``abs(qty) * last_price`` for every open lot (longs and shorts both
    count positively — gross exposure is capital at work, not net). Pure.

    Pass **live MTM equity** as ``base_equity`` so the returned fraction is
    directly comparable to an equity-scaled lot fraction
    (``size * current_equity / price``). Dividing by *initial* capital instead
    makes a fixed cap lock the book out progressively as equity grows — the
    bug :func:`gross_gate` exists to avoid.
    """
    if base_equity <= 0:
        return 0.0
    total = 0.0
    for lots in positions.values():
        for pos in lots:
            total += abs(float(pos.qty)) * float(pos.last_price)
    return total / base_equity


def gross_gate(
    equity: float,
    positions: Mapping[str, Tuple[Position, ...]],
    cap: float,
    candidate_fraction: float,
) -> bool:
    """True when an entry should be **skipped** by the gross-exposure cap.

    ``cap`` is the aggregate |notional| ceiling as a fraction of live equity;
    ``cap >= 1.0`` (or ``<= 0``) disables the gate. ``candidate_fraction`` is
    the prospective lot's notional as a fraction of the **same** ``equity``
    base. Blocks the entry when the current gross plus the candidate would
    reach or exceed ``cap``.
    """
    if not 0.0 < cap < 1.0:
        return False
    return gross_exposure(equity, positions) + candidate_fraction >= cap
