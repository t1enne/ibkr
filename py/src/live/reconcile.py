"""Reconcile — the pure heart of a live cycle: posture diff → ``OrderIntent``s.

Target posture (from the screen's ``LiveSignal``) vs the live book's current
side, per symbol, in ``config.symbols`` order. No I/O, no mutation, no broker.
**Closes always precede opens** (mirrors ``apply_fills``' non-opens-first rule),
so cash freed by a close is available to an open in the same cycle.

Absence of a signal is HOLD — never flatten. Only an explicit ``close`` (or a
side flip) closes a live lot. Sizing is never re-derived here: an unsized open
routes through the shared ``SizingParams``/``compute_qty`` layer, and an open
that still sizes to ``<= 0`` raises rather than placing an accidental order.
"""

from __future__ import annotations

from typing import Literal

from src.bt.portfolio.pure import calculate_positions_value
from src.bt.size.pure import SizingParams, compute_qty
from src.bt.state import ActionType, Position
from src.live.types import LiveConfig, OrderIntent, PortfolioView

Side = Literal["long", "short", "flat"]


def current_side(portfolio: PortfolioView, symbol: str) -> Side:
    """Net the symbol's lots into a side.

    ``Position.qty`` is positive; the side lives on ``Position.type``, so long
    lots add ``+qty`` and short lots add ``-qty``. Positive net → ``long``,
    negative → ``short``, zero (or no lots) → ``flat``.
    """
    net = 0.0
    for pos in portfolio.positions.get(symbol, ()):
        net += pos.qty if pos.type is ActionType.long else -pos.qty
    if net > 0:
        return "long"
    if net < 0:
        return "short"
    return "flat"


def target_side(sig) -> Side:
    """The posture a signal asks for: ``close`` means ``flat``."""
    return "flat" if sig.action == "close" else sig.action


def size_qty(price: float, portfolio: PortfolioView, config: LiveConfig) -> float:
    """Shares for an unsized open, via the shared sizing layer.

    Equity is rebuilt as ``cash + calculate_positions_value(positions)`` rather
    than calling ``equity_of`` — the latter takes the concrete ``PortfolioState``
    and would not accept a ``PortfolioView``-typed book. Same arithmetic, still
    the one sizing implementation.
    """
    if price <= 0:
        return 0.0
    params = SizingParams.from_dict(
        {
            "sizing_mode": config.size_mode,
            "size": config.size,
            "max_symbol_allocation": config.max_symbol_allocation,
        }
    )
    equity = portfolio.cash + calculate_positions_value(portfolio.positions)
    return compute_qty(equity=equity, cash=portfolio.cash, price=price, params=params)


def reconcile(
    signals: tuple,
    portfolio: PortfolioView,
    config: LiveConfig,
    owned: frozenset[str] | None = None,
) -> tuple[OrderIntent, ...]:
    """Posture diff → orders. Deterministic, closes-before-opens, config order.

    ``owned`` scopes closes to lots this strategy opened (``position_id`` in the
    set); ``None`` means every lot. Only the first ``config.symbols`` entry per
    symbol is considered. Duplicate signals: last wins.
    """
    by_symbol = {sig.symbol: sig for sig in signals}
    extra = set(by_symbol) - set(config.symbols)
    assert not extra, f"signal symbol not in config.symbols: {sorted(extra)}"

    closes: list[OrderIntent] = []
    opens: list[OrderIntent] = []
    seen: set[str] = set()
    for symbol in config.symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        sig = by_symbol.get(symbol)
        if sig is None:
            continue
        cur = current_side(portfolio, symbol)
        tgt = target_side(sig)
        if tgt == cur:
            continue
        lots = _lots(portfolio, symbol, owned)
        if tgt == "flat":
            closes.extend(_close_intents(symbol, sig, lots, f"{cur}->flat"))
            continue
        if cur != "flat":
            closes.extend(_close_intents(symbol, sig, lots, f"{cur}->{tgt}"))
        opens.append(_open_intent(sig, portfolio, config, cur, tgt))
    return tuple(closes + opens)


def _lots(
    portfolio: PortfolioView, symbol: str, owned: frozenset[str] | None
) -> tuple[Position, ...]:
    """The symbol's lots, filtered to ``owned`` ids when scoping is active.

    A lot with no ``position_id`` is skipped under ownership scoping — it cannot
    be proven ours, and ``_close_position`` could not target it anyway.
    """
    lots = portfolio.positions.get(symbol, ())
    if owned is None:
        return tuple(lots)
    return tuple(p for p in lots if p.position_id and p.position_id in owned)


def _close_intents(
    symbol: str, sig, lots: tuple[Position, ...], transition: str
) -> list[OrderIntent]:
    """One close intent per targeted lot; each carries its lot's ``position_id``.

    A ``close`` signal naming a lot targets that lot only; a bare close (or a
    side flip) closes every lot. ``_close_position`` requires the id and the
    positive lot qty.
    """
    if sig.action == "close" and sig.position_id:
        lots = tuple(p for p in lots if p.position_id == sig.position_id)
    return [
        OrderIntent(
            symbol=symbol,
            action=ActionType.close,
            qty=pos.qty,
            ref_price=sig.price,
            reason=f"close lot {pos.position_id} ({transition})",
            position_id=pos.position_id,
        )
        for pos in lots
    ]


def _open_intent(
    sig, portfolio: PortfolioView, config: LiveConfig, cur: Side, tgt: Side
) -> OrderIntent:
    """A single open intent, sized from the signal or config; never <= 0 shares."""
    qty = sig.qty if sig.qty > 0 else size_qty(sig.price, portfolio, config)
    if qty <= 0:
        raise ValueError(
            f"unsized open {sig.symbol}: signal qty={sig.qty}, sized qty={qty} "
            f"(size_mode={config.size_mode!r}, size={config.size})"
        )
    return OrderIntent(
        symbol=sig.symbol,
        action=ActionType.long if tgt == "long" else ActionType.short,
        qty=qty,
        ref_price=sig.price,
        reason=f"open {tgt} ({cur}->{tgt})",
        position_id=None,
        stop_loss=sig.stop_loss,
        take_profit=sig.take_profit,
        tag=sig.tag,
    )
