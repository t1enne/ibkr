"""Reconcile — the pure heart of a live cycle: posture diff → ``OrderIntent``s.

Target posture (from the screen's ``LiveSignal``) vs the live book's current
side, per symbol, in ``config.symbols`` order. No I/O, no mutation, no broker.
**Closes always precede opens** (mirrors ``apply_fills``' non-opens-first rule),
so cash freed by a close is available to an open in the same cycle.

Absence of a signal is HOLD — never flatten. Only an explicit ``close`` (or a
side flip) closes a live lot. Sizing is never re-derived here: opens are sized
through the shared ``sized_signal``/``compute_qty`` layer, and an open that still
sizes to ``<= 0`` raises rather than placing an accidental order.

The book opens are sized against is this cycle's closes settled for real: the
close intents are priced with the SAME ``execute_signal`` the broker fills with
and folded through the shared ``apply_fills`` (non-opens-first), so the freed
cash and dropped lots are the broker's actual accounting — not a
``Σ qty*ref_price`` approximation of it.

NOTE: this module prices through the module-level ``execute_signal``
(``src.bt.exchange``), NOT through an injected exchange. Swapping the live
broker's exchange adapter therefore does NOT reroute reconcile's sizing book —
the two must be kept in step deliberately.
"""

from __future__ import annotations

import math
from typing import Literal, cast

import pandas as pd

from src.bt.exchange import execute_signal
from src.bt.portfolio.pure import apply_fills
from src.bt.size.pure import SizingParams, equity_of, sized_signal
from src.bt.state import (
    ActionType,
    ExecutionParams,
    PortfolioState,
    Position,
)
from src.bt.state.factories import create_execution_params, build_commission_model
from src.live.broker import intent_to_signal, ref_candle, trade_signal
from src.live.types import LiveConfig, LiveSignal, OrderIntent, PortfolioView

Side = Literal["long", "short", "flat"]


class UnknownSignalSymbol(ValueError):
    """A signal named a symbol outside ``config.symbols``.

    An explicit typed error (never a bare ``assert``): it must reject a stray
    symbol even under ``python -O``, where asserts are stripped. Subclasses
    ``ValueError`` so the existing CLI edge reports it as a usage error.
    """


#: Deterministic timestamp for the synthetic sizing fills. Only the discarded
#: ``FillEvent``/trade timestamps depend on it — cash and positions do not — so a
#: constant keeps reconcile pure and its sizing book reproducible.
_SETTLE_TS = cast("pd.Timestamp", pd.Timestamp(0))


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


def target_side(sig: LiveSignal) -> Side:
    """The posture a signal asks for: ``close`` means ``flat``."""
    return "flat" if sig.action == "close" else sig.action


def size_qty(price: float, portfolio: PortfolioView, config: LiveConfig) -> float:
    """Shares for an unsized open at *price*, via the shared sizing layer.

    Delegates to ``sized_signal`` so there is ONE sizing rule; equity comes from
    the shared ``equity_of``, which now accepts the ``PortfolioView`` Protocol.
    A non-finite price (NaN) or ``<= 0`` sizes to ``0.0`` rather than passing
    garbage downstream.
    """
    if not math.isfinite(price) or price <= 0:
        return 0.0
    probe = trade_signal(
        symbol="",
        action=ActionType.long,
        price=price,
        qty=0.0,
        ts=_SETTLE_TS,
    )
    sized = sized_signal(
        probe,
        equity_of(portfolio),
        portfolio.cash,
        ref_candle(price, "", _SETTLE_TS),
        _sizing_params(config),
    )
    return sized.qty


def reconcile(
    signals: tuple[LiveSignal, ...],
    portfolio: PortfolioView,
    config: LiveConfig,
    owned: frozenset[str] | None = None,
) -> tuple[OrderIntent, ...]:
    """Posture diff → orders. Deterministic, closes-before-opens, config order.

    ``owned`` scopes closes to lots this strategy opened (``position_id`` in the
    set); ``None`` means every lot. Only the first ``config.symbols`` entry per
    symbol is considered. Duplicate signals: last wins.

    ``current_side`` nets the WHOLE book (posture semantics — the broker account
    is the truth for what a symbol's net exposure is), while close intents are
    ownership-scoped by ``owned`` (a cycle must never close a lot it did not
    open). This asymmetry is deliberate: a foreign lot still counts toward the
    symbol's posture, but is never emitted as a close. Its consequence is made
    explicit here: when ``cur != "flat"`` but the ownership-scoped close list
    for that symbol is EMPTY (the whole book on the symbol is foreign), an
    opposite-side open is SKIPPED (HOLD) — opening it would double gross
    exposure without ever reaching the target posture.
    """
    by_symbol = {sig.symbol: sig for sig in signals}
    extra = set(by_symbol) - set(config.symbols)
    if extra:
        raise UnknownSignalSymbol(
            f"signal symbol not in config.symbols: {sorted(extra)}"
        )

    closes: list[OrderIntent] = []
    specs: list[tuple[LiveSignal, Side, Side]] = []
    seen: set[str] = set()
    for symbol in config.symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        sig = by_symbol.get(symbol)
        if sig is None:
            continue
        symbol_closes, spec = _plan_symbol(portfolio, sig, owned)
        closes.extend(symbol_closes)
        if spec is not None:
            specs.append(spec)
    # Only settle this cycle's closes if an open actually needs sizing against
    # them; a closes-only cycle never has to lift the (possibly minimal) view to
    # a full PortfolioState.
    view = _settled_book(portfolio, closes, config) if specs else portfolio
    opens = [_open_intent(sig, view, config, cur, tgt) for sig, cur, tgt in specs]
    return tuple(closes + opens)


def _plan_symbol(
    portfolio: PortfolioView, sig: LiveSignal, owned: frozenset[str] | None
) -> tuple[list[OrderIntent], tuple[LiveSignal, Side, Side] | None]:
    """One symbol's posture diff: its close intents plus an open spec (or ``None``).

    HOLD (``[], None``) when the target already matches the current side, or
    when the whole book on the symbol is foreign and only a flip would reach the
    target (opening would double gross exposure without closing anything).
    """
    symbol = sig.symbol
    cur = current_side(portfolio, symbol)
    tgt = target_side(sig)
    if tgt == cur:
        return [], None
    lots = _lots(portfolio, symbol, owned)
    if tgt == "flat":
        return _close_intents(symbol, sig, lots, f"{cur}->flat"), None
    if cur == "flat":
        return [], (sig, cur, tgt)
    flips = _close_intents(symbol, sig, lots, f"{cur}->{tgt}")
    # Foreign-only book on this symbol: closing is not ours to do, so opening
    # the opposite side would add exposure without closing the old one. HOLD
    # instead of doubling gross exposure.
    if not flips:
        return [], None
    return flips, (sig, cur, tgt)


def _settled_book(
    portfolio: PortfolioView, closes: list[OrderIntent], config: LiveConfig
) -> PortfolioView:
    """The book opens are sized against: this cycle's closes settled for real.

    Each close is priced with the same ``execute_signal`` the broker fills with
    and the whole set is folded through the shared ``apply_fills``
    (non-opens-first), so freed cash is the broker's actual proceeds (spread /
    slippage / commission included) and closed lots are dropped by the same code
    path — one implementation of close proceeds, shared with settlement. Only
    ``cash``/``positions`` are needed downstream, so the view is lifted to a
    ``PortfolioState`` with empty trades/equity to reuse the shared settler.
    """
    if not closes:
        return portfolio
    state = PortfolioState(
        cash=portfolio.cash,
        positions=portfolio.positions,
        trades=(),
        equity_curve=(),
        initial_capital=portfolio.initial_capital,
    )
    params = _exec_params(config)
    fills = tuple(
        execute_signal(
            intent_to_signal(intent, _SETTLE_TS, state),
            ref_candle(intent.ref_price, intent.symbol, _SETTLE_TS),
            params,
        )
        for intent in closes
    )
    settled, _rejections, _scales = apply_fills(
        state, fills, commission_model=params.commission_model
    )
    # ``apply_fills`` stamps its settlements at ``_SETTLE_TS`` (the epoch probe),
    # so the returned book's ``trades``/``equity_curve`` are artifacts. Only
    # ``cash``/``positions`` are consumed downstream: re-emit a clean view so no
    # epoch-0 trade can leak. ``initial_capital`` is unchanged by the settler.
    return PortfolioState(
        cash=settled.cash,
        positions=settled.positions,
        trades=(),
        equity_curve=(),
        initial_capital=settled.initial_capital,
    )


def _lots(
    portfolio: PortfolioView, symbol: str, owned: frozenset[str] | None
) -> tuple[Position, ...]:
    """The symbol's closable lots, filtered to ``owned`` ids when scoping is active.

    A lot with no ``position_id`` is always skipped — it cannot be proven ours
    and ``_close_position`` could not target it anyway. Under ownership scoping
    the survivors must additionally be in ``owned``.
    """
    lots = tuple(p for p in portfolio.positions.get(symbol, ()) if p.position_id)
    if owned is None:
        return lots
    return tuple(p for p in lots if p.position_id in owned)


def _close_intents(
    symbol: str, sig: LiveSignal, lots: tuple[Position, ...], transition: str
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
            decision_ts=sig.bar_ts,
        )
        for pos in lots
    ]


def _open_intent(
    sig: LiveSignal,
    portfolio: PortfolioView,
    config: LiveConfig,
    cur: Side,
    tgt: Side,
) -> OrderIntent:
    """A single open intent, sized from the signal or config; never <= 0 shares.

    Reuses ``sized_signal`` for the whole "explicit qty else compute it" rule so
    the branch lives in ONE place (``src/bt/size/pure.py``), not here. An open
    that still sizes to ``<= 0`` raises — never place an order sized by accident.
    """
    action = ActionType.long if tgt == "long" else ActionType.short
    price = sig.price if math.isfinite(sig.price) and sig.price > 0 else 0.0
    signal = trade_signal(
        symbol=sig.symbol,
        action=action,
        price=price,
        qty=sig.qty,
        ts=_SETTLE_TS,
        stop_loss=sig.stop_loss,
        take_profit=sig.take_profit,
        tag=sig.tag,
        reason=f"open {tgt} ({cur}->{tgt})",
    )
    sized = sized_signal(
        signal,
        equity_of(portfolio),
        portfolio.cash,
        ref_candle(price, sig.symbol, _SETTLE_TS),
        _sizing_params(config),
    )
    if sized.qty <= 0:
        raise ValueError(
            f"unsized open {sig.symbol}: signal qty={sig.qty}, sized qty={sized.qty} "
            f"(size_mode={config.size_mode!r}, size={config.size})"
        )
    return OrderIntent(
        symbol=sig.symbol,
        action=action,
        qty=sized.qty,
        ref_price=sig.price,
        reason=f"open {tgt} ({cur}->{tgt})",
        position_id=None,
        stop_loss=sig.stop_loss,
        take_profit=sig.take_profit,
        tag=sig.tag,
        decision_ts=sig.bar_ts,
        # The exact cash the sizer clamped against (the settled ``view``), carried
        # to the edge so the notional guard re-checks the SAME quantity (finding M5).
        cash_bound=portfolio.cash,
    )


def _sizing_params(config: LiveConfig) -> SizingParams:
    """The shared ``SizingParams`` for this config (one sizing rule)."""
    return SizingParams.from_dict(
        {
            "sizing_mode": config.size_mode,
            "size": config.size,
            "max_symbol_allocation": config.max_symbol_allocation,
        }
    )


def _exec_params(config: LiveConfig) -> ExecutionParams:
    """The execution params the broker fills with (same construction the CLI uses)."""
    return create_execution_params(
        spread_bps=config.spread_bps,
        slippage_bps=config.slippage_bps,
        fixed_commission=config.commission,
        commission_model=build_commission_model(
            config.commission,
            config.commission_per_share,
            config.commission_min,
            config.commission_max_pct,
        ),
    )
