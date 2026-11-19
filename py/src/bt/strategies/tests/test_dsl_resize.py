"""``ctx.resize`` — the target-weight primitive's contract (blockers B2, B3).

Methodology: build a real ``PortfolioState`` (``create_initial_portfolio``) plus
one open lot per case, hand the DSL a minimal ``StrategyContext`` over it, and
assert what the method EMITS. Nothing here re-runs the engine: the engine-side
cash invariant is pinned in ``src/bt/portfolio/tests/test_apply_fills.py``.

Each test names whether it is a regression guard (it fails the pre-fix code) or
a characterisation of documented behaviour.
"""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from src.bt.state import (
    ActionType,
    BacktestState,
    Candle,
    PortfolioState,
    Position,
    create_initial_portfolio,
)
from src.bt.strategies.dsl import StrategyContext
from src.bt.strategies.ta_context import TaContext


def _ts(value: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value))


def _ta(price: float) -> TaContext:
    """One-bar feed so ``ctx.price`` resolves."""
    data = pd.DataFrame(
        {
            "open": [price],
            "high": [price],
            "low": [price],
            "close": [price],
            "volume": [1_000.0],
        },
        index=pd.DatetimeIndex([_ts("2024-01-02")]),
    )
    data.columns = pd.MultiIndex.from_product([["AAA"], data.columns])
    return TaContext.from_data(data, ("AAA",), "1d")


def _ctx(
    portfolio: PortfolioState, price: float, cash: float | None = None
) -> StrategyContext:
    if cash is not None:
        portfolio = PortfolioState(
            cash=cash,
            positions=portfolio.positions,
            trades=portfolio.trades,
            equity_curve=portfolio.equity_curve,
            initial_capital=portfolio.initial_capital,
        )
    state = cast(BacktestState, type("S", (), {"portfolio": portfolio})())
    candle = Candle(
        timestamp=_ts("2024-01-02"),
        symbol="AAA",
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1_000.0,
        interval="1d",
    )
    return StrategyContext(
        state=state,
        candle=candle,
        params=None,
        ta=_ta(price),
        symbols=("AAA",),
        interval="1d",
    )


def _lot(side: ActionType, qty: float, entry: float, last: float) -> PortfolioState:
    portfolio = create_initial_portfolio(
        initial_capital=10_000.0, start_timestamp=_ts("2024-01-01")
    )
    position = Position(
        symbol="AAA",
        qty=qty,
        entry_price=entry,
        entry_time=_ts("2024-01-01"),
        stop_loss=None,
        take_profit=None,
        last_price=last,
        type=side,
        position_id="AAA_lot_0",
    )
    return PortfolioState(
        cash=portfolio.cash,
        positions={"AAA": (position,)},
        trades=portfolio.trades,
        equity_curve=portfolio.equity_curve,
        initial_capital=portfolio.initial_capital,
    )


def test_resize_refuses_a_short_book():
    """REGRESSION (B3): a short lot must raise, not trade the wrong direction.

    ``Position.qty`` is a positive magnitude for both sides, so the pre-fix
    ``sum(lot.qty)`` read a 100-share SHORT as +100 long: a target above 100
    shares produced ``delta > 0`` and the emitted ``rebalance`` made the short
    BIGGER while the caller asked for long exposure.
    """
    ctx = _ctx(_lot(ActionType.short, 100.0, 100.0, 100.0), price=100.0)
    with pytest.raises(ValueError, match="long-only"):
        ctx.resize("AAA", 0.5)


def test_resize_values_a_short_at_the_engine_value():
    """REGRESSION (B3): ``current_equity`` marks a short like the engine does.

    Engine (``calculate_positions_value``): ``qty * (2*entry - last)``. The
    pre-fix DSL summed ``qty * last`` for every lot, so a short that is LOSING
    (last above entry) was marked UP and the equity base -- hence every
    ``size_mode="equity"`` order -- was too large.
    """
    # 100 shares short at 100, now 120: unrealized -2,000 on a 10,000 book.
    portfolio = _lot(ActionType.short, 100.0, 100.0, 120.0)
    ctx = _ctx(portfolio, price=120.0)
    position_value = ctx.current_equity() - portfolio.cash
    assert position_value == pytest.approx(100.0 * (2 * 100.0 - 120.0))  # +8,000
    # The pre-fix formula (qty * last) said +12,000 -- a losing short marked UP.
    assert position_value != pytest.approx(100.0 * 120.0)


def test_resize_flat_book_emits_one_long_of_the_target_weight():
    """Characterisation: flat -> a single cash-guarded ``long``, not a rebalance."""
    ctx = _ctx(
        create_initial_portfolio(
            initial_capital=10_000.0, start_timestamp=_ts("2024-01-01")
        ),
        price=100.0,
    )
    ctx.resize("AAA", 0.5)
    signals = ctx._signals
    assert len(signals) == 1
    assert signals[0].action == ActionType.long
    assert signals[0].qty == pytest.approx(50.0)  # 50% of 10,000 at 100


def test_resize_held_book_emits_a_signed_delta():
    """Characterisation: holding -> ONE ``rebalance`` delta, same lot."""
    ctx = _ctx(_lot(ActionType.long, 10.0, 100.0, 100.0), price=100.0, cash=5_000.0)
    # equity = 5,000 cash + 1,000 position = 6,000; 50% = 30 shares.
    ctx.resize("AAA", 0.5)
    signals = ctx._signals
    assert len(signals) == 1
    assert signals[0].action == ActionType.rebalance
    assert signals[0].qty == pytest.approx(20.0)
    assert signals[0].position_id == "AAA_lot_0"


def test_resize_buy_is_capped_by_cash():
    """Characterisation: a buy can never ask for more cash than the book holds."""
    ctx = _ctx(_lot(ActionType.long, 10.0, 100.0, 100.0), price=100.0, cash=500.0)
    # equity = 500 + 1,000 = 1,500; a 99% target wants 14.85 shares (+4.85),
    # which costs 485 <= 500 * (1 - 0.01) = 495, so the cap does not bind here.
    ctx.resize("AAA", 0.99)
    assert ctx._signals[0].qty == pytest.approx(4.85)
    # A 100%-of-equity target on a 50 book: the requested 0.5 shares is trimmed
    # to what the cash reserve allows, and the emitted buy never exceeds it.
    ctx = _ctx(
        create_initial_portfolio(
            initial_capital=50.0, start_timestamp=_ts("2024-01-01")
        ),
        price=100.0,
    )
    ctx.resize("AAA", 1.0)
    assert len(ctx._signals) == 1
    assert ctx._signals[0].qty < 0.5, "the request was trimmed by the cash cap"
    assert ctx._signals[0].qty * 100.0 <= 50.0 * 0.99 + 1e-9


def test_resize_does_not_re_emit_a_dust_add_every_bar():
    """REGRESSION (rejection flood): a sub-dollar surplus is a NO-OP.

    The RBAA arm re-emits ``w = 1.0`` every bar, so once its cash stub is dust
    ``resize`` asked for the stub's worth of shares. Pre-fix the cap
    ``round(cash * 0.99 / price, 4)`` rounded a 0.000258-share surplus UP to
    0.0003 shares, which costs more than the $0.0314 of cash at ANY price (the
    fixed commission alone exceeds it) -- so the fill was rejected, cash did not
    move, and the same order was re-emitted on the next bar: 5127 rejections on
    one config. Post-fix the budget is ``cash - max(2% * cash, $1)`` <= 0, so
    nothing is emitted and the loop cannot start.
    """
    # Book: 1,000.0 cash -- wait, the stub case is the point, so: a lot worth
    # 826.9787 * 120.25 and 0.0314 of cash (the measured RBAA bar).
    ctx = _ctx(
        _lot(ActionType.long, 826.9787, 100.0, 120.25), price=120.25, cash=0.0314
    )
    ctx.resize("AAA", 1.0)
    assert ctx._signals == [], "a sub-dollar surplus buys no shares: emit nothing"


def test_resize_add_never_exceeds_the_cash_budget():
    """Characterisation: the emitted qty is a whole share-tick UNDER the cap.

    ``round(cap/price, 4)`` could round a sub-tick surplus UP past the budget it
    was capped to; the emitted qty is truncated to the ``0.0001`` tick instead,
    so ``qty * price <= cash - max(2% * cash, $1)`` always holds.
    """
    ctx = _ctx(_lot(ActionType.long, 10.0, 100.0, 100.0), price=100.0, cash=1_000.0)
    ctx.resize("AAA", 1.0)
    qty = ctx._signals[0].qty
    assert qty == pytest.approx(round(qty, 4), abs=0.0)
    assert qty * 100.0 <= 1_000.0 - max(0.02 * 1_000.0, 1.0) + 1e-6
