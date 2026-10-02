"""Tests for ``apply_fills`` — the atomic, order-invariant cohort settlement.

Focus: order invariance, the single-open legacy no-op, closes freeing capital
before opens, SL/TP untouched, empty cohort, and genuine-exhaustion reporting.
"""

from typing import cast

import pandas as pd
import pytest

from src.bt.portfolio.pure import FillRejection, apply_fills, next_position_id
from src.bt.state import (
    ActionType,
    FillEvent,
    PortfolioState,
    TradeSignal,
    create_initial_portfolio,
)


def _ts(val: str) -> pd.Timestamp:
    result = cast(pd.Timestamp, pd.Timestamp(val))
    assert not pd.isna(result)
    return result


def _portfolio(cash: float = 10_000.0) -> PortfolioState:
    return create_initial_portfolio(
        initial_capital=cash, start_timestamp=_ts("2024-01-01")
    )


def _open(
    symbol: str,
    qty: float,
    price: float,
    *,
    commission: float = 0.0,
    sl: float | None = None,
    tp: float | None = None,
) -> FillEvent:
    return FillEvent(
        signal=TradeSignal(
            action=ActionType.long,
            symbol=symbol,
            timestamp=_ts("2024-01-02"),
            price=price,
            qty=qty,
            stop_loss=sl,
            take_profit=tp,
        ),
        filled_qty=qty,
        executed_price=price,
        commission=commission,
        slippage=0.0,
        timestamp=_ts("2024-01-02"),
    )


def test_empty_cohort_is_a_noop():
    portfolio = _portfolio()
    result, rejections = apply_fills(portfolio, ())
    assert result == portfolio
    assert rejections == ()


def test_single_open_is_bit_identical_to_apply_fill():
    """The legacy guard: a one-open cohort is never scaled or trimmed."""
    from src.bt.portfolio.pure import apply_fill

    portfolio = _portfolio(cash=10_000.0)
    # Requests MORE than cash -> must still be rejected, exactly as before.
    fill = _open("AAPL", 200.0, 100.0)
    legacy = apply_fill(portfolio, fill)
    result, rejections = apply_fills(portfolio, (fill,))
    assert result == legacy
    assert len(rejections) == 1
    assert rejections[0].symbol == "AAPL"


def test_two_opens_share_cash_and_both_fill():
    """Two opens requesting the whole book each end up ~1/N and both fill."""
    portfolio = _portfolio(cash=10_000.0)
    a = _open("AAA", 100.0, 100.0)  # requests 10_000
    b = _open("BBB", 100.0, 100.0)  # requests 10_000
    result, rejections = apply_fills(portfolio, (a, b))
    assert rejections == ()
    assert set(result.positions) == {"AAA", "BBB"}
    for lots in result.positions.values():
        assert lots[0].qty == pytest.approx(50.0, abs=1e-4)
    assert result.cash >= 0


def test_scale_cohorts_false_rejects_overflow_instead_of_scaling():
    """The research counterfactual: no shared scale, full-size first-come fills."""
    portfolio = _portfolio(cash=10_000.0)
    a = _open("AAA", 100.0, 100.0)  # requests 10_000 -> fills full
    b = _open("BBB", 100.0, 100.0)  # no cash left -> rejected
    result, rejections = apply_fills(portfolio, (a, b), scale_cohorts=False)
    assert set(result.positions) == {"AAA"}
    assert result.positions["AAA"][0].qty == pytest.approx(100.0, abs=1e-4)
    assert tuple(r.symbol for r in rejections) == ("BBB",)


def test_scale_cohorts_false_lone_open_unchanged():
    """A one-open cohort is identical with scaling on or off."""
    portfolio = _portfolio(cash=10_000.0)
    fill = _open("AAA", 50.0, 100.0)
    on, _ = apply_fills(portfolio, (fill,), scale_cohorts=True)
    off, _ = apply_fills(portfolio, (fill,), scale_cohorts=False)
    assert on == off


def test_cohort_order_does_not_change_the_book():
    portfolio = _portfolio(cash=10_000.0)
    fills = (
        _open("AAA", 100.0, 100.0),
        _open("BBB", 80.0, 100.0),
        _open("CCC", 120.0, 100.0),
    )
    forward, rej_f = apply_fills(portfolio, fills)
    backward, rej_b = apply_fills(portfolio, tuple(reversed(fills)))
    assert forward == backward
    assert rej_f == rej_b


def _invested(symbol: str, qty: float, price: float) -> PortfolioState:
    """A book holding one lot of ``symbol`` with ZERO cash left (fully invested)."""
    portfolio = _portfolio(cash=qty * price + 1.0)
    funded, _ = apply_fills(portfolio, (_open(symbol, qty, price),))
    assert symbol in funded.positions
    return funded


def test_close_frees_cash_before_open():
    """A cohort close settles first, so its proceeds fund a same-cohort open."""
    funded = _invested("AAA", 100.0, 100.0)
    close = FillEvent(
        signal=TradeSignal(
            action=ActionType.close,
            symbol="AAA",
            timestamp=_ts("2024-01-03"),
            price=100.0,
            position_id=funded.positions["AAA"][0].position_id,
        ),
        filled_qty=100.0,
        executed_price=100.0,
        commission=0.0,
        slippage=0.0,
        timestamp=_ts("2024-01-03"),
    )
    later = _open("BBB", 90.0, 100.0)
    # Close listed AFTER the open, yet must settle FIRST and fund it.
    result, rejections = apply_fills(funded, (later, close))
    assert rejections == ()
    assert "BBB" in result.positions
    assert "AAA" not in result.positions


def test_closes_and_rebalances_are_never_scaled():
    funded = _invested("AAA", 100.0, 100.0)
    assert funded.positions["AAA"][0].qty == 100.0

    close = FillEvent(
        signal=TradeSignal(
            action=ActionType.close,
            symbol="AAA",
            timestamp=_ts("2024-01-03"),
            price=100.0,
            position_id=funded.positions["AAA"][0].position_id,
        ),
        filled_qty=100.0,
        executed_price=100.0,
        commission=0.0,
        slippage=0.0,
        timestamp=_ts("2024-01-03"),
    )
    two_opens = (_open("BBB", 100.0, 100.0), _open("CCC", 100.0, 100.0))
    result, _ = apply_fills(funded, (close,) + two_opens)
    assert "AAA" not in result.positions  # close applied in full, unscaled
    assert len(result.trades) == 3


def test_sl_tp_levels_survive_scaling_untouched():
    portfolio = _portfolio(cash=10_000.0)
    a = _open("AAA", 100.0, 100.0, sl=95.0, tp=110.0)
    b = _open("BBB", 100.0, 100.0, sl=90.0, tp=120.0)
    result, _ = apply_fills(portfolio, (a, b))
    stops = {s: lots[0].stop_loss for s, lots in result.positions.items()}
    targets = {s: lots[0].take_profit for s, lots in result.positions.items()}
    assert stops == {"AAA": 95.0, "BBB": 90.0}
    assert targets == {"AAA": 110.0, "BBB": 120.0}
    # ...while qty WAS scaled.
    assert result.positions["AAA"][0].qty < 100.0


def test_genuine_exhaustion_still_reports():
    """A cohort that cannot fit even scaled reports the shortfall."""
    portfolio = _portfolio(cash=100.0)
    a = _open("AAA", 100.0, 100.0, commission=1000.0)
    rejections: tuple[FillRejection, ...] = ()
    _result, rejections = apply_fills(portfolio, (a, _open("BBB", 1.0, 1.0)))
    assert any(r.symbol == "AAA" for r in rejections)


# ---------------------------------------------------------------------------
# position_id collision: same symbol, same fill bar, multi-lot
# ---------------------------------------------------------------------------


def test_same_symbol_same_bar_opens_get_distinct_ids():
    """Two opens on one symbol at one fill bar must NOT share a position_id."""
    portfolio = _portfolio(cash=100_000.0)
    result, rejections = apply_fills(
        portfolio, (_open("AAA", 1.0, 100.0), _open("AAA", 2.0, 100.0))
    )
    assert rejections == ()
    lots = result.positions["AAA"]
    assert len(lots) == 2
    assert lots[0].position_id != lots[1].position_id


def test_close_removes_exactly_one_same_bar_lot():
    """Closing one of two same-bar lots removes only that lot."""
    portfolio = _portfolio(cash=100_000.0)
    funded, _ = apply_fills(
        portfolio, (_open("AAA", 1.0, 100.0), _open("AAA", 2.0, 100.0))
    )
    target, other = funded.positions["AAA"]
    close = FillEvent(
        signal=TradeSignal(
            action=ActionType.close,
            symbol="AAA",
            timestamp=_ts("2024-01-03"),
            price=100.0,
            position_id=target.position_id,
        ),
        filled_qty=target.qty,
        executed_price=100.0,
        commission=0.0,
        slippage=0.0,
        timestamp=_ts("2024-01-03"),
    )
    result, _ = apply_fills(funded, (close,))
    remaining = result.positions["AAA"]
    assert len(remaining) == 1
    assert remaining[0].position_id == other.position_id
    assert remaining[0].position_id != target.position_id


def test_auto_ids_are_deterministic_across_runs():
    """Identical inputs -> identical auto-generated id sequence."""

    def run() -> tuple[str, ...]:
        p = _portfolio(cash=100_000.0)
        r, _ = apply_fills(p, (_open("AAA", 1.0, 100.0), _open("AAA", 1.0, 100.0)))
        return tuple(lot.position_id for lot in r.positions["AAA"])

    first, second = run(), run()
    assert first == second
    assert len(set(first)) == 2


def test_next_position_id_shape_and_uniqueness():
    ts = _ts("2024-01-02")
    assert next_position_id("SPY", ts, 0) == f"SPY_{ts.timestamp()}_0"
    assert next_position_id("SPY", ts, 0) == next_position_id("SPY", ts, 0)
    assert next_position_id("SPY", ts, 0) != next_position_id("SPY", ts, 1)
