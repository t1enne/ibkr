"""Unit tests for the portfolio-level gross-exposure cap (``utils.gross_gate``).

No DB, no engine. ``Position`` and plain mappings are enough — the helpers are
pure. Covers the regression that motivated the refactor: a cap fixed to
*initial* capital locks the book out progressively as equity grows; an
*equity*-relative cap stays meaningful.
"""

from __future__ import annotations

from typing import cast

import pandas as pd

from src.bt.state import ActionType, Position
from src.bt.strategies.utils import gross_exposure, gross_gate


def _ts(val: str) -> pd.Timestamp:
    result = cast(pd.Timestamp, pd.Timestamp(val))
    assert not pd.isna(result)
    return result


def _pos(qty: float, last: float) -> Position:
    return Position(
        symbol="X",
        qty=qty,
        entry_price=last,
        entry_time=_ts("1970-01-01"),
        stop_loss=None,
        take_profit=None,
        last_price=last,
        type=ActionType.long if qty >= 0 else ActionType.short,
        position_id="p1",
    )


def test_gross_exposure_zero_when_flat() -> None:
    assert gross_exposure(100_000.0, {}) == 0.0


def test_gross_exposure_sums_absolute_long_and_short() -> None:
    positions = {"A": (_pos(10.0, 100.0),), "B": (_pos(-5.0, 200.0),)}
    # 10*100 + 5*200 = 2000 ; / 10_000 = 0.2
    assert gross_exposure(10_000.0, positions) == 0.2


def test_gross_exposure_non_positive_base_fails_closed() -> None:
    positions = {"A": (_pos(10.0, 100.0),)}
    assert gross_exposure(0.0, positions) == 0.0
    assert gross_exposure(-1.0, positions) == 0.0


def test_gate_disabled_when_cap_off() -> None:
    positions = {"A": (_pos(10.0, 100.0),)}
    # cap >= 1.0 or <= 0 disables the gate
    assert gross_gate(10_000.0, positions, 1.0, 0.9) is False
    assert gross_gate(10_000.0, positions, 0.0, 0.9) is False


def test_gate_blocks_when_gross_plus_candidate_reaches_cap() -> None:
    positions = {"A": (_pos(10.0, 100.0),)}  # notional 1000 / 10_000 = gross 0.1
    assert gross_gate(10_000.0, positions, 0.5, 0.45) is True  # 0.1+0.45 > 0.5
    assert gross_gate(10_000.0, positions, 0.5, 0.1) is False  # 0.1+0.1 < 0.5


def test_gate_stays_meaningful_as_equity_grows() -> None:
    """Regression: an equity-relative cap must NOT freeze a compounding book.

    Cap 0.8, candidate lot = 0.4 of live equity. One 0.4 lot fits (0.4 < 0.8);
    a second would reach 0.8 and is blocked — the same decision at 1x and at
    2.35x equity. An init-relative cap instead blocks even the *first* lot once
    equity > init/risk_pct = 2.5x, freezing the book.
    """
    cap = 0.8
    candidate = 0.4
    for equity in (50_000.0, 117_500.0):  # 1.0x and 2.35x initial capital
        flat: dict[str, tuple[Position, ...]] = {}
        assert gross_gate(equity, flat, cap, candidate) is False  # first lot fits
        # one open 0.4 lot already in the book (qty*price == 0.4 * equity)
        one = {"A": (_pos(0.4 * equity / 100.0, 100.0),)}
        assert gross_exposure(equity, one) == 0.4
        assert gross_gate(equity, one, cap, candidate) is True  # second blocked
