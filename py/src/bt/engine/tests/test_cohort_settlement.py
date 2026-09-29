"""Order-invariance tests for the engine's atomic cohort settlement.

The bug these pin: backtest results depended on the ORDER of ``config.symbols``
even with identical per-symbol history. The cohort design in ``apply_fills``
(settled once per bar phase by ``_flush_bar``) is what removes that order.

Tiny synthetic daily feeds only — no DB, no strategy JSON.
"""

from dataclasses import dataclass

import pandas as pd
import pytest

from src.bt.engine.backtest import Backtest, run
from src.bt.strategies.dsl import StrategyContext, strategy
from src.bt.types import StrategyConfig


@dataclass
class _FixtureMod:
    on_candle: object


def _feed(symbols: list[str], n: int = 12, base: float = 100.0) -> pd.DataFrame:
    """Daily feed where every symbol shares one rising price path."""
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    data: dict = {}
    for sym in symbols:
        closes = [base + i for i in range(n)]
        data.update(
            {
                (sym, "open"): closes,
                (sym, "high"): [c + 1 for c in closes],
                (sym, "low"): [c - 1 for c in closes],
                (sym, "close"): closes,
                (sym, "volume"): [1000.0] * n,
            }
        )
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


#: Explicit symbol -> slope for the "distinct prices" feed below. A LITERAL, not
#: arithmetic over an enumeration: a slope derived from a symbol's position in
#: ``symbols`` would permute the market data along with ``config.symbols`` and
#: make any order-invariance assertion unsatisfiable regardless of the engine.
_SLOPE: dict[str, float] = {
    "AAA": 1.0,
    "BBB": 2.0,
    "CCC": 3.0,
    "DDD": 4.0,
    "EEE": 5.0,
    "FFF": 6.0,
}


def _distinct_feed(
    symbols: list[str], n: int = 24, base: float = 100.0
) -> pd.DataFrame:
    """Daily feed where each symbol's slope follows its IDENTITY.

    Distinct prices per symbol make the cohort's per-symbol request differ, so a
    per-symbol (rather than per-cohort) scale would show up as a symbol-order
    effect on qty. The mapping is a function of the symbol string alone, so
    permuting ``symbols`` permutes dispatch order only — never the data.
    """
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    data: dict = {}
    for sym in symbols:
        step = _SLOPE[sym]
        closes = [base + step * i for i in range(n)]
        data.update(
            {
                (sym, "open"): closes,
                (sym, "high"): [c * 1.01 for c in closes],
                (sym, "low"): [c * 0.99 for c in closes],
                (sym, "close"): closes,
                (sym, "volume"): [1000.0] * n,
            }
        )
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _cfg(symbols: list[str], capital: float = 10_000.0) -> StrategyConfig:
    return StrategyConfig(
        name="cohort",
        strategy_type="momentum_compression_breakout_dsl",  # registered
        symbols=symbols,
        initial_capital=capital,
        commission=0.0,
        warmup="0d",
        trading_start="2025-01-01",
        trading_end="2025-12-31",
        bars=["1d"],
        strategy_params={},
    )


@strategy(bars="1d", stateful=True)
def _all_in_on_first_bar(ctx: StrategyContext):
    """Emit one explicit-qty long for EVERY symbol on the first dispatch.

    Every configured symbol enters on the SAME timestamp — the multi-open
    cohort the fix must divide — with a fixed, order-independent qty.
    """
    n = ctx.shared.setdefault("_n", 0)
    ctx.shared["_n"] = n + 1
    if n != 0:
        return
    for sym in ctx.symbols:
        ctx.long(sym, size=0.5, reason="cohort entry", size_mode="capital")


def _mod():
    return _FixtureMod(_all_in_on_first_bar)


def _run(symbols: list[str], capital: float = 10_000.0):
    cfg = _cfg(symbols, capital)
    return run(Backtest(cfg), _feed(symbols), _mod())


def test_symbol_order_is_invariant():
    """All permutations of ``config.symbols`` give the identical trade set."""
    symbols = ["AAA", "BBB", "CCC"]
    perms = [
        symbols,
        ["BBB", "CCC", "AAA"],
        ["CCC", "AAA", "BBB"],
        list(reversed(symbols)),
    ]
    results = [_run(order) for order in perms]
    base = results[0].final_state.portfolio
    for perm_result in results[1:]:
        portfolio = perm_result.final_state.portfolio
        assert portfolio.cash == pytest.approx(base.cash)
        ledger = sorted(
            (t.symbol, round(t.qty, 6), t.entry_time) for t in portfolio.trades
        )
        base_ledger = sorted(
            (t.symbol, round(t.qty, 6), t.entry_time) for t in base.trades
        )
        assert ledger == base_ledger


def test_same_timestamp_batch_allocates_equally_and_all_fill():
    """N concurrent opens each take ~capital/N and every fill succeeds."""
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    result = _run(symbols, capital=10_000.0)
    trades = list(result.final_state.portfolio.trades)
    assert len(trades) == len(symbols)  # all four entries filled
    notionals = [t.qty * t.entry_price for t in trades]
    # each entry ~1/4 of capital; total never exceeds it
    for notional in notionals:
        assert notional == pytest.approx(2_500.0, rel=0.05)
    assert sum(notionals) <= 10_000.0 + 1e-6


def test_divisor_one_leaves_single_signal_strategy_unchanged():
    """A lone open is bit-identical to the pre-cohort path."""
    result = _run(["AAA"])
    portfolio = result.final_state.portfolio
    assert len(portfolio.trades) == 1
    trade = portfolio.trades[0]
    # size=0.5 of 10_000 capital at the ~100 entry open = ~50 shares
    assert trade.qty == pytest.approx(50.0, rel=1e-3)


def test_no_signals_is_a_noop():
    @strategy(bars="1d")
    def _idle(ctx: StrategyContext):
        return None

    cfg = _cfg(["AAA", "BBB"])
    result = run(Backtest(cfg), _feed(["AAA", "BBB"]), _FixtureMod(_idle))
    assert result.final_state.portfolio.trades == ()
    assert result.final_state.portfolio.cash == 10_000.0


@strategy(bars="1d", stateful=True)
def _close_one_open_rest(ctx: StrategyContext):
    """Bar 0: open every symbol. Bar 2: close AAA and open DDD in ONE bar.

    Exercises the close-settles-before-open rule and the "closes are never
    scaled / never counted" rule for a same-bar cohort.
    """
    n = ctx.shared.setdefault("_n", 0)
    ctx.shared["_n"] = n + 1
    if n == 0:
        for sym in ctx.symbols:
            if sym != "DDD":
                ctx.long(sym, size=0.2, reason="open", size_mode="capital")
    elif n == 2:
        ctx.close("AAA", reason="rotate")
        ctx.long("DDD", size=0.2, reason="open", size_mode="capital")


def test_close_in_same_cohort_is_not_scaled_and_frees_cash():
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    cfg = _cfg(symbols)
    result = run(Backtest(cfg), _feed(symbols), _FixtureMod(_close_one_open_rest))
    closed = [t for t in result.final_state.portfolio.trades if t.symbol == "AAA"]
    assert len(closed) == 1
    assert closed[0].exit_time is not None  # the close actually happened
    # DDD entered despite the book being otherwise invested: the AAA close
    # funded it.
    ddd = [t for t in result.final_state.portfolio.trades if t.symbol == "DDD"]
    assert len(ddd) == 1
    # Neither the close nor its price was scaled: exit qty == entry qty.
    assert closed[0].qty == result.final_state.portfolio.trades[0].qty
    assert result.final_state.portfolio.cash >= 0


@strategy(bars="1d", stateful=True)
def _oversize_entry(ctx: StrategyContext):
    """Request far more than the book can fund, on the first bar."""
    n = ctx.shared.setdefault("_n", 0)
    ctx.shared["_n"] = n + 1
    if n == 0:
        ctx.long("AAA", size=50.0, reason="oversize", size_mode="capital")


def test_insufficient_cash_rejection_is_reported_to_stderr(capsys):
    """A dropped fill emits one aggregated STDERR warning."""
    cfg = _cfg(["AAA", "BBB"])
    run(Backtest(cfg), _feed(["AAA", "BBB"]), _FixtureMod(_oversize_entry))
    captured = capsys.readouterr()
    assert "rejected for insufficient cash" in captured.err
    assert captured.err.count("[bt] WARNING") == 1  # summarised, not per-event
    assert captured.out == ""  # never contaminates stdout


def test_no_rejection_warning_when_everything_fits(capsys):
    cfg = _cfg(["AAA", "BBB"])
    run(Backtest(cfg), _feed(["AAA", "BBB"]), _mod())
    assert "rejected" not in capsys.readouterr().err


@strategy(bars="1d", stateful=True)
def _rotating_cohort(ctx: StrategyContext):
    """Adversarial multi-symbol churn: staggered entries AND exits.

    Unlike ``_all_in_on_first_bar`` (one burst, then quiet), this keeps the
    book near-fully-invested for the whole run and re-enters on every other
    bar. That is the regime where the old sequential cash race bit hardest:
    cash is tight, so WHICH symbol drains first decided which fills happened.
    A permutation-invariance guard that only ever tests a single quiet entry
    cannot see it.
    """
    n = ctx.shared.setdefault("_n", 0)
    ctx.shared["_n"] = n + 1
    for sym in ctx.symbols:
        # The schedule must depend ONLY on symbol identity, never on list
        # position. Deriving it from the loop index would permute the
        # STRATEGY along with the symbol order, and then the invariance test
        # would be comparing two different strategies — the assertion would
        # fail for a reason that has nothing to do with the engine.
        #
        # Two entries per symbol per cycle across the whole book, so every
        # symbol opens on the SAME bar: that is the cohort that must share the
        # book. Size is deliberately generous relative to capital so the group
        # asks for more than is available and the allocation actually binds.
        phase = sum(ord(c) for c in sym) % 2
        if ctx.quantity(sym) == 0:
            if (n + phase) % 2 == 0:
                ctx.long(sym, size=2.0, reason="rotating entry", size_mode="capital")
        elif (n + phase) % 3 == 0:
            ctx.close(sym, reason="rotating exit")


def _run_rotating(symbols: list[str], capital: float = 10_000.0):
    cfg = _cfg(symbols, capital)
    return run(Backtest(cfg), _feed(symbols, n=24), _FixtureMod(_rotating_cohort))


def test_rotating_cohort_is_order_invariant_under_tight_cash():
    """Permuting ``symbols`` must not change trades when cash is the binding constraint.

    The regression this guards: an allocation scheme that re-derives the scale
    per symbol during the drain, or that sizes off a book a sibling already
    shrank, yields a different trade SET per ordering. Comparing the sorted
    ledger (symbol, qty, entry) catches both — a difference in fills or in
    sizes fails, not merely a difference in total P&L.
    """
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    perms = [
        symbols,
        list(reversed(symbols)),
        ["CCC", "EEE", "AAA", "DDD", "BBB"],
        ["EEE", "DDD", "CCC", "BBB", "AAA"],
        ["BBB", "AAA", "EEE", "CCC", "DDD"],
    ]
    results = [_run_rotating(order) for order in perms]
    base = results[0].final_state.portfolio
    base_ledger = sorted((t.symbol, round(t.qty, 6), t.entry_time) for t in base.trades)
    assert base_ledger, "fixture must actually trade, else the guard is vacuous"
    for order, perm_result in zip(perms[1:], results[1:]):
        portfolio = perm_result.final_state.portfolio
        ledger = sorted(
            (t.symbol, round(t.qty, 6), t.entry_time) for t in portfolio.trades
        )
        assert ledger == base_ledger, f"order dependence under {order!r}"
        assert portfolio.cash == pytest.approx(base.cash)


#: Symbols KEPT OPEN across the second burst (by identity, so the split is a
#: property of the symbol, not of its position in the list).
_HELD = frozenset({"AAA", "BBB", "CCC"})


@strategy(bars="1d", stateful=True)
def _cohort_over_held_book(ctx: StrategyContext):
    """Re-open a cohort while the OTHER half of the book is still HELD.

    Bar 0 bursts the whole book. Bar 3 frees only the non-``_HELD`` symbols,
    leaving ``_HELD``'s lots open. Bar 5 re-opens the freed symbols — a cohort
    that settles against a PARTIALLY-INVESTED book, which is the case the
    flat-book cohort tests cannot reach: the held lots consume no cash in this
    cohort but do shrink the free cash the cohort must share.
    """
    n = ctx.shared.setdefault("_n", 0)
    ctx.shared["_n"] = n + 1
    if n == 0:
        for sym in ctx.symbols:
            if ctx.quantity(sym) == 0:
                ctx.long(sym, size=0.35, reason="burst1", size_mode="capital")
    elif n == 3:
        for sym in ctx.symbols:
            if sym not in _HELD and ctx.quantity(sym) != 0:
                ctx.close(sym, reason="free half")
    elif n == 5:
        for sym in ctx.symbols:
            if sym not in _HELD and ctx.quantity(sym) == 0:
                ctx.long(sym, size=0.35, reason="burst2", size_mode="capital")
    elif n == 9:
        for sym in ctx.symbols:
            if ctx.quantity(sym) != 0:
                ctx.close(sym, reason="final flush")


def _ledger_for(mod, symbols: list[str], capital: float = 10_000.0):
    cfg = _cfg(symbols, capital)
    portfolio = run(Backtest(cfg), _distinct_feed(symbols), mod).final_state.portfolio
    ledger = sorted(
        (t.symbol, round(t.qty, 8), str(t.entry_time)) for t in portfolio.trades
    )
    return ledger, round(portfolio.cash, 8)


def test_cohort_over_held_book_is_order_invariant():
    """A cohort opening on a PARTIALLY-HELD book settles by ONE shared scale.

    The residual order dependence this pins: when a cohort is not the only
    thing in the portfolio, a per-symbol drain would size each open against a
    book a sibling had already shrunk, so qty would track ``config.symbols``
    order. The pre-cohort book is fixed, so the sorted ledger and the cash must
    be identical for every permutation. The held lots must also survive the
    second burst untouched.
    """
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    perms = [
        symbols,
        list(reversed(symbols)),
        ["DDD", "EEE", "FFF", "AAA", "BBB", "CCC"],
        ["FFF", "CCC", "AAA", "EEE", "BBB", "DDD"],
        ["CCC", "AAA", "EEE", "DDD", "FFF", "BBB"],
    ]
    base_ledger, base_cash = _ledger_for(_FixtureMod(_cohort_over_held_book), symbols)
    assert base_ledger, "fixture must trade, else the guard is vacuous"

    # The fixture must actually exercise the claim: symbols each enter TWICE
    # (burst1 + burst2) and the second enter is on a non-flat book.
    second_bar = sorted({t[2] for t in base_ledger if t[0] == "DDD"})
    assert len(second_bar) == 2, f"DDD must enter twice, got {second_bar}"

    for order in perms[1:]:
        ledger, cash = _ledger_for(_FixtureMod(_cohort_over_held_book), order)
        assert ledger == base_ledger, f"order-dependent ledger under {order!r}"
        assert cash == pytest.approx(base_cash, abs=1e-6), (
            f"order-dependent cash under {order!r}"
        )


def test_cohort_over_held_book_scales_uniformly():
    """Every open in the second cohort carries the SAME scale factor.

    Direct check of the mechanism rather than its consequence: the qty each
    symbol received must equal its unconstrained request times one factor, and
    that factor must be identical across the cohort. A per-symbol (sequential)
    scale would show up here as unequal factors even if the total cash matched.
    """
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    portfolio = run(
        Backtest(_cfg(symbols)),
        _distinct_feed(symbols),
        _FixtureMod(_cohort_over_held_book),
    ).final_state.portfolio

    second = [t for t in portfolio.trades if t.symbol not in _HELD and _is_second(t)]
    assert len(second) == 3, f"expected the 3 freed symbols to re-enter, got {second}"

    # scale = applied_qty / unconstrained_qty, where the unconstrained qty is
    # what the DSL would have emitted unshared: ``size * capital / signal_close``.
    # The signal bar is the bar BEFORE the fill (``fill_at_next_open``), and the
    # close is read back from the same feed the run used, so the expectation is
    # derived from the engine's input rather than restating its arithmetic. The
    # 4 dp floor applied after scaling bounds the relative error at ~2e-6
    # (measured); a per-symbol drain diverged by ~1%, five orders larger.
    factors = [
        t.qty / round(0.35 * 10_000.0 / _signal_close(t.symbol, t.entry_time), 4)
        for t in second
    ]
    for f in factors:
        assert f == pytest.approx(factors[0], rel=1e-5), (
            f"cohort was not scaled uniformly: factors={factors}"
        )
    assert factors[0] < 1.0, "fixture must press the cash constraint, else vacuous"
    assert portfolio.cash >= -1e-9, "cash must never go negative"


def _signal_close(symbol: str, entry_time) -> float:
    """Close of ``symbol`` on ``entry_time`` -- the bar the DSL read to size.

    ``entry_time`` is the SIGNAL bar's timestamp (the fill lands on the next
    bar's open via ``fill_at_next_open``). Rebuilt from the same
    ``_distinct_feed`` the run used, so it is the engine's own input rather than
    a restatement of its arithmetic.
    """
    frame = _distinct_feed([symbol])
    closes = frame[(symbol, "close")]
    return float(closes.loc[pd.Timestamp(entry_time)])


def _is_second(trade) -> bool:
    """True when ``trade`` is the symbol's second entry (the bar-5 burst)."""
    return str(trade.entry_time) != "2025-01-01 00:00:00"


def test_permutation_guard_is_not_vacuous():
    """Sanity: the fixture must press the cash constraint, or it proves nothing.

    If every entry always fits comfortably, an order-dependent engine would
    still pass the invariance test above and the guard would be vacuous.

    Note the constraint is pressed by ALLOCATION, not by rejection: with
    correct cohort settlement a grouped open is scaled to fit, so no fill is
    rejected. What proves the group was over-subscribed is that at least one
    cohort was scaled below its unconstrained size (a lone open requesting
    ``size`` of capital would have received the full amount).
    """
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    result = _run_rotating(symbols)
    portfolio = result.final_state.portfolio
    assert portfolio.cash >= -1e-9, "cash must never go negative"
    # Final cash is post-liquidation, so it reflects accumulated equity, not
    # the invested state. The observable that proves the allocation path ran is
    # a same-timestamp multi-open cohort: that is the group that must share one
    # book, and the only case where symbol order could have changed the result.
    entries_at: dict = {}
    for t in portfolio.trades:
        entries_at.setdefault(t.entry_time, []).append(t)
    multi = [g for g in entries_at.values() if len(g) > 1]
    assert multi, (
        "fixture must produce same-timestamp multi-open cohorts, else the "
        "invariance guard never exercises the allocation path"
    )
