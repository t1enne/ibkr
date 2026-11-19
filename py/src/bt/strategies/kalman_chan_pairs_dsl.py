"""Chan (2013) Example 3.3 — the trading layer over the published Kalman filter.

The filter itself is :mod:`src.indicators.kalman.chan_pairs` (item-for-item the
reference recursion, already tested).  This module adds only the rules the
filter refuses to decide — its docstring says so: it emits ``beta``, the
innovation ``e`` and its prediction sd ``S``, and the trading rules (reference
items 9-12) are the caller's.

Rules, each mapped to the code that applies it (``reference.md`` §2 numbering):

  (6) No warm start, no lookback — the reference's only guard is "trade if
      days > 1".  The filter starts empty on the first bar the feed can see and
      its own ``ready`` flag (``bars > 1``) is the ONLY readiness gate: while
      ``not step.ready`` the strategy is flat.  There is no rolling window, no
      OLS seed and no ``warmup_bars`` param -- see :func:`_filter`.
  (9) Entry, and only when flat: trade the innovation against ``±k*S``.
      ``e < -k*S`` -> LONG the spread (buy ``y``, sell ``floor(beta*N)`` of
      ``x``); ``e > +k*S`` -> SHORT the spread (sell ``y``, buy ``beta*N`` of
      ``x``).  ``k`` is ``Params.k`` (published value 1).
 (10) Exit — the one AMBIGUOUS item in the published reproductions, so both
      variants are published here as ``exit_mode``:
        * ``"band"`` [S2][S1]: close a long once ``e > -k*S``, a short once
          ``e < +k*S``.
        * ``"zero"`` [S3]: close a long once ``e > 0``, a short once ``e < 0``.
      ``"band"`` is the default (two of the three sources).
 (11) Hedge ratio: ``beta_t`` is read off the filter AT ENTRY and the hedge leg
      is ``floor(beta_t * N)`` shares of ``x``.  **A position is never rehedged
      while it is open** -- the reference recomputes beta only when it opens, so
      this strategy touches the hedge leg exactly twice: at entry and at exit.
      Any beta drift between those two moments is deliberately ignored, which
      is what makes this a reproduction and not a delta-neutral overlay.
 (11) Sizing: the reference trades ``N = 2000`` shares of the ``y`` leg on
      $100,000 of equity.  That is a *notional fraction* of capital, not a
      share count we can copy (our prices are not EWC's), so ``Params.size`` is
      the 0..1 fraction of capital the ``y`` leg takes.  The hedge is then the
      published SHARE count, not a notional one: ``N = size*capital/price_y``
      shares of ``y`` and ``floor(beta*N)`` shares of ``x``, the latter emitted
      as the capital fraction ``floor(beta*N) * price_x / capital`` (see
      :func:`_leg_sizes`).  Sizing the hedge as the FRACTION ``size*beta``
      instead would hold ``beta`` times the ``y``-leg NOTIONAL, which is
      ``price_y/price_x`` times the reference's hedge -- a partly directional
      book, not the reference's spread.  ``size`` default 0.3 = the reference's
      own 2000 shares of ~$15 SLV on $100k.
 (12) Costs, fills, spread: none here.  The engine supplies them (next-open
      fills, ``commission`` per fill, ``spread_bps`` + ``slippage_bps``); the
      strategy never fakes a price.

Declared deviation -- fill timing.  The reference trades at the close of the
bar it decides on; our DSL intents fill at the NEXT bar's open.  So the share
counts are derived from the signal bar's close (``price_x``/``price_y`` and
``beta`` all come from that bar) and fill one open later.  Consequence: the
realised hedge/y gearing is ``beta * price_x/price_y`` measured at the SIGNAL
bar, not the ``floor(beta*N)`` of the fill bar -- the signal is exact, the
prices move in between.  Measured across the six filed runs, the realised
ratio of x-leg notional to y-leg notional at entry sits near
``beta*price_x/price_y`` and NOT near ``beta``; the run report carries the
median and range.  Cost, slippage and the next-open gap are the engine's
(item 12) -- the strategy never adjusts a size to them.

Two-level state lives in ``ctx.shared`` (a fresh dict per run/window): the
:class:`ChanPairsFilter` and the side of the open spread (``+1`` long the
spread, ``-1`` short it) recorded at entry, which the exit test needs and which
also proves nothing re-hedges in between.  Filter knobs (``price_domain``,
``ve``, ``ve_mode``) pass through unchanged from ``strategy_params``; the
defaults are the reference's (absolute prices, ``Ve = 1e-3`` absolute).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

from src.bt.strategies.dsl import StrategyContext, strategy
from src.bt.strategies.types import StrategyParams
from src.indicators.kalman.chan_pairs import (
    ChanPairsConfig,
    ChanPairsFilter,
    ChanPairsStep,
    PriceDomain,
    VeMode,
)

STRATEGY_TYPE = "kalman_chan_pairs_dsl"

# ctx.shared keys. The filter is strategy-owned (no engine model pipeline), so
# it must be minted once per run and fed exactly once per bar.
_KF_KEY = "chan_kf"
# Side of the open spread: +1 = long the spread (bought y, sold x), -1 = short.
# Set at entry, cleared at exit; an open position with no side is a bug.
_SIDE_KEY = "chan_side"

# Decimal places of the DSL's ``size`` -> share-count conversion (``dsl._emit``).
# Repeated here only to floor the hedge off the y count that will actually fill.
_QTY_DP = 4

ExitMode = Literal["band", "zero"]


@dataclass(frozen=True)
class Params(StrategyParams):
    """Reference rules as parameters. Every default is the published value."""

    # (x, y) = (regressor, observed leg) — the reference's (EWA, EWC) roles.
    # ``None`` falls back to the config's first two symbols, same order.
    pair: tuple[str, str] | None = None
    # Item 9: the band is k*S with the published coefficient k = 1.
    k: float = 1.0
    # Item 10: the ambiguous exit. "band" is the majority reading.
    exit_mode: ExitMode = "band"
    # Item 11: 0..1 fraction of capital for the y leg. The hedge is the share
    # count floor(beta*N), not this fraction times beta (see module docstring).
    size: float = 0.3
    # Filter knobs, verbatim (items 2/4): "absolute" is the reference.
    price_domain: PriceDomain = "absolute"
    ve: float = 1e-3
    ve_mode: VeMode = "absolute"


def _pair(ctx: StrategyContext, params: Params) -> tuple[str, str] | None:
    """Resolve (x, y): config option first, then the config's symbol order."""
    if params.pair is not None:
        x, y = params.pair
        return (x, y)
    symbols: Sequence[str] = ctx.symbols
    if len(symbols) >= 2:
        return (symbols[0], symbols[1])
    return None


def _filter(ctx: StrategyContext, params: Params) -> ChanPairsFilter:
    """The shared filter, minted once per run (item 6: no warm start)."""
    shared = ctx.shared
    kf = shared.get(_KF_KEY)
    if not isinstance(kf, ChanPairsFilter):
        kf = ChanPairsFilter(
            ChanPairsConfig(
                ve=params.ve,
                price_domain=params.price_domain,
                ve_mode=params.ve_mode,
            )
        )
        shared[_KF_KEY] = kf
    return kf


def _close_now(ctx: StrategyContext, sym: str) -> float:
    """Cursor-safe as-of close, NaN when the feed has no visible bar yet.

    ``ctx.price`` raises on an empty view, and the pair's two legs do not share
    a start date (SLV lists 2014-12-11, GDX 2014-11-14), so the early bars would
    crash the run instead of simply not being tradable.
    """
    view = ctx.ta.close(sym, ctx.interval)
    return view[-1] if len(view) > 0 else math.nan


def _exit_due(step: ChanPairsStep, side: float, k: float, mode: ExitMode) -> bool:
    """Item 10, both published variants. Strict comparisons, as published."""
    if mode == "band":
        # Long the spread closes when e > -k*S; short when e < +k*S.
        return (
            step.innovation > -k * step.s if side > 0 else step.innovation < k * step.s
        )
    # Zero exit: long closes when e > 0; short when e < 0.
    return step.innovation > 0.0 if side > 0 else step.innovation < 0.0


def _leg_sizes(
    capital: float, px: float, py: float, beta: float, size: float
) -> tuple[float, float]:
    """Item 11 as ``(y_fraction, x_fraction)`` of capital, hedged in SHARES.

    The reference trades ``N`` SHARES of ``y`` and hedges ``floor(beta*N)``
    SHARES of ``x`` -- a share ratio. The DSL only accepts capital fractions,
    so the share counts are derived here and the ``x`` fraction is back-solved
    from its own count::

        N      = round(size*capital / price_y, 4)   (the y leg's emitted qty)
        x      = floor(|beta| * N)                  (absolute count)
        x_frac = x * price_x / capital

    A notional hedge (``size*|beta|``) instead sells ``|beta|*size*capital/
    price_x`` shares, i.e. ``price_y/price_x`` times the published hedge: not
    the reference's spread whenever the legs trade at different prices.
    """
    assert capital > 0.0, "capital must be positive to size a spread"
    assert px > 0.0, "the x leg's price must be positive"
    assert py > 0.0, "the y leg's price must be positive"
    assert size > 0.0, "size must be positive (0..1 fraction of capital)"
    assert beta != 0.0, "a zero hedge ratio cannot be traded as a spread"
    # ``_QTY_DP`` is the DSL's own share-count rounding (``dsl._emit``), applied
    # here first so the hedge is floored off the y shares that actually fill.
    y_shares = round(size * capital / py, _QTY_DP)
    assert y_shares > 0.0, "size and capital must buy at least one y share"
    x_shares = math.floor(abs(beta) * y_shares)
    # A zero-share hedge would open the y leg naked, which is a directional bet
    # and not the reference's spread: refuse instead of silently un-hedging.
    assert x_shares >= 1.0, "beta*y_shares floors to zero shares -- no hedge"
    return (y_shares * py / capital, x_shares * px / capital)


def _open_spread(
    ctx: StrategyContext,
    x: str,
    y: str,
    side: float,
    beta: float,
    px: float,
    py: float,
) -> None:
    """Item 9 + 11: buy/sell ``y``, hedge ``floor(|beta|*N)`` shares of ``x``.

    ``px``/``py`` are the DECISION bar's closes -- the same prices ``beta`` was
    filtered on -- so the share counts are the reference's, derived from the
    bar that produced the signal (the fill happens at the next open; see the
    module docstring).
    """
    y_size, x_size = _leg_sizes(
        ctx.state.portfolio.initial_capital, px, py, beta, ctx.params.size
    )
    if side > 0.0:
        ctx.long(y, size=y_size, reason="chan e < -kS long spread")
        ctx.short(x, size=x_size, reason="chan e < -kS hedge")
    else:
        ctx.short(y, size=y_size, reason="chan e > +kS short spread")
        ctx.long(x, size=x_size, reason="chan e > +kS hedge")
    ctx.shared[_SIDE_KEY] = side


def _close_spread(
    ctx: StrategyContext, x: str, y: str, step: ChanPairsStep, mode: ExitMode
) -> None:
    reason = f"chan {mode} exit z={step.z:.3f}"
    ctx.close(y, reason=reason)
    ctx.close(x, reason=reason)
    ctx.shared.pop(_SIDE_KEY, None)


@strategy(stateful=True)
def on_candle(ctx: StrategyContext):
    params: Params = ctx.params
    pair = _pair(ctx, params)
    if pair is None:
        return
    x, y = pair

    px = _close_now(ctx, x)
    py = _close_now(ctx, y)
    # A bar where a leg has no candle yet (the two feeds do not share a start)
    # cannot advance the recursion: skip it rather than feed a NaN. The filter
    # counts only the bars it was actually given, so item 6's "days > 1" still
    # means "two observations of this pair".
    if not (math.isfinite(px) and math.isfinite(py)) or px <= 0.0 or py <= 0.0:
        return

    step = _filter(ctx, params).update(px, py)

    # Warmup bars fill the filter only (the DSL refuses entries there anyway).
    if ctx.phase == "warmup":
        return
    # Item 6: flat until the filter has more than one observation.
    if not step.ready:
        return

    side = ctx.shared.get(_SIDE_KEY)
    has_pos = ctx.position(x) is not None or ctx.position(y) is not None

    if has_pos:
        assert side is not None, "open spread without a recorded side"
        if _exit_due(step, side, params.k, params.exit_mode):
            _close_spread(ctx, x, y, step, params.exit_mode)
            # No same-bar re-entry: the position is flat from the next bar on,
            # so a flip can never happen inside one bar.
        return

    # Item 9: entry band, flat book only.
    if abs(step.innovation) <= params.k * step.s:
        return
    _open_spread(ctx, x, y, 1.0 if step.innovation < 0.0 else -1.0, step.beta, px, py)
