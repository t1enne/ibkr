"""Typed strategy parameter dataclasses.

Each strategy defines a frozen dataclass with its parameters and defaults.
The engine instantiates these once from StrategyConfig.strategy_params dict,
and strategies receive a typed instance instead of a raw dict.

Usage in a strategy module:
    from dataclasses import dataclass
    from src.bt.strategies.types import StrategyParams

    @dataclass(frozen=True)
    class Params(StrategyParams):
        fast: int = 9
        slow: int = 14
        vol_window: int = 20

    def on_candle(state, candle, params: Params) -> list[TradeSignal]:
        ema_fast = ta.ema(closes, params.fast)  # typed, no .get()
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
from typing import Literal, TypeVar, cast

import pandas as pd

T = TypeVar("T", bound="StrategyParams")


@dataclass(frozen=True)
class StrategyParams:
    """Base class for typed strategy parameters.

    Provides from_dict() that extracts only declared fields,
    ignoring extras. Subclass with your strategy's parameters.

    Example:
        @dataclass(frozen=True)
        class Params(StrategyParams):
            fast: int = 9
            slow: int = 14
            vol_window: int = 20
            vol_multiplier: float = 1.5
    """

    @classmethod
    def from_dict(cls: type[T], d: dict) -> T:
        """Extract declared fields from a raw dict, filling defaults for missing keys.

        Extra keys in the dict (e.g., engine-injected 'symbols', 'rolling_window_size')
        are ignored — they're available via StrategyConfig directly.
        """
        field_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in field_names}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Plot spec — declarative, JSON-safe chart description a DSL module's optional
# module-level ``plot(ctx, params) -> PlotSpec`` returns. ``output.py`` splices
# it under each symbol/frame; the dashboard renders it. No plotly in here.
# ---------------------------------------------------------------------------

PlotStyle = Literal["line", "dots", "band", "histogram"]
MarkerKind = Literal["pivot_low", "pivot_high", "div_long", "div_short", "signal"]

# A sparse series is ``((ms_epoch, value), ...)`` — a tuple, not a list, so a
# PlotSpec stays hashable/frozen and serializes without a custom encoder.
SparseSeries = tuple[tuple[int, float], ...]


def ms(ts: pd.Timestamp | datetime | int) -> int:
    """Epoch milliseconds for ``ts`` (``int`` passes through).

    Milliseconds to match the candle DB's ``timestamp`` unit; an int is assumed
    to already be ms epoch.
    """
    if isinstance(ts, int):
        return ts
    return int(pd.Timestamp(ts).value // 1_000_000)


def sparse(s: pd.Series) -> SparseSeries:
    """``Series`` -> ``((ms_epoch, value), ...)`` with NaN rows dropped.

    NaN is dropped (not zero-filled) so a chart shows a genuine gap rather than
    a plotted spike to zero at the warmup head.
    """
    out: list[tuple[int, float]] = []
    for ts, val in s.items():
        fval = float(val)
        if fval == fval:  # NaN-safe without importing numpy for one check
            out.append((ms(cast("pd.Timestamp", ts)), fval))
    return tuple(out)


@dataclass(frozen=True)
class Overlay:
    """A line/dots series drawn on the price pane (or a named sub-pane)."""

    series: SparseSeries
    name: str
    style: PlotStyle = "line"
    panel: str = "price"


@dataclass(frozen=True)
class Panel:
    """A sub-row series under the price pane (e.g. an oscillator).

    ``hlines`` draws horizontal reference levels (MFI 20/80) on the same row.
    """

    series: SparseSeries
    name: str
    hlines: tuple[float, ...] = ()


@dataclass(frozen=True)
class Marker:
    """A single point glyph on the price pane (pivot / divergence / signal).

    No ``symbol`` field: ``plot(ctx, params)`` is called once per symbol and the
    payload nests the spec under that symbol's frame.
    """

    ts: int
    price: float
    kind: MarkerKind


@dataclass(frozen=True)
class Leg:
    """A divergence leg: the line joining the two pivots of an accepted pair.

    A ``Marker`` is one point; a leg is four numbers, so it needs its own type.
    """

    a_ts: int
    a_price: float
    b_ts: int
    b_price: float
    kind: Literal["div_long", "div_short"]


@dataclass(frozen=True)
class PlotSpec:
    """Declarative chart description returned by a module-level ``plot``.

    Serialized verbatim with ``dataclasses.asdict`` into the plot payload, so
    every field must be a JSON-native scalar/tuple.
    """

    overlays: tuple[Overlay, ...] = ()
    panels: tuple[Panel, ...] = ()
    markers: tuple[Marker, ...] = ()
    legs: tuple[Leg, ...] = ()
