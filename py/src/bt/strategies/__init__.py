"""Strategy auto-discovery.

Scans src/bt/strategies/ for modules with on_candle + STRATEGY_TYPE.
No manual registration needed — drop a .py file with STRATEGY_TYPE
and on_candle() and it's available.

Typed params: if a strategy module defines a Params dataclass (subclass of
StrategyParams), the engine auto-instantiates it from strategy_params dict.
"""

from __future__ import annotations

import glob
import importlib
import os
from typing import Protocol, TYPE_CHECKING, cast, Any, Callable

if TYPE_CHECKING:
    from src.bt.strategies.types import PlotSpec, StrategyParams


class _StrategyModule(Protocol):
    """Shape of a discovered strategy module."""

    STRATEGY_TYPE: str
    __name__: str
    Params: type[StrategyParams] | None

    def on_candle(
        self, state: object, candle: object, params: object
    ) -> list[object]: ...

    def reset_global(self) -> None: ...


_strategy_registry: dict[str, _StrategyModule] | None = None


def _discover() -> dict[str, _StrategyModule]:
    """Scan strategies/ directory for strategy modules with STRATEGY_TYPE."""
    pkg_dir = os.path.dirname(__file__)
    pkg_name = "src.bt.strategies"

    registry: dict[str, _StrategyModule] = {}

    for path in sorted(glob.glob(os.path.join(pkg_dir, "*.py"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.startswith("_") or stem == "utils":
            continue

        mod = importlib.import_module(f".{stem}", package=pkg_name)

        if not (hasattr(mod, "on_candle") and hasattr(mod, "STRATEGY_TYPE")):
            continue

        strategy_type: str = getattr(mod, "STRATEGY_TYPE")
        if strategy_type in registry:
            existing = registry[strategy_type]
            raise ImportError(
                f"Duplicate STRATEGY_TYPE '{strategy_type}' in "
                f"{existing.__name__} and {mod.__name__}"
            )

        registry[strategy_type] = cast(_StrategyModule, mod)

    return registry


def init_strat(strat_name: str) -> _StrategyModule:
    """Look up a strategy module by its STRATEGY_TYPE.

    Modules are discovered on first call by scanning src/bt/strategies/*.py
    for modules with both on_candle() and STRATEGY_TYPE defined.
    """
    global _strategy_registry
    if _strategy_registry is None:
        _strategy_registry = _discover()

    try:
        return _strategy_registry[strat_name]
    except KeyError:
        available = sorted(_strategy_registry.keys())
        raise KeyError(
            f"Unknown strategy type '{strat_name}'. Available: {available}"
        ) from None


def resolve_params(
    strat_name: str,
    raw_params: dict,
) -> object:
    """Instantiate typed Params if strategy defines them, else return raw dict.

    ``strategy_params`` holds the strategy's own parameters (sizing, SL/TP,
    signal knobs). ``StrategyParams.from_dict`` extracts only declared fields,
    ignoring extra keys, and fills defaults for missing ones.
    """
    mod = init_strat(strat_name)
    params_cls = getattr(mod, "Params", None)
    if params_cls is None:
        return raw_params
    params = {**raw_params}
    return params_cls.from_dict(params)


def plot_fn_for(strat_name: str) -> Callable[..., "PlotSpec"] | None:
    """The strategy's optional module-level ``plot`` callable, or ``None``.

    Read off the module (not the adapter): a module defines ``plot`` *after* its
    decorated ``on_candle``, so at decoration time the attribute may not exist
    yet. A strategy without a ``plot`` returns ``None`` and the plot payload
    gains no ``plot`` key — keeping output byte-identical for every existing
    strategy.
    """
    return getattr(init_strat(strat_name), "plot", None)


# Lazy re-export of the DSL symbols. Imported on demand (functions only) so
# ``from src.bt.strategies.dsl import strategy`` stays available without forcing
# ``src.bt.strategies`` -> dsl -> ``src.bt.state`` -> engine -> backtest ->
# ``src.bt.strategies`` at package import time (a circular-init hazard).
_DSL_EXPORTS = {
    "strategy": "strategy",
    "StrategyContext": "StrategyContext",
    "SeriesView": "SeriesView",
    "OhlcvView": "OhlcvView",
    "TaContext": "TaContext",
}


def __getattr__(name: str) -> Any:
    if name in _DSL_EXPORTS:
        import importlib as _il

        mod = _il.import_module("src.bt.strategies.dsl")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "init_strat",
    "resolve_params",
    "plot_fn_for",
    "strategy",
    "StrategyContext",
    "SeriesView",
    "OhlcvView",
    "TaContext",
]
