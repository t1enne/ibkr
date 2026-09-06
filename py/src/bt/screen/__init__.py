"""Screen module — surface a real strategy's current-bar manual-trade intent.

The ``screen`` command runs a strategy's own ``on_candle`` through the engine
(gathering every fresh ``TradeSignal`` via a signal observer before the engine's
``_finalize`` flattens the book) and projects each symbol's latest intent into a
ranked, plain-looking table. There is no separate scoring vocabulary: a screen
is a strategy config, run for its signals rather than its fills.
"""

from __future__ import annotations

from src.bt.screen.run_strategy import (
    Action,
    COMMON_COLS,
    Posture,
    ScreenRow,
    SignalCollector,
    common_metrics,
    run_screen_from_strategy,
)

__all__ = [
    "Action",
    "COMMON_COLS",
    "Posture",
    "ScreenRow",
    "SignalCollector",
    "common_metrics",
    "run_screen_from_strategy",
]
