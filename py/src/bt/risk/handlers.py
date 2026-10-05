"""Injectable risk-check seam for the backtest engine.

Formerly ``src/bt/engine/handlers.py``; the execution half of that module became
``SimExchange`` (``src/bt/exchange``), leaving risk as its own seam here.
``RiskHandler`` holds the ``check_risk`` function so a test can swap it.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bt.risk.pure import check_risk
from src.bt.types import RiskCheckFn


@dataclass
class RiskHandler:
    """Handler for checking and executing risk events.

    ``check_risk`` is the ``RiskCheckFn`` protocol, so a test swaps in any
    object with the same contract rather than a concrete function.
    """

    check_risk: RiskCheckFn


def default_risk_handler() -> RiskHandler:
    """Create default risk handler with production functions."""
    return RiskHandler(check_risk=check_risk)
