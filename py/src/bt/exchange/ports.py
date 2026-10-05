"""The backtest engine's fill-surface port — what ``SimExchange`` must expose.

The engine composes a broker and drives it through the legacy parity surface:
signal/risk pricing, single-fill application, and atomic cohort settlement.
Those methods are bt-shaped (``TradeSignal``/``Candle``/``FillEvent``/
``PortfolioState``/``ExecutionParams`` and they return portfolio settlements),
so this port lives in the bt layer — the shared pure core ``src/exec`` must not
import ``src.bt`` (the same layering rule that keeps friction out of the pure
matcher).

``SimExchange`` implements ``FillSurface`` and the broker-agnostic ``Broker``
port (``src/exec/ports.py``); the pure matcher satisfies ``Exchange``. Phase 3
routes live through the ``src/live`` cycle instead, so this seam is sim-only for
now and ``Broker`` stays order-only.
"""

from __future__ import annotations

from typing import Protocol

from src.bt.portfolio.pure import FillRejection, ScaleRecord
from src.bt.state import (
    Candle,
    CommissionModel,
    ExecutionParams,
    FillEvent,
    PortfolioState,
    TradeSignal,
)
from src.bt.types import RiskEvent


class FillSurface(Protocol):
    """Price one signal/risk event, apply one fill, or settle a whole cohort."""

    def execute_signal(
        self, signal: TradeSignal, tick: Candle, params: ExecutionParams
    ) -> FillEvent: ...

    def execute_risk_event(
        self, event: RiskEvent, tick: Candle, params: ExecutionParams
    ) -> FillEvent: ...

    def apply_fill(
        self, portfolio: PortfolioState, fill: FillEvent
    ) -> PortfolioState: ...

    def settle_cohort(
        self,
        portfolio: PortfolioState,
        fills: tuple[FillEvent, ...],
        *,
        scale_cohorts: bool = True,
        commission_model: CommissionModel,
    ) -> tuple[PortfolioState, tuple[FillRejection, ...], tuple[ScaleRecord, ...]]: ...
