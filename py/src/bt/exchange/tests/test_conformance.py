"""Type-level conformance of the adapters to the shared ports (``ty``-checked).

These are static checks: ``ty`` fails if ``SimExchange`` drifts off the ``Broker``
or ``FillSurface`` contract, or if the pure matcher stops satisfying ``Exchange``.
The one runtime assertion pins the deliberate NON-conformance of
``SimExchange.match_bar`` to ``Exchange`` (it adds required friction args).
"""

from __future__ import annotations

import inspect
from typing import Optional

import pandas as pd

from src.bt.exchange import FillSurface, SimExchange
from src.exec.matching import match_bar as pure_match_bar
from src.exec.ports import Broker, Exchange
from src.exec.types import Fill, OrderRequest


class _PureMatcher:
    """Binds the frictionless module-level matcher to the ``Exchange`` port."""

    def match_bar(self, order: OrderRequest, bars: pd.DataFrame) -> Optional[Fill]:
        return pure_match_bar(order, bars)


def test_sim_exchange_satisfies_broker_and_fill_surface() -> None:
    broker: Broker = SimExchange()
    surface: FillSurface = SimExchange()
    assert broker is not None
    assert surface is not None


def test_pure_matcher_satisfies_exchange_port() -> None:
    exchange: Exchange = _PureMatcher()
    assert exchange is not None


def test_sim_match_bar_is_not_the_pure_exchange() -> None:
    # SimExchange.match_bar REQUIRES friction kwargs the Exchange port lacks, so
    # it is deliberately a different, adapter-shaped contract. This reddens if
    # someone drops those required args and makes it (falsely) conformant.
    pure = set(inspect.signature(Exchange.match_bar).parameters)
    sim = set(inspect.signature(SimExchange.match_bar).parameters)
    assert {"spread_bps", "slippage_bps", "commission_model"} <= sim - pure
