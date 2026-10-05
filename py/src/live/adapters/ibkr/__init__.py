"""IBKR read path (plan phase 2): client/gateway wiring, mapping, trade replay.

Read-only. Order placement, LMT and stops are phase 3/4 and are deliberately
absent here — ``--adapter ibkr`` refuses to run without ``--dry-run``.
"""

from src.live.adapters.ibkr.trades import BrokerSnapshot, Execution, ReplayedLot, replay

__all__ = ["BrokerSnapshot", "Execution", "ReplayedLot", "replay"]
