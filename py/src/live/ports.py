"""Live ports not already defined beside their concrete types.

Most live seams live with the code that owns them: ``PortfolioSource``
(``src/live/portfolio_source.py``), ``LiveBroker`` (``src/live/broker.py``),
``PendingIntents`` (``src/live/identity.py``), ``SignalSource``
(``src/live/engine.py``). The one seam with no natural home is the **account
exposure oracle**: the IBKR broker must compare the account net against what the
ledger can account for before it opens, but the broker is an adapter and must not
import the sqlite ledger. So the Protocol lives here, at the port layer, and the
ledger stays behind it.

Deliberately this ONE port only — moving the other seams here would duplicate
definitions that already sit beside their implementations.
"""

from __future__ import annotations

from typing import Protocol


class BookExposure(Protocol):
    """What the ledger can account for on one conid, across ALL scopes.

    The IBKR edge reads it to cross-check an account net it did not derive from
    our own book: an OPEN is refused when the two disagree (``net_exposure`` is
    our durable rows; the account is the broker's truth). A read that cannot
    answer is a failure the caller treats as "unknown" and fails closed on, never
    a zero book — a zero book reads as "flat" and would re-open everything.
    """

    def net_exposure(self, conid: int) -> float:
        """Signed net quantity the ledger books on *conid* across every scope."""
        ...
