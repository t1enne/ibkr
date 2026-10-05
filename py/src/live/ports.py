"""Live ports not already defined beside their concrete types.

Most live seams live with the code that owns them: ``PortfolioSource``
(``src/live/portfolio_source.py``), ``LiveBroker`` (``src/live/broker.py``),
``Ledger`` (``src/live/ledger.py``), ``SignalSource`` (``src/live/engine.py``).
The one seam with no natural home is the **gateway**: the live cycle must ask
"is the broker session ready?" before it reads a book, and the thing that
answers (``IbkrGateway``) is an adapter the engine never imports. So the Protocol
lives here, at the port layer, and the adapter stays behind it.

Deliberately Gateway ONLY — moving the other ports here would duplicate
definitions that already sit beside their implementations.
"""

from __future__ import annotations

from typing import Protocol

from src.live.result import Result
from src.live.types import FeedError


class Gateway(Protocol):
    """The broker gateway session: a readiness probe and an ensure-ready step.

    Fails as a value (``Result[.., FeedError]``), never by raising: a gateway
    that is down is an edge failure the cycle reports, not an exception that
    kills the process. ``is_ready`` never mutates the session; ``ensure_ready``
    may (keepalive / login) and must be safe to call before every cycle.
    """

    async def is_ready(self) -> Result[bool, FeedError]: ...

    async def ensure_ready(self) -> Result[None, FeedError]: ...
