"""The single definition of the order-ref scheme, shared by both sides.

``SimExchange`` uses it for stable ids in tests; a real broker adapter uses it
as the order's client id (``cOID``), so a re-sent cycle re-uses the same ref and
the broker dedupes. The ref is deterministic in its inputs and contains no
prices or sizes, so a resized order is not silently treated as a new order.
"""

from __future__ import annotations

import pandas as pd


def order_ref(strategy_id: str, cycle_ts: pd.Timestamp, seq: int) -> str:
    """Stable order ref: ``<strategy8>-<YYYYMMDDTHHMM>-<seq3>``.

    ``strategy_id`` is truncated to 8 chars, ``cycle_ts`` formatted to the
    minute, and ``seq`` the zero-padded 3-digit intent index within the cycle.
    """
    return f"{strategy_id[:8]}-{cycle_ts:%Y%m%dT%H%M}-{seq:03d}"
