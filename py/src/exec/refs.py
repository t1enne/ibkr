"""The single definition of the order-ref scheme, shared by both sides.

``SimExchange`` uses it for stable ids in tests; a real broker adapter uses it
as the order's client id (``cOID``), so a re-sent cycle re-uses the same ref and
the broker dedupes. The ref is deterministic in its inputs and contains no
prices or sizes, so a resized order is not silently treated as a new order.
"""

from __future__ import annotations

import hashlib

import pandas as pd


def _identity_digest(strategy_id: str) -> str:
    """Short stable digest of the FULL strategy id (8 hex chars).

    Two strategies whose ids share a first-8-chars prefix would otherwise mint
    identical refs, letting a real broker silently dedupe a legitimate order.
    The digest of the whole id disambiguates them while staying deterministic.
    """
    return hashlib.blake2b(strategy_id.encode("utf-8"), digest_size=4).hexdigest()


def order_ref(strategy_id: str, cycle_ts: pd.Timestamp, seq: int) -> str:
    """Stable order ref: ``<strategy8>-<digest8>-<YYYYMMDDTHHMM>-<seq3>``.

    ``strategy_id`` is truncated to 8 chars for the human-readable prefix (kept
    first so ``startswith(strategy_id[:8])`` ownership scoping still holds), the
    full id's 4-byte digest disambiguates shared prefixes, ``cycle_ts`` is
    formatted to the minute, and ``seq`` the zero-padded 3-digit intent index
    within the cycle. No prices or sizes, so a resized order is the same order.
    """
    return (
        f"{strategy_id[:8]}-{_identity_digest(strategy_id)}-"
        f"{cycle_ts:%Y%m%dT%H%M}-{seq:03d}"
    )
