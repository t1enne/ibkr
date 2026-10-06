"""The single definition of the order-ref scheme, shared by both sides.

A ref is ``f"{slug(scope)}-{cycle_ts:%Y%m%dT%H%M%S}-{seq:03d}"`` (plan rev 4.1
§4). ``scope`` is a stable strategy identity that survives a config edit (the
live config's ``scope`` key, defaulting to the strategy name), and its ``slug``
prefix is ALSO the attribution key: a strategy recognises its own executions on a
shared account by that whole-slug prefix (never a slice of a hash).

``seq`` is keyed on the intent's **stable identity** (symbol + action + lot), not
its position in the batch. That is the fix for the shifted-batch trap: a close
filling removes an intent from the next cycle's batch, so a positional seq would
let the following open inherit the close's already-seen ref and be deduped away.
Keying on identity mints the same ref for the same intent across runs, and a
different one for a different intent, regardless of who else is in the batch.
"""

from __future__ import annotations

import re
import zlib

import pandas as pd


def slug(scope: str) -> str:
    """Broker-safe, lowercase prefix for *scope* (ref's attribution key).

    Non-alphanumerics collapse to ``-`` so the token is safe as an IBKR client
    id prefix. An empty/degenerate scope falls back to ``"scope"`` rather than
    minting an empty prefix (which would claim every order).
    """
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", scope).strip("-").lower()
    return cleaned or "scope"


def scope_tag(scope: str) -> str:
    """Broker-safe, NON-COLLAPSING ownership tag for *scope*: ``<slug>-<crc32>``.

    ``slug`` alone maps two distinct scopes such as ``momentum_v2`` and
    ``momentum-v2`` to one token, so both would share a single owner on a shared
    account (each booking the other's fills). Appending the raw scope's crc32 as
    a dashless hex tail makes the tag distinct for any two differing scope
    strings short of a crc32 collision, while staying broker-safe as a cOID
    prefix and keeping the readable ``slug`` ahead of it.
    """
    return f"{slug(scope)}-{zlib.crc32(scope.encode('utf-8')):08x}"


def _seq_of(identity: str) -> int:
    """Deterministic seq in ``[0, 99999]`` from an intent's stable identity.

    A hash (not a batch index): the same intent always gets the same seq, so a
    re-run of the same cycle re-mints the same ref and a batch whose membership
    changed cannot hand one intent another's ref. The space is wide enough that
    a within-cycle collision is negligible; :func:`assign_seqs` breaks any that
    does occur deterministically.
    """
    return zlib.crc32(identity.encode("utf-8")) % 100000


def assign_seqs(identities: list[str]) -> list[int]:
    """Seq per identity, collision-broken deterministically within the batch.

    ``_seq_of`` is already stable across runs; on the (rare) collision of two
    identities mapping to the same seq, the *later* one in sorted identity order
    is bumped until free — a function of the identity SET, so a re-run of the
    same set assigns identically.
    """
    assigned: dict[str, int] = {}
    taken: set[int] = set()
    # Iterate the SORTED identity set, not the input order: which identity is
    # bumped on a collision is then a function of the set alone, so a batch whose
    # membership or ordering changed still assigns the same seq to the same
    # intent. ``sorted(set(...))`` also gives duplicate identities one seq.
    for identity in sorted(set(identities)):
        seq = _seq_of(identity)
        while seq in taken:
            seq = (seq + 1) % 100000
        taken.add(seq)
        assigned[identity] = seq
    return [assigned[identity] for identity in identities]


def order_ref(scope: str, cycle_ts: pd.Timestamp, seq: int) -> str:
    """Stable order ref: ``<slug(scope)>-<YYYYMMDDTHHMMSS>-<seq:03d>``.

    ``cycle_ts`` is second-granular so two different cycles can never share a
    timestamp; no prices or sizes enter the ref, so a resized order is the same
    order and a re-sent cycle dedupes at the broker.
    """
    return f"{slug(scope)}-{cycle_ts:%Y%m%dT%H%M%S}-{seq:03d}"
