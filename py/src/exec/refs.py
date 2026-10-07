"""The legacy, CYCLE-ANCHORED order-ref scheme — backtest/sim only now.

These helpers previous defined the ref for BOTH sides and are still exported for
compat and by ``src.exec.tests.test_refs``, but the LIVE path does not use them:
``src.live.identity.order_ref`` mints the live ref ``scope_tag-token-attempt``,
bar-free, so adoption survives bars, days and reruns (INV-2 is the INVERSE of
what the cycle-anchored scheme here assumes).

A ref here is ``f"{slug(scope)}-{cycle_ts:%Y%m%dT%H%M%S}-{seq:03d}"`` (plan rev
4.1 §4). ``scope`` is a stable strategy identity that survives a config edit,
and its ``slug`` prefix is also the attribution key. ``seq`` is keyed on the
intent's **stable identity** (symbol + action + lot), not its batch position, so
no intent inherits another's ref on a shifted batch.

**Upgrade caveat.** Refs minted here (the old scheme) are NOT ``trades.is_ours``
under the new live ref — the scope segment no longer pins the bar timestamp, and
``is_ours``/``ref_is_ours`` match the new shape exactly. So pre-upgrade
``live_order_intent`` refs and pre-upgrade executions still inside the 7-day
trades window will NOT be attributed (and will not advance the book). A rolling
fresh cold-start after upgrade is the intended migration.
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
    """BACKTEST/SIM-ONLY: seq per identity, collision-broken deterministically.

    The live path never calls it — ``identity.order_ref`` keys the ref directly
    on the intent with an attempt counter, no batch seq. ``_seq_of`` is already
    stable across runs; on the (rare) collision of two identities mapping to the
    same seq, the *later* one in sorted identity order is bumped until free.
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
    """BACKTEST/SIM-ONLY stable order ref: ``<slug>-<YYYYMMDDTHHMMSS>-<seq>``.

    Bar-anchored, so it is the INV-2 scheme the live path REPLACED with the
    bar-free ``identity.order_ref``. Left for the simulator/matching seam whose
    ref is a pure in-process dedupe token (no cross-cycle adoption), and for the
    test ``test_refs``. A live reader should look at ``src.live.identity``.
    """
    return f"{slug(scope)}-{cycle_ts:%Y%m%dT%H%M%S}-{seq:03d}"
