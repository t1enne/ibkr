"""Determinism tests for the shared order-ref scheme (plan rev 4.1 §4)."""

from __future__ import annotations

import pandas as pd

from src.exec.refs import assign_seqs, order_ref, slug
from src.utils import parse_timestamp


def ts(value: str) -> pd.Timestamp:
    """``pd.Timestamp`` narrowed past its ``NaTType`` union (repo convention)."""
    return parse_timestamp(value)


TS = ts("2025-03-04 09:30:00")


def test_slug_is_broker_safe_and_lowercased() -> None:
    assert slug("Momentum Alpha!") == "momentum-alpha"
    assert slug("") == "scope"


def test_order_ref_format() -> None:
    assert order_ref("momentum_x", TS, 7) == "momentum-x-20250304T093000-007"


def test_order_ref_is_deterministic() -> None:
    assert order_ref("abcdefghij", TS, 0) == order_ref("abcdefghij", TS, 0)


def test_order_ref_uses_whole_slug_not_a_hash_slice() -> None:
    # Two scopes sharing a prefix must not collide: ownership is the whole slug.
    a = order_ref("momentum_alpha", TS, 0)
    b = order_ref("momentum_beta", TS, 0)
    assert a.startswith("momentum-alpha-") and b.startswith("momentum-beta-")
    assert a != b


def test_order_ref_sequence_is_disjoint() -> None:
    refs = {order_ref("strat", TS, seq) for seq in range(100)}
    assert len(refs) == 100


def test_order_ref_is_second_granular() -> None:
    assert order_ref("strat", TS, 1) != order_ref("strat", ts("2025-03-04 09:30:59"), 1)


def test_assign_seqs_is_stable_by_identity() -> None:
    ids = ["AAPL|long|", "AAPL|close|55", "MSFT|long|"]
    first = assign_seqs(ids)
    assert assign_seqs(ids) == first
    assert len(set(first)) == len(ids)


def test_assign_seqs_shifted_batch_keeps_the_open_ref() -> None:
    # A close filling drops it from the next cycle's batch. The open must keep
    # the SAME seq (and so the same cOID), not inherit the close's.
    both = ["AAPL|close|1", "AAPL|long|"]
    assert assign_seqs(["AAPL|long|"])[0] == assign_seqs(both)[1]


def test_assign_seqs_breaks_collisions_deterministically() -> None:
    # Force two identities onto the same base seq by monkeypatching is awkward;
    # instead assert the invariant holds for a big batch (all seqs distinct).
    ids = [f"S{i}|long|" for i in range(500)]
    seqs = assign_seqs(ids)
    assert len(set(seqs)) == len(seqs)


def test_assign_seqs_collision_is_input_order_independent() -> None:
    # ``X184|long|`` and ``X444|long|`` collide on base seq 17612 (crc32). The
    # bumped seq must follow SORTED identity order, so the assignment is a
    # function of the identity SET, not the input order — a shifted batch cannot
    # hand an intent another's seq (finding L2).
    a, b = "X184|long|", "X444|long|"
    assert assign_seqs([a])[0] == assign_seqs([b])[0]  # same base seq: collides
    forward = dict(zip([a, b], assign_seqs([a, b]), strict=True))
    reverse = dict(zip([b, a], assign_seqs([b, a]), strict=True))
    assert forward == reverse
    assert forward[a] < forward[b]  # the identity sorting later is bumped
