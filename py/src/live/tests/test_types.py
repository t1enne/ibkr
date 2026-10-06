"""Tests for the live domain types (the shared edge vocabulary)."""

from __future__ import annotations

from typing import get_args

from src.live.types import FeedKind, feed_error


def test_feed_error_preserves_every_declared_kind() -> None:
    # Guards the L3 hazard: a kind added to the ``FeedKind`` Literal but not to the
    # accepted set would silently degrade to "transport" here. Because the set is
    # DERIVED from the Literal, this can never drift.
    for kind in get_args(FeedKind):
        assert feed_error(kind, "m").kind == kind


def test_feed_error_degrades_an_unknown_kind_to_transport() -> None:
    assert feed_error("not-a-real-kind", "m").kind == "transport"
