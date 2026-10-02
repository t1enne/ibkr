"""Minimal ``Ok``/``Err`` result type — errors as values, no exceptions-as-flow.

A building block for the live edges (portfolio fetch, broker placement): a
failure is a value the caller inspects, not a raised exception, so the pure
reconcile core never sees an ``except``. ~15 lines, no dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E")


@dataclass(frozen=True)
class Ok(Generic[T, E]):
    """Success branch carrying ``value``."""

    value: T


@dataclass(frozen=True)
class Err(Generic[T, E]):
    """Failure branch carrying ``error``."""

    error: E


Result = Ok[T, E] | Err[T, E]
