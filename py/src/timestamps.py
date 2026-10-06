"""Timestamp parsing — a dependency-free leaf module (pandas only).

Deliberately NOT under ``src.data``: ``src.utils`` imports this, and importing
ANY ``src.data.*`` module runs ``src.data.__init__`` (which reaches back into
``src.utils`` via ``_shared``). A helper that both ``src.utils`` and ``src.data.*``
must share therefore has to live outside ``src.data`` — otherwise
``src.utils -> src.data.resample -> src.data.__init__ -> ... -> src.data._shared
-> src.utils`` closes a cycle and importing ``src.utils`` first raises. ``src.utils``
re-exports these so existing importers are untouched.
"""

from __future__ import annotations

from typing import Union, cast

import pandas as pd


def to_optional_ts(value: str | None) -> pd.Timestamp | None:
    """Convert optional string to Optional[Timestamp]. NaT → None."""
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    assert isinstance(ts, pd.Timestamp), f"Expected Timestamp, got {type(ts)}"
    return ts


def parse_timestamp(value: Union[str, pd.Timestamp]) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            raise ValueError(f"Invalid timestamp: {value}")
        return value
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"Invalid timestamp: {value}")
    return cast(pd.Timestamp, timestamp)
