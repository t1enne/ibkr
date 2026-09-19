"""The declarative plot spec: helpers, frozen-ness, JSON round-trip.

No DB, no engine. ``sparse``/``ms`` are the only computations here, so they get
the edge cases (NaN, tz, empty, int passthrough); the dataclasses get a
frozen-ness + ``asdict`` JSON check because ``output.py`` serializes them
verbatim with zero conversion (so they must be JSON-safe by construction).
"""

from __future__ import annotations

import dataclasses
import datetime
import json

import pandas as pd

import pytest

from src.bt.strategies.types import (
    Leg,
    Marker,
    Overlay,
    Panel,
    PlotSpec,
    ms,
    sparse,
)
from src.utils import parse_timestamp


# --- ms ----------------------------------------------------------------------


def test_ms_from_timestamp_datetime_and_int() -> None:
    ts = parse_timestamp("2024-01-02 00:00:00")
    assert ms(ts) == int(ts.value // 1_000_000)
    assert ms(datetime.datetime(2024, 1, 2)) == ms(ts)
    assert ms(1_700_000_000_000) == 1_700_000_000_000  # int passes through


# --- sparse ------------------------------------------------------------------


def test_sparse_drops_nan_and_uses_ms_index() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    s = pd.Series([100.0, float("nan"), 103.0], index=idx)
    out = sparse(s)
    assert out == ((ms(idx[0]), 100.0), (ms(idx[2]), 103.0))


def test_sparse_intraday_and_tz_aware_index() -> None:
    idx = pd.DatetimeIndex(["2024-01-02 09:30", "2024-01-02 10:30"]).tz_localize("UTC")
    out = sparse(pd.Series([1.5, 2.5], index=idx))
    assert len(out) == 2
    assert out[0][1] == 1.5 and out[1][1] == 2.5
    assert out[1][0] - out[0][0] == 3_600_000  # one hour apart, in ms


def test_sparse_empty_series() -> None:
    assert sparse(pd.Series([], dtype=float)) == ()


# --- dataclasses -------------------------------------------------------------


def test_plot_spec_is_frozen_and_defaults_empty() -> None:
    spec = PlotSpec()
    assert spec.overlays == () and spec.panels == () and spec.markers == ()
    assert spec.legs == ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(spec, "panels", (Panel(series=(), name="x"),))


def test_plot_spec_asdict_json_round_trip() -> None:
    spec = PlotSpec(
        overlays=(Overlay(series=((1, 2.0),), name="ma", style="dots"),),
        panels=(Panel(series=((1, 50.0),), name="MFI(14)", hlines=(20.0, 80.0)),),
        markers=(Marker(ts=1, price=2.0, kind="pivot_low"),),
        legs=(Leg(a_ts=1, a_price=2.0, b_ts=3, b_price=1.0, kind="div_long"),),
    )
    # ``dataclasses.asdict`` is exactly how output.py serializes the spec, so
    # the invariant is that its output is JSON-native and stable on reparsing.
    payload = dataclasses.asdict(spec)
    reparsed = json.loads(json.dumps(payload))
    assert json.loads(json.dumps(reparsed)) == reparsed
    assert reparsed["panels"][0]["hlines"] == [20.0, 80.0]
    assert reparsed["legs"][0]["kind"] == "div_long"
    assert reparsed["overlays"][0]["style"] == "dots"
