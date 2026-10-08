"""Tests for src.data.resample — session-anchored intraday resampling."""

from __future__ import annotations


import pandas as pd

from src.data.resample import (
    _intraday_hours,
    resample_multiindex,
    resample_ohlcv,
)


def _hourly(day: str, hours: list[str]) -> pd.DataFrame:
    idx = pd.to_datetime([f"{day} {h}" for h in hours])
    n = len(idx)
    return pd.DataFrame(
        {
            "open": [10.0 + i for i in range(n)],
            "high": [11.0 + i for i in range(n)],
            "low": [9.0 + i for i in range(n)],
            "close": [10.5 + i for i in range(n)],
            "volume": [100.0 * (i + 1) for i in range(n)],
        },
        index=idx,
    )


def test_intraday_hours_helper() -> None:
    assert _intraday_hours("4h") == 4
    assert _intraday_hours("2h") == 2
    assert _intraday_hours("1h") is None  # native granularity, untouched
    assert _intraday_hours("1D") is None
    assert _intraday_hours("30min") is None


def test_4h_gives_exactly_two_bars_per_session() -> None:
    df = _hourly(
        "2024-01-02", ["14:30", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00"]
    )
    out = resample_ohlcv(df, "4h", completed_only=False)
    assert len(out) == 2
    # anchored at 14:30 -> buckets 14:30 and 18:30
    assert list(out.index) == [
        pd.Timestamp("2024-01-02 14:30"),
        pd.Timestamp("2024-01-02 18:30"),
    ]
    # bucket 0 = bars 14:30..18:00 (offsets 0..3.5h, rows 0-4);
    # bucket 1 = 19:00..20:00 (rows 5-6)
    assert out["open"].iloc[0] == 10.0
    assert out["high"].iloc[0] == 15.0
    assert out["low"].iloc[0] == 9.0
    assert out["close"].iloc[0] == 14.5
    assert out["volume"].iloc[0] == sum(100.0 * (i + 1) for i in range(5))
    assert out["open"].iloc[1] == 15.0
    assert out["close"].iloc[1] == 16.5
    assert out["volume"].iloc[1] == sum(100.0 * (i + 1) for i in range(5, 7))


def test_4h_two_bars_per_day_across_days_and_dst_shift() -> None:
    # summer session opens 13:30, winter 14:30 (UTC-naive data) -> 2 bars each
    summer = _hourly(
        "2024-07-01", ["13:30", "14:30", "15:30", "16:30", "17:30", "18:30", "19:30"]
    )
    winter = _hourly(
        "2024-01-02", ["14:30", "15:30", "16:30", "17:30", "18:30", "19:30", "20:30"]
    )
    out = resample_ohlcv(pd.concat([summer, winter]), "4h", completed_only=False)
    assert len(out) == 4
    per_day = out.groupby([t.date() for t in out.index]).size().to_dict()
    assert all(v == 2 for v in per_day.values())


def test_partial_session_yields_fewer_buckets() -> None:
    half_day = _hourly("2024-11-29", ["14:30", "15:30", "16:30"])  # early close
    out = resample_ohlcv(half_day, "4h", completed_only=False)
    assert len(out) == 1  # all three bars fall inside the first 4h bucket


def test_daily_resample_stays_calendar_aligned() -> None:
    df = pd.concat(
        [
            _hourly("2024-01-02", ["14:30", "16:00", "20:00"]),
            _hourly("2024-01-03", ["14:30", "16:00", "20:00"]),
        ]
    )
    out = resample_ohlcv(df, "1d", completed_only=False)
    assert len(out) == 2
    assert list(out.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


def test_empty_input() -> None:
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    assert resample_ohlcv(empty, "4h", completed_only=False).empty


def test_multiindex_uses_session_anchoring_per_symbol() -> None:
    a = _hourly("2024-01-02", ["14:30", "16:00", "18:00", "20:00"])
    b = _hourly("2024-01-02", ["14:30", "16:00", "18:00", "20:00"])
    a["symbol"] = "AAA"
    b["symbol"] = "BBB"
    df = pd.concat([a, b])
    df.index.name = "timestamp"
    df = df.set_index("symbol", append=True).reorder_levels(["symbol", "timestamp"])
    out = resample_multiindex(df, "4h", completed_only=False)
    counts = out.groupby(level="symbol").size().to_dict()
    assert counts == {"AAA": 2, "BBB": 2}
