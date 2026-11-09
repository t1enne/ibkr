"""``ctx.macro`` — cursor-safe macro print series.

Methodology: write a tiny FRED two-column asset into a tmp directory
(``load_daily``'s exact on-disk contract), mint a ``Macro`` store against it,
bind a mutable cursor reader, and assert the lookahead rule at intraday and
daily bars, the memoised-series/cursor trap, ``rank`` over visible prints only,
and the loud failure modes. Each test states whether it is a regression guard
(fails a naive/mistaken implementation) or characterisation of new behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from src.bt.state import BacktestState, Candle
from src.bt.engine.candle_store import CandleStore, CandleRows
from src.bt.strategies.dsl import StrategyContext
from src.bt.strategies.macro_context import Macro, is_daily_or_larger
from src.bt.strategies.ta_context import TaContext


def _ts(s: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(s))


def _write_series(tmp: Path, stem: str, rows: list[tuple[str, float]]) -> None:
    """Write a FRED two-column asset (``date,<stem>``) the loader accepts."""
    body = "".join(f"{day},{value}\n" for day, value in rows)
    (tmp / f"{stem}.csv").write_text(body, encoding="utf-8")


def _macro(tmp: Path, clock: dict[str, pd.Timestamp], interval: str = "1h") -> Macro:
    """A store whose cursor is read live from ``clock`` (engine-style binding)."""
    return Macro(source_dir=tmp, cursor=lambda: clock["ts"], interval=interval)


# -- the lookahead rule ------------------------------------------------------


def test_intraday_bar_hides_same_day_print(tmp_path: Path) -> None:
    """REGRESSION: a print dated D is the day's CLOSE — invisible intraday at D."""
    _write_series(tmp_path, "vix", [("2020-01-02", 17.0), ("2020-01-03", 18.0)])
    clock = {"ts": _ts("2020-01-03 15:00")}
    vix = _macro(tmp_path, clock).vix

    assert len(vix) == 1
    assert vix.last() == 17.0
    assert vix.date == _ts("2020-01-02")


def test_weekend_forward_fill_cannot_leak_monday(tmp_path: Path) -> None:
    """REGRESSION: the FF'd grid must not carry Monday's print into Monday.

    Prints Friday 01-03 and Monday 01-06; at Monday intraday only Friday's value
    is known. An inclusive cutoff (``bisect_left`` at the day) returns 19.0
    here, and a snapshot-at-build view cannot answer this at all.
    """
    _write_series(tmp_path, "vix", [("2020-01-03", 18.0), ("2020-01-06", 19.0)])
    clock = {"ts": _ts("2020-01-06 15:00")}
    vix = _macro(tmp_path, clock).vix

    assert vix.last() == 18.0
    # The date is the FF'd grid row: Sunday, carrying Friday's print. Documented
    # behaviour, pinned so a rename to "release date" cannot pass unnoticed.
    assert vix.date == _ts("2020-01-05")

    clock["ts"] = _ts("2020-01-07 15:00")
    assert vix.last() == 19.0


def test_bound_but_unadvanced_cursor_sees_nothing(tmp_path: Path) -> None:
    """REGRESSION: bound cursor + no advance yet == empty, not the file's tail.

    Collapsing "no reader bound" and "reader returned None" exposes years of
    future prints before the first bar.
    """
    _write_series(tmp_path, "vix", [("2020-01-02", 17.0), ("2020-01-03", 18.0)])
    clock: dict[str, pd.Timestamp | None] = {"ts": None}
    macro = Macro(source_dir=tmp_path, cursor=lambda: clock["ts"], interval="1h")
    vix = macro.vix

    assert len(vix) == 0
    with pytest.raises(IndexError):
        vix.last()
    with pytest.raises(IndexError):
        _ = vix.date

    clock["ts"] = _ts("2020-01-03 15:00")
    assert vix.last() == 17.0


def test_unbound_cursor_serves_the_whole_series(tmp_path: Path) -> None:
    """Characterisation: no reader at all is the notebook path (everything visible)."""
    _write_series(tmp_path, "vix", [("2020-01-02", 17.0), ("2020-01-03", 18.0)])
    vix = Macro(source_dir=tmp_path).vix

    assert len(vix) == 2
    assert vix.last() == 18.0


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        ("1d", True),
        ("2d", True),
        ("1 day", True),
        ("1w", True),
        ("1mo", True),
        ("1y", True),
        ("1h", False),
        ("4h", False),
        ("30m", False),
        ("1min", False),
        ("", False),
        (None, False),
        ("daily", False),
    ],
)
def test_interval_grammar_only_widens_the_rule_when_it_must(
    interval: str | None, expected: bool
) -> None:
    """REGRESSION: an unrecognised unit must fall to the intraday (safe) arm."""
    assert is_daily_or_larger(interval) is expected


def test_cursor_regression_shrinks_visibility(tmp_path: Path) -> None:
    """REGRESSION: visibility is derived per read, so it can also shrink."""
    _write_series(
        tmp_path,
        "vix",
        [("2020-01-02", 17.0), ("2020-01-03", 18.0), ("2020-01-06", 19.0)],
    )
    clock = {"ts": _ts("2020-01-07 15:00")}
    vix = _macro(tmp_path, clock).vix
    # The FF'd grid spans 01-02..01-06: five rows, three prints.
    assert len(vix) == 5

    clock["ts"] = _ts("2020-01-03 15:00")
    assert len(vix) == 1
    assert vix.last() == 17.0


@pytest.mark.parametrize("interval", ["1d", "1w"])
def test_daily_or_larger_bucket_sees_same_day_print(
    tmp_path: Path, interval: str
) -> None:
    """REGRESSION: a completed daily bucket dated D carries D's close, so D shows."""
    _write_series(tmp_path, "dgs10", [("2020-01-02", 1.5), ("2020-01-03", 1.6)])
    clock = {"ts": _ts("2020-01-03")}
    series = _macro(tmp_path, clock, interval=interval).dgs10

    assert len(series) == 2
    assert series.last() == 1.6
    assert series.date == _ts("2020-01-03")


def test_advancing_the_cursor_only_ever_adds_visible_prints(tmp_path: Path) -> None:
    """REGRESSION: monotone visibility — no cursor step can hide a known print."""
    _write_series(
        tmp_path,
        "vix",
        [("2020-01-02", 17.0), ("2020-01-03", 18.0), ("2020-01-06", 19.0)],
    )
    clock = {"ts": _ts("2020-01-02 15:00")}
    vix = _macro(tmp_path, clock).vix

    seen = [len(vix)]
    for stamp in ("2020-01-03 15:00", "2020-01-06 15:00", "2020-01-07 15:00"):
        clock["ts"] = _ts(stamp)
        seen.append(len(vix))
    assert seen == sorted(seen)
    assert seen == [0, 1, 4, 5]


def test_memoised_series_rebinds_to_the_current_cursor(tmp_path: Path) -> None:
    """REGRESSION: the memoised-series trap — visibility is never snapshotted.

    One view object captured at an early bar must reflect every later cursor
    move; an implementation that caches the visible window at build time fails
    here.
    """
    _write_series(tmp_path, "vix", [("2020-01-02", 17.0), ("2020-01-03", 18.0)])
    clock = {"ts": _ts("2020-01-03 15:00")}
    macro = _macro(tmp_path, clock)
    vix = macro.vix
    assert (len(vix), vix.last()) == (1, 17.0)

    clock["ts"] = _ts("2020-01-06 15:00")
    assert (len(vix), vix.last()) == (2, 18.0)
    # The store hands out the same series identity-wise per name; the data is
    # memoised, only the cursor is re-read.
    assert macro.vix.last() == 18.0


# -- rank --------------------------------------------------------------------


def test_rank_counts_visible_prints_only(tmp_path: Path) -> None:
    """REGRESSION: ``rank`` is computed over the VISIBLE tail, never the full file."""
    # Full series: 10, 40, 20 on consecutive days (the loader forward-fills onto
    # a DAILY grid, so gaps would duplicate values). At 2020-01-03 intraday only
    # 10 and 40 are known, so the latest visible print is 40, not the file's last
    # (20) — an implementation reading the untruncated tail answers 0.5.
    _write_series(
        tmp_path,
        "vix",
        [("2020-01-01", 10.0), ("2020-01-02", 40.0), ("2020-01-03", 20.0)],
    )
    clock = {"ts": _ts("2020-01-03 15:00")}
    vix = _macro(tmp_path, clock).vix

    assert vix.last() == 40.0
    assert vix.rank(1) == 1.0
    clock["ts"] = _ts("2020-01-04 15:00")
    assert vix.last() == 20.0
    assert vix.rank(2) == 0.5  # 10 and 40 below 20 -> one of two
    assert vix.rank(1) == 0.0  # only the prior print (40) is above 20


def test_rank_needs_a_prior_visible_print(tmp_path: Path) -> None:
    """Characterisation: an undefined percentile raises instead of guessing."""
    _write_series(tmp_path, "vix", [("2020-01-02", 17.0)])
    clock = {"ts": _ts("2020-01-02 15:00")}
    vix = _macro(tmp_path, clock).vix

    assert len(vix) == 0
    with pytest.raises(IndexError):
        vix.rank(5)
    clock["ts"] = _ts("2020-01-03 15:00")
    with pytest.raises(IndexError):
        vix.rank(5)


def test_store_attachment_binds_cursor_and_base_interval(tmp_path: Path) -> None:
    """Characterisation: ``attach_macro`` wires the store cursor + base interval."""
    _write_series(tmp_path, "vix", [("2020-01-02", 17.0), ("2020-01-03", 18.0)])
    macro = Macro(source_dir=tmp_path)
    rows: CandleRows = {}
    store = CandleStore(rows)
    store.attach_macro(macro, "1h")
    store.advance(_ts("2020-01-03 15:00"))

    # 17.0 (not 18.0) pins the interval reaching the series: the intraday arm
    # hides the same-day print, the daily arm would show it.
    assert macro.vix.last() == 17.0


def test_blank_asset_yields_no_prints_without_reloading(tmp_path: Path) -> None:
    """REGRESSION: an all-blank asset memoises as empty (no endless reload loop)."""
    (tmp_path / "vix.csv").write_text("2020-01-02,\n2020-01-03,\n", encoding="utf-8")
    macro = Macro(source_dir=tmp_path)

    assert len(macro.vix) == 0
    with pytest.raises(IndexError):
        macro.vix.last()
    assert "vix" in macro._cache


# -- resolution + failure modes ---------------------------------------------


def test_unknown_stem_names_available_stems_and_fetch_script(tmp_path: Path) -> None:
    """REGRESSION: no silent empty series and no fallback for an unknown stem."""
    _write_series(tmp_path, "vix", [("2020-01-02", 17.0)])
    macro = Macro(source_dir=tmp_path)

    with pytest.raises(AttributeError) as exc:
        macro.dgs10
    message = str(exc.value)
    assert "dgs10" in message
    assert "vix" in message
    assert "fetch_macro_fred" in message


def test_missing_asset_file_names_the_fetch_script(tmp_path: Path) -> None:
    """REGRESSION: a listed stem whose file vanished raises FileNotFoundError."""
    _write_series(tmp_path, "vix", [("2020-01-02", 17.0)])
    macro = Macro(source_dir=tmp_path)
    assert macro.stems == ("vix",)  # listing memoised while the file existed
    (tmp_path / "vix.csv").unlink()

    with pytest.raises(FileNotFoundError, match="fetch_macro_fred"):
        macro.vix


def test_ctx_macro_raises_when_store_is_absent() -> None:
    """Characterisation: ``ctx.macro`` fails loud, never serves an empty store."""
    ctx = StrategyContext(
        state=cast(BacktestState, None),
        candle=cast(Candle, None),
        params=None,
        ta=cast(TaContext, None),
        symbols=("SPY",),
        interval="1h",
    )

    with pytest.raises(RuntimeError, match="ctx.macro"):
        ctx.macro
