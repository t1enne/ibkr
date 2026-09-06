"""Cup-size handle scaling in cup_handle_dsl.

Tests the two cup-size relationships wired into :func:`_find_handle`:

* width  - the handle search window is a fraction of the parent cup's bar
  width, so a cup short in bars cannot reach a far-distant handle retrace.
* depth  - the allowed handle retrace is capped to a fraction of the cup's
  price depth, so a shallow bowl tolerates only a shallow handle.

Direct pure-function tests (no detector walk) keep this fast/deterministic.
"""

from typing import Optional

from src.bt.strategies.cup_handle_dsl import _find_handle, Handle, Swing


def _swing(idx: int, high: bool, level: float) -> Swing:
    return Swing(idx=idx, high=high, level=level)


def _find(
    lows: list[Swing],
    right: Swing,
    n: int,
    cup_bars: int,
    cup_depth: float,
    handle_width_scale: float,
    handle_depth_scale: float,
) -> Optional[Handle]:
    return _find_handle(
        lows,
        right,
        n,
        max_handle_bars=40,
        max_handle_drop_pct=0.12,
        handle_width_scale=handle_width_scale,
        handle_depth_scale=handle_depth_scale,
        cup_bars=cup_bars,
        cup_depth=cup_depth,
        handle_width_floor=3,
    )


def test_width_scale_limits_handle_horizon() -> None:
    """Wide cup harvests a far retrace; a short cup only sees the near one."""
    right = _swing(10, True, 100.0)
    lows = [_swing(12, False, 96.0), _swing(20, False, 90.0)]

    # wide cup (100 bars) -> window 25 -> it reaches the deeper far low.
    wide = _find(lows, right, 30, 100, 50.0, 0.25, 0.0)
    assert wide is not None and wide.low_idx == 20

    # short cup (8 bars) -> window max(3, 2) = 3 -> only the near low at
    # idx12 is searched; the far deep low at idx20 is out of scope.
    short = _find(lows, right, 30, 8, 5.0, 0.25, 0.0)
    assert short is not None and short.low_idx == 12


def test_depth_scale_ties_handle_retrace_to_bowl() -> None:
    """Same handle geometry fits a deep bowl but is rejected on a shallow one."""
    right = _swing(10, True, 100.0)
    lows = [_swing(14, False, 97.0), _swing(20, False, 96.0)]  # low ~ -4%

    # deep bowl: 15% * 50/100 = 7.5% cap >= 4% drop -> accepted
    deep = _find(lows, right, 30, 50, 50.0, 0.0, 0.15)
    assert deep is not None

    # shallow bowl: 15% * 10/100 = 1.5% cap < 4% drop -> rejected
    shallow = _find(lows, right, 30, 50, 10.0, 0.0, 0.15)
    assert shallow is None


def test_scales_disabled_matches_legacy_fixed_cap() -> None:
    """Scales off => the depth gate is just max_handle_drop_pct (12%)."""
    right = _swing(10, True, 100.0)
    lows = [_swing(14, False, 97.0), _swing(20, False, 96.0)]
    h = _find(lows, right, 30, 50, 5.0, 0.0, 0.0)
    assert h is not None and h.low_idx == 20  # saucer still allows the -4% pull
