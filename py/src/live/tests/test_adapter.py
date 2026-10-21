"""Critical-path tests for adapter resolution: the precedence and its refusals.

No behaviour here places an order — these pin the CONFIG contract that decides
which backend a cycle trades through, where the failure mode is silent (a typo
falls back to the default adapter and trades on the wrong side).
"""

from __future__ import annotations

from pathlib import Path
from typing import get_args

import pytest

from src.config import LiveConfig, read_live_config
from src.live.options import LiveOptions, adapter_names, merge_live_options
from src.live.scope import AdapterName

_DEFAULT = LiveOptions()


def test_every_layer_shares_one_adapter_vocabulary() -> None:
    """The CLI's choice list and the typing definition cannot drift apart.

    The CLI builds ``click.Choice`` from :func:`adapter_names`, which derives
    from ``scope.AdapterName``: a name accepted by the flag but rejected by the
    merge (or the reverse) is the bug this pins.
    """
    assert set(adapter_names()) == set(get_args(AdapterName))


def test_code_default_is_used_when_nothing_is_configured() -> None:
    assert merge_live_options(None, None, LiveConfig(), _DEFAULT) == _DEFAULT


def test_unknown_cli_flag_raises() -> None:
    with pytest.raises(ValueError, match="adapter must be one of"):
        merge_live_options("ibkrr", None, LiveConfig(), _DEFAULT)


def test_config_file_typo_raises() -> None:
    """``[live] adapter`` is refused exactly like a bad flag."""
    with pytest.raises(ValueError, match="adapter must be one of"):
        merge_live_options(None, None, LiveConfig(adapter="smi"), _DEFAULT)


def test_every_option_takes_the_configured_value() -> None:
    """Each ``[live]`` key reaches its resolved field, with no cross-talk."""
    cfg = LiveConfig(
        adapter="sim", allow_oth=True, max_age=9, dry_run=True, format="json"
    )
    merged = merge_live_options(None, None, cfg, _DEFAULT)
    assert merged == LiveOptions(
        adapter="sim", allow_oth=True, max_age=9, dry_run=True, format="json"
    )


def test_each_flag_overrides_its_own_key_only() -> None:
    """Precedence is per FIELD: a flag set on every field wins, independently."""
    cfg = LiveConfig(
        adapter="sim", allow_oth=True, max_age=9, dry_run=True, format="json"
    )
    merged = merge_live_options(
        "ibkr", False, cfg, _DEFAULT, max_age=1, dry_run=False, fmt="text"
    )
    assert merged == LiveOptions(
        adapter="ibkr", allow_oth=False, max_age=1, dry_run=False, format="text"
    )


def test_unknown_configured_format_raises() -> None:
    """``[live] format`` is refused exactly like a bad ``--format``."""
    with pytest.raises(ValueError, match="format must be one of"):
        merge_live_options(None, None, LiveConfig(format="xml"), _DEFAULT)


def test_reader_rejects_bad_types_and_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The leaf rejects a wrong TYPE/VALUE eagerly; names are the merge's job.

    A zero gate would let every stale cycle through as fresh, so it is refused
    at read time rather than silently disabling the staleness check.
    """
    settings = tmp_path / "config.toml"
    monkeypatch.setattr("src.config.CONFIG_PATH", settings)

    settings.write_text("[live]\nmax_age = 0\n")
    with pytest.raises(ValueError, match="live.max_age must be a positive"):
        read_live_config()

    settings.write_text('[live]\ndry_run = "yes"\n')
    with pytest.raises(ValueError, match="live.dry_run must be true/false"):
        read_live_config()

    settings.write_text("[live]\nformat = 1\n")
    with pytest.raises(ValueError, match="live.format must be a string"):
        read_live_config()
