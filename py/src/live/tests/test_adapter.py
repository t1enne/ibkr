"""Critical-path tests for adapter resolution: the vocabulary and its refusals.

No behaviour here places an order — these pin the CONFIG contract that decides
which backend a cycle trades through, where the failure mode is silent (a typo
falls back to the default adapter and trades on the wrong side).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import get_args

import pytest

from src import config as config_mod
from src.config import live_adapter
from src.live.adapter import resolve_adapter_name
from src.live.scope import AdapterName


def test_every_layer_shares_one_adapter_vocabulary() -> None:
    """The runtime list(s) and the typing definition cannot drift apart.

    ``src.config`` is a leaf and cannot import ``AdapterName``, so it carries the
    one legitimate duplication; this is what pins it. Timed-out staleness here is
    a name accepted by the CLI and rejected by the seam (or the reverse).
    """
    assert set(config_mod._ADAPTERS) == set(get_args(AdapterName))
    assert live_adapter() in get_args(AdapterName)


def test_unknown_adapter_in_the_config_raises() -> None:
    """A typo is a config error — never a silent fallback to the default."""
    with pytest.raises(ValueError, match="adapter must be one of"):
        resolve_adapter_name(None, {"adapter": "ibkrr"}, {})


def test_unknown_adapter_in_strategy_params_raises() -> None:
    """The back-compat ``broker`` key is validated in params too, not trusted."""
    with pytest.raises(ValueError, match="adapter must be one of"):
        resolve_adapter_name(None, {}, {"broker": "simm"})


def test_explicit_null_adapter_is_a_bad_value_not_an_unset_key() -> None:
    """``"adapter": null`` must not quietly become the config.toml default."""
    with pytest.raises(ValueError, match="explicit null"):
        resolve_adapter_name(None, {"adapter": None}, {})


def test_absent_adapter_key_falls_back_to_the_config_file() -> None:
    assert resolve_adapter_name(None, {}, {}) == live_adapter()


def test_cli_flag_wins_and_the_broker_key_is_the_back_compat_source() -> None:
    assert resolve_adapter_name("sim", {"adapter": "ibkr"}, {}) == "sim"
    assert resolve_adapter_name(None, {"adapter": "sim", "broker": "ibkr"}, {}) == "sim"
    assert resolve_adapter_name(None, {"broker": "sim"}, {"broker": "ibkr"}) == "sim"


def test_config_file_typo_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``[live] adapter`` is validated the same way the config key is."""
    path = tmp_path / "config.toml"
    path.write_text('[live]\nadapter = "smi"\n')
    monkeypatch.setenv("IBKR_CONFIG_PATH", str(path))
    reloaded = importlib.reload(config_mod)
    try:
        with pytest.raises(ValueError, match="live.adapter must be one of"):
            reloaded.live_adapter()
    finally:
        monkeypatch.delenv("IBKR_CONFIG_PATH")
        importlib.reload(config_mod)
