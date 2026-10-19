"""Critical-path tests for adapter resolution: the vocabulary and its refusals.

No behaviour here places an order — these pin the CONFIG contract that decides
which backend a cycle trades through, where the failure mode is silent (a typo
falls back to the default adapter and trades on the wrong side).
"""

from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path
from typing import get_args

import pytest

from src import config as config_mod
from src.config import live_adapter
from src.live.adapter import resolve_adapter_name
from src.live.scope import AdapterName
from src.live.types import LiveConfig

_SAMPLE_CFG: LiveConfig = LiveConfig(
    strategy_type="dummy",
    symbols=("SPY",),
    initial_capital=100_000.0,
    strategy_params={},
    bars=("1d",),
    warmup="1y",
)


def test_every_layer_shares_one_adapter_vocabulary() -> None:
    """The runtime list(s) and the typing definition cannot drift apart.

    ``src.config`` is a leaf and cannot import ``AdapterName``, so it carries the
    one legitimate duplication; this is what pins it. Timed-out staleness here is
    a name accepted by the CLI and rejected by the seam (or the reverse).
    """
    assert set(config_mod._ADAPTERS) == set(get_args(AdapterName))
    assert live_adapter() in get_args(AdapterName)


def test_cli_flag_wins_over_cfg_adapter() -> None:
    cfg = replace(_SAMPLE_CFG, adapter="ibkr")
    assert resolve_adapter_name("sim", cfg) == "sim"


def test_absent_cli_flag_uses_cfg_adapter() -> None:
    cfg = replace(_SAMPLE_CFG, adapter="sim")
    assert resolve_adapter_name(None, cfg) == "sim"


def test_unknown_cli_flag_raises() -> None:
    with pytest.raises(ValueError, match="adapter must be one of"):
        resolve_adapter_name("ibkrr", _SAMPLE_CFG)


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
