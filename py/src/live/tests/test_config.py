"""Tests for the repo's ``config.toml`` reader (``src.config``)."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from src import config


def _reload_with(path: Path, monkeypatch: pytest.MonkeyPatch):
    """Reload ``src.config`` pointed at *path* (the path is read at import)."""
    monkeypatch.setenv("IBKR_CONFIG_PATH", str(path))
    return importlib.reload(config)


def test_live_adapter_reads_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "config.toml"
    target.write_text('[live]\nadapter = "sim"\n')
    module = _reload_with(target, monkeypatch)
    try:
        assert module.live_adapter() == "sim"
    finally:
        monkeypatch.delenv("IBKR_CONFIG_PATH")
        importlib.reload(config)


def test_live_adapter_defaults_when_the_file_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _reload_with(tmp_path / "missing.toml", monkeypatch)
    try:
        assert module.live_adapter() == module.DEFAULT_LIVE_ADAPTER == "ibkr"
    finally:
        monkeypatch.delenv("IBKR_CONFIG_PATH")
        importlib.reload(config)


def test_live_adapter_defaults_when_the_table_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "config.toml"
    target.write_text("[backtest]\nfoo = 1\n")
    module = _reload_with(target, monkeypatch)
    try:
        assert module.live_adapter() == "ibkr"
    finally:
        monkeypatch.delenv("IBKR_CONFIG_PATH")
        importlib.reload(config)


def test_a_typo_in_the_adapter_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "config.toml"
    target.write_text('[live]\nadapter = "paper-trader"\n')
    module = _reload_with(target, monkeypatch)
    try:
        with pytest.raises(ValueError, match="live.adapter must be"):
            module.live_adapter()
    finally:
        monkeypatch.delenv("IBKR_CONFIG_PATH")
        importlib.reload(config)


def test_a_non_table_section_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "config.toml"
    target.write_text('live = "ibkr"\n')
    module = _reload_with(target, monkeypatch)
    try:
        with pytest.raises(ValueError, match="must be a table"):
            module.live_adapter()
    finally:
        monkeypatch.delenv("IBKR_CONFIG_PATH")
        importlib.reload(config)
