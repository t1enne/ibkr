"""Universe config: the optional ``exchange`` venue field."""

from __future__ import annotations

import json

import pytest

from src.data.symbols import load_universe_config


def _write(tmp_path, payload: object) -> str:
    path = tmp_path / "universe.json"
    path.write_text(json.dumps(payload))
    return str(path)


def test_exchange_is_loaded_uppercased(tmp_path):
    conf = load_universe_config(
        _write(tmp_path, {"symbols": ["eni.mi"], "exchange": "bvme"})
    )

    assert conf.symbols == ["ENI.MI"]
    assert conf.exchange == "BVME"


def test_exchange_defaults_to_none_for_a_us_universe(tmp_path):
    conf = load_universe_config(_write(tmp_path, {"symbols": ["AAPL", "MSFT"]}))

    assert conf.exchange is None


@pytest.mark.parametrize("bad", ["", "   ", 42, ["BVME"]])
def test_malformed_exchange_is_refused(tmp_path, bad):
    with pytest.raises(ValueError, match="exchange"):
        load_universe_config(_write(tmp_path, {"symbols": ["ENI.MI"], "exchange": bad}))
