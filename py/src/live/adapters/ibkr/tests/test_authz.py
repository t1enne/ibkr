"""``authz`` mode guard matrix: paper/live x account x --allow-live x dry_run."""

from __future__ import annotations

import pytest

from src.live.adapters.ibkr.authz import authorize, is_paper_account
from src.live.result import Ok

PAPER = "DU1234567"
LIVE = "U7654321"


@pytest.mark.parametrize(
    ("account", "expected"),
    [(PAPER, True), (LIVE, False)],
)
def test_is_paper_account_keys_on_prefix(account: str, expected: bool) -> None:
    assert is_paper_account(account) is expected


def test_paper_config_against_live_account_hard_fails() -> None:
    result = authorize(mode="paper", account=LIVE, allow_live=True, dry_run=True)
    assert not isinstance(result, Ok)
    assert result.error.kind == "auth"
    assert "refuses" in result.error.message


def test_live_mode_against_live_account_requires_allow_live() -> None:
    result = authorize(mode="live", account=LIVE, allow_live=False, dry_run=True)
    assert not isinstance(result, Ok)
    assert result.error.kind == "auth"
    assert "--allow-live" in result.error.message


def test_live_mode_with_allow_live_is_permitted() -> None:
    result = authorize(mode="live", account=LIVE, allow_live=True, dry_run=True)
    assert isinstance(result, Ok)
    assert result.value is True  # the account IS live


def test_paper_account_is_permitted_whatever_the_mode() -> None:
    for mode in ("paper", "live"):
        result = authorize(mode=mode, account=PAPER, allow_live=False, dry_run=True)
        assert isinstance(result, Ok)
        assert result.value is False  # not a live account


def test_dry_run_does_not_waive_the_live_account_requirement() -> None:
    for dry_run in (True, False):
        result = authorize(mode="live", account=LIVE, allow_live=False, dry_run=dry_run)
        assert not isinstance(result, Ok)
