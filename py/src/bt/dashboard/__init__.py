"""Backtest dashboard — Streamlit viewer for ``ibkr bt run --plot`` payloads.

Public surface:
    launch_dashboard(payload) — write payload and serve the Streamlit app.

The render logic is pure (``render.py``); the Streamlit wiring lives in
``__main__.py`` and is only imported by the Streamlit process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any

from src.bt.cmds._shared import _json_default

__all__ = [
    "app_path",
    "launch_dashboard",
]


def app_path() -> str:
    """Absolute path to the Streamlit entry module."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "__main__.py")


def launch_dashboard(payload: dict[str, Any]) -> None:
    """Write ``payload`` to a temp file and serve the Streamlit dashboard."""
    tmp = tempfile.NamedTemporaryFile(
        "w", suffix=".json", prefix="bt_dash_", delete=False
    )
    with tmp as fh:
        json.dump(payload, fh, default=_json_default)
    sys.stderr.write(f"dashboard payload: {tmp.name}\n")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            app_path(),
            "--server.headless",
            "true",
            "--",
            tmp.name,
        ],
        check=False,
    )
