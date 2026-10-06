"""Import-order regression: ``src.utils`` must import cleanly on its own.

``src.utils`` imports ``src.data.resample``, which runs ``src.data.__init__``.
Before the leaf split that chain reached ``src.data._shared`` and
``src.data.fundamentals.normalize``, each importing back into the partially
initialised ``src.utils`` — an ``ImportError`` that fired ONLY when ``src.utils``
was imported first (``uv run pytest src/exec``), so ``make check`` masked it
behind import order. A fresh interpreter is the only honest reproduction: inside
this process ``src.utils`` is already imported, so the cycle cannot fire.

Chosen over a nested ``pytest src/exec`` subprocess because it is the minimal
reproduction of the exact failing order and does not pay for a second full
pytest session.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_import_src_utils_first_does_not_cycle() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.utils; "
            "from src.data._shared import to_optional_ts; "
            "from src.data.fundamentals.normalize import parse_timestamp; "
            "print(to_optional_ts('2024-01-01'), parse_timestamp('2024-01-01'))",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
