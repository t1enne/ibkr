"""Thin shim: the gateway/candle sync lives in ``src.data.ibkr.sync`` (plan §5).

Same contract as before (no args, cron-safe, exit code forwarded), but no logic:
this delegates to ``uv run python -m src.data.ibkr.sync`` inside ``py/`` so the
project's own environment, dependencies and package layout are used — exactly the
way the previous script already shelled out to ``uv --directory py run``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PY_DIR = Path(__file__).resolve().parent / "py"


def main() -> int:
    result = subprocess.run(
        [
            "uv",
            "--directory",
            str(PY_DIR),
            "run",
            "python",
            "-m",
            "src.data.ibkr.sync",
            *sys.argv[1:],
        ]
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
