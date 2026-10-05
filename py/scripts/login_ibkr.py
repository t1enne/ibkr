"""Thin shim: the Playwright login lives in ``src.data.ibkr.login`` (plan §5).

The full CLI (``--username``/``--password``/``--mode``, ``.env`` loading, exit
code 1 on missing creds) is preserved unchanged; only the body moved, so the cron
entry ``uv run python scripts/login_ibkr.py`` keeps working.
"""

from __future__ import annotations

import sys
from pathlib import Path

# When running as uv run python scripts/login_ibkr.py, sys.path[0] is set to
# scripts/ rather than the project root. Insert the project root so that
# imports from src/ resolve correctly.
_proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_proj_root))

from src.data.ibkr.login import load_env, main  # noqa: E402

if __name__ == "__main__":
    load_env()
    main()
