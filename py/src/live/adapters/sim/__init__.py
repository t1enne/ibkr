"""The simulated live adapter (stateless, JSON-backed book, no gateway)."""

from src.live.adapters.sim.adapter import (
    SimAdapter,
    build_account_book,
    build_sim_adapter,
)
from src.live.adapters.sim.store import (
    SimBook,
    SimBookStore,
    resolve_sim_book_path,
    synthetic_conid,
)

__all__ = [
    "SimAdapter",
    "SimBook",
    "SimBookStore",
    "build_account_book",
    "build_sim_adapter",
    "resolve_sim_book_path",
    "synthetic_conid",
]
