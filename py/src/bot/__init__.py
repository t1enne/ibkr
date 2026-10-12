"""Unattended trading supervisors — registry, spawn and the resident loop.

No trading logic here: ``start``/``status``/``stop`` command the registry, and a
bot process is a resident ``sync -> cycle -> sleep`` loop (``loop.py``) whose
cycle it borrows from ``src.live.run_once``.
"""
