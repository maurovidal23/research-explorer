"""Textual terminal interface for Research Explorer.

Textual is only imported by :mod:`research_explorer.tui.app`; event models,
navigation derivation, and projection live in :mod:`research_explorer.events`
and stay UI-independent.
"""

from research_explorer.tui.app import ResearchTUIApp, build_app
from research_explorer.tui.controller import TUIController

__all__ = ["ResearchTUIApp", "TUIController", "build_app"]
