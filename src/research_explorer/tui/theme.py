"""Semantic color and status tokens for the Convoy-style dashboard.

Colors mirror Convoy's restrained dark grammar: a dim border, one accent border
for the focused panel, and semantic status colors. Every status also carries a
text glyph and label so information is never color-only.
"""

from __future__ import annotations

from research_explorer.events.models import (
    AGENT_ACTIVE,
    AGENT_COMPLETED,
    AGENT_EVALUATING,
    AGENT_EXHAUSTED,
    AGENT_FAILED,
    AGENT_WAITING,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_EVALUATING,
    STATUS_FAILED,
    STATUS_INITIALIZING,
    STATUS_RUNNING,
)

COLOR_BG = "#1a1b26"
BORDER_DIM = "#3b4261"
BORDER_ACCENT = "#7aa2f7"
COLOR_ACCENT = "#7aa2f7"
COLOR_MUTED = "#565f89"
COLOR_TEXT = "#c0caf5"
COLOR_GOOD = "#9ece6a"
COLOR_WARN = "#e0af68"
COLOR_BAD = "#f7768e"
COLOR_INFO = "#7dcfff"
COLOR_CYAN = "#73daca"

STATUS_GLYPHS: dict[str, tuple[str, str]] = {
    STATUS_INITIALIZING: ("◌", COLOR_MUTED),
    STATUS_RUNNING: ("▶", COLOR_ACCENT),
    AGENT_ACTIVE: ("▶", COLOR_ACCENT),
    STATUS_EVALUATING: ("◐", COLOR_CYAN),
    AGENT_EVALUATING: ("◐", COLOR_CYAN),
    AGENT_WAITING: ("·", COLOR_MUTED),
    AGENT_EXHAUSTED: ("■", COLOR_WARN),
    AGENT_COMPLETED: ("✔", COLOR_GOOD),
    STATUS_COMPLETED: ("✔", COLOR_GOOD),
    STATUS_CANCELLED: ("⊘", COLOR_WARN),
    AGENT_FAILED: ("✖", COLOR_BAD),
    STATUS_FAILED: ("✖", COLOR_BAD),
    "converged": ("◆", COLOR_GOOD),
    "exhausted": ("■", COLOR_WARN),
    "pending": ("·", COLOR_MUTED),
    "active": ("▶", COLOR_ACCENT),
    "completed": ("✔", COLOR_GOOD),
    "failed": ("✖", COLOR_BAD),
    "skipped": ("⊘", COLOR_MUTED),
    "warning": ("!", COLOR_WARN),
}


def status_style(status: str) -> tuple[str, str]:
    return STATUS_GLYPHS.get(status, ("?", COLOR_MUTED))


SELECTION_MARK = "▸"
