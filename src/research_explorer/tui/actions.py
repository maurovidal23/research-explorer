"""Single action registry driving bindings, footer hints, palette, and help.

Keeping every user action in one list is what prevents the footer, the command
palette, and the shortcut help from drifting apart. The app generates its
Textual bindings from this registry and renders footer/palette/help from it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UIAction:
    """One user-triggerable action."""

    id: str
    method: str
    label: str
    key: str | None = None
    key_display: str = ""
    help: str = ""
    category: str = "General"
    footer: bool = False
    narrow: bool = False


ACTIONS: tuple[UIAction, ...] = (
    UIAction(
        "quit", "quit_flow", "Quit", "q", "q",
        "Exit, confirming when a run is still active", "Run", footer=True,
    ),
    UIAction(
        "cancel", "cancel_flow", "Cancel", "ctrl+c", "Ctrl+C",
        "Request graceful cancellation of the active run", "Run", footer=True,
    ),
    UIAction(
        "palette", "show_palette", "Commands", "ctrl+p", "Ctrl+P",
        "Open the command palette", "General", footer=True,
    ),
    UIAction(
        "help", "show_help", "Help", "question_mark", "?",
        "Open complete shortcut help", "General", footer=True,
    ),
    UIAction(
        "follow", "follow_live", "Follow", "home", "Home",
        "Restore follow-live mode and jump to the newest node", "Navigation",
    ),
    UIAction(
        "focus_left", "focus_left", "Focus tree", "left", "←",
        "Move focus to the execution tree", "Navigation", footer=True,
    ),
    UIAction(
        "focus_right", "focus_right", "Focus content", "right", "→",
        "Move focus to the content pane", "Navigation", footer=True,
    ),
    UIAction(
        "reader", "open_reader", "Reader", "v", "v",
        "Open the active tab in a fullscreen reader", "Content", footer=True,
    ),
    UIAction(
        "tab_research", "tab_research", "Research tab", "1", "1",
        "Select the Research tab", "Content",
    ),
    UIAction(
        "tab_paper", "tab_paper", "Paper tab", "2", "2",
        "Select the Paper tab", "Content",
    ),
    UIAction(
        "tab_evaluation", "tab_evaluation", "Evaluation tab", "3", "3",
        "Select the Evaluation tab", "Content",
    ),
    UIAction(
        "tab_events", "tab_events", "Events tab", "4", "4",
        "Select the Events tab", "Content",
    ),
    UIAction(
        "tab_next", "tab_next", "Next tab", "tab", "Tab",
        "Select the next content tab", "Content",
    ),
    UIAction(
        "tab_prev", "tab_prev", "Previous tab", "shift+tab", "Shift+Tab",
        "Select the previous content tab", "Content",
    ),
    UIAction(
        "evaluation", "view_evaluation", "Evaluation", "e", "e",
        "Open the detailed evaluation for the selected agent", "Views",
    ),
    UIAction(
        "frontier", "view_frontier", "Frontier", "f", "f",
        "Open the recorded decision-time frontier ranking", "Views",
    ),
    UIAction(
        "paper", "view_paper", "Paper", "p", "p",
        "Open the full structured analysis of the selected paper", "Views",
    ),
    UIAction(
        "narrative", "view_narrative", "Narrative", "n", "n",
        "Open the complete research line of the selected agent", "Views",
    ),
    UIAction(
        "events", "view_events", "Events", "l", "l",
        "Open the structured event log for the selected agent", "Views",
    ),
    UIAction(
        "metadata", "view_metadata", "Metadata", "r", "r",
        "Open run metadata and resolved configuration", "Views",
    ),
    UIAction(
        "outcome_filter", "cycle_event_outcome", "Outcome filter", "o", "o",
        "Cycle the Events tab outcome filter", "Filters",
    ),
    UIAction(
        "agent_filter", "cycle_event_agent", "Agent filter", "a", "a",
        "Cycle the Events tab agent filter", "Filters",
    ),
    UIAction(
        "events_older", "events_older", "Older events", "shift+left", "Shift+\u2190",
        "Page to older events in the bounded Events tab", "Filters",
    ),
    UIAction(
        "events_newer", "events_newer", "Newer events", "shift+right", "Shift+\u2192",
        "Page to newer events in the bounded Events tab", "Filters",
    ),
    UIAction(
        "toggle_panes", "toggle_panes", "Toggle panes", "t", "t",
        "Switch the visible pane on compact terminals", "Layout",
        footer=True, narrow=True,
    ),
)


def action_by_id(action_id: str) -> UIAction | None:
    for action in ACTIONS:
        if action.id == action_id:
            return action
    return None


def footer_hints(narrow: bool = False) -> str:
    visible = (
        ("toggle_panes", "reader", "palette", "help", "quit")
        if narrow
        else ("focus_left", "focus_right", "palette", "help", "quit")
    )
    hints = [
        f"{a.key_display or a.key} {a.label.lower()}"
        for a in ACTIONS
        if a.id in visible
    ]
    return " · ".join(hints)


def help_text() -> str:
    lines = ["Research Explorer — keyboard shortcuts", ""]
    categories: dict[str, list[UIAction]] = {}
    for action in ACTIONS:
        categories.setdefault(action.category, []).append(action)
    for category, actions in categories.items():
        lines.append(f"{category}")
        for action in actions:
            key = action.key_display or action.key or ""
            lines.append(f"  {key:<8} {action.help}")
        lines.append("")
    lines.append("Navigation")
    lines.append("  ↑/↓      Move through the execution tree")
    lines.append("  PgUp/PgDn Scroll the focused panel")
    lines.append("  Enter    Expand or collapse the selected tree node")
    lines.append("  Esc      Close the topmost view")
    return "\n".join(lines).rstrip()


def palette_commands() -> list[tuple[str, str, str]]:
    """(action_id, title, subtitle) rows for the command palette."""
    return [(a.id, a.label, a.help) for a in ACTIONS]
