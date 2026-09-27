"""Transient UI-only session state for the research dashboard.

Selection, expansion, focus, the active content tab, scroll positions, and
event filters are interactive state. They are deliberately *not* durable run
events: replay reconstructs them from scratch and the engine never observes
them. Keeping them in one dataclass keeps live rendering and replay symmetric.
"""

from __future__ import annotations

from dataclasses import dataclass, field

TREE_FOCUS = "tree"
CONTENT_FOCUS = "content"

TAB_RESEARCH = "research"
TAB_PAPER = "paper"
TAB_EVALUATION = "evaluation"
TAB_EVENTS = "events"

PRIMARY_TABS: tuple[str, ...] = (TAB_RESEARCH, TAB_PAPER, TAB_EVALUATION, TAB_EVENTS)


def tab_label(tab: str) -> str:
    return {
        TAB_RESEARCH: "Research",
        TAB_PAPER: "Paper",
        TAB_EVALUATION: "Evaluation",
        TAB_EVENTS: "Events",
    }.get(tab, tab.title())


def next_tab(tab: str, step: int = 1) -> str:
    index = PRIMARY_TABS.index(tab) if tab in PRIMARY_TABS else 0
    return PRIMARY_TABS[(index + step) % len(PRIMARY_TABS)]


EVENT_OUTCOMES: tuple[str, ...] = ("", "error", "warning", "info")


def next_event_outcome(current: str) -> str:
    index = EVENT_OUTCOMES.index(current) if current in EVENT_OUTCOMES else 0
    return EVENT_OUTCOMES[(index + 1) % len(EVENT_OUTCOMES)]


EVENT_AGENT_SCOPE = "scope"
EVENT_AGENT_ALL = "all"
EVENT_AGENT_WINNER = "winner"

EVENT_AGENT_SCOPES: tuple[str, ...] = (
    EVENT_AGENT_SCOPE,
    EVENT_AGENT_ALL,
    EVENT_AGENT_WINNER,
)


def next_event_agent(current: str) -> str:
    index = EVENT_AGENT_SCOPES.index(current) if current in EVENT_AGENT_SCOPES else 0
    return EVENT_AGENT_SCOPES[(index + 1) % len(EVENT_AGENT_SCOPES)]


@dataclass
class UISession:
    """Interactive state that is never persisted as a run event."""

    active_tab: str = TAB_RESEARCH
    focus: str = TREE_FOCUS
    selected_node_id: str | None = None
    selected_agent_id: str = ""
    follow_live: bool = True
    expanded: set[str] = field(default_factory=set)
    tree_signature: tuple = ()
    event_outcome: str = ""
    event_agent: str = EVENT_AGENT_SCOPE
    narrowed: bool = False

    def select_tab(self, tab: str) -> None:
        if tab in PRIMARY_TABS:
            self.active_tab = tab
