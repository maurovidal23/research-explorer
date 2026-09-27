"""Derived agent-first navigation data.

The durable timeline is grouped wave-first so replay and reports stay stable.
The dashboard, however, navigates agent-first: each colony agent is a root whose
children are waves, turns, and the papers, discoveries, frontiers, reference
mappings, evaluations, and warnings recorded beneath them. This module derives
that view from the projected
:class:`~research_explorer.events.models.RunViewState` without introducing
durable events or a second source of truth.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from research_explorer.events.models import (
    AGENT_ACTIVE,
    AGENT_EVALUATING,
    NODE_ACTIVE,
    NODE_COMPLETED,
    NODE_FAILED,
    NODE_PENDING,
    NODE_SKIPPED,
    AgentSummary,
    RunViewState,
    TimelineEntry,
)

AGENT_NODE = "agent"
WAVE_NODE = "wave"
TURN_NODE = "turn"
LEAF_NODE = "leaf"

_TERMINAL_NODE_STATUSES = frozenset({NODE_COMPLETED, NODE_FAILED, NODE_SKIPPED})
_LEAF_KINDS = frozenset(
    {"paper", "discovery", "frontier", "reference_mapping", "evaluation", "warning"}
)


class NavNode(BaseModel):
    """One node in the derived agent-first navigation tree."""

    node_id: str
    kind: str
    label: str
    status: str = NODE_PENDING
    agent_id: str = ""
    entry_id: str | None = None
    wave: int = 0
    turn: int = 0
    children: list[NavNode] = Field(default_factory=list)


def build_agent_navigation(state: RunViewState) -> list[NavNode]:
    """Build one root :class:`NavNode` per colony agent, in colony order."""
    return [_build_agent_node(state, summary) for summary in state.ordered_agents()]


def _build_agent_node(state: RunViewState, summary: AgentSummary) -> NavNode:
    agent_id = summary.agent_id
    entries = [e for e in state.timeline if e.agent_id == agent_id]
    root = NavNode(
        node_id=f"agent:{agent_id}",
        kind=AGENT_NODE,
        label=summary.label,
        status=summary.status,
        agent_id=agent_id,
    )
    waves: dict[int, list[TimelineEntry]] = {}
    for entry in entries:
        waves.setdefault(entry.wave, []).append(entry)
    for wave in sorted(waves):
        root.children.append(_build_wave_node(state, agent_id, wave, waves[wave]))
    return root


def _build_wave_node(
    state: RunViewState, agent_id: str, wave: int, entries: list[TimelineEntry]
) -> NavNode:
    label = "Seed" if wave == 0 else f"Wave {wave}"
    turn_entries = sorted(
        (e for e in entries if e.kind == "turn"), key=lambda e: (e.turn, e.seq)
    )
    node = NavNode(
        node_id=f"agent:{agent_id}:wave:{wave}",
        kind=WAVE_NODE,
        label=label,
        status=_wave_status(state, wave, entries),
        agent_id=agent_id,
        wave=wave,
    )
    assigned: set[str] = set()
    for turn_entry in turn_entries:
        assigned.add(turn_entry.entry_id)
        leaves = [
            e
            for e in entries
            if e.parent_id == turn_entry.entry_id and e.kind != "turn"
        ]
        assigned.update(e.entry_id for e in leaves)
        node.children.append(
            _build_turn_node(agent_id, turn_entry, sorted(leaves, key=lambda e: e.seq))
        )
    orphans = [
        e
        for e in entries
        if e.kind != "turn" and e.entry_id not in assigned
    ]
    for orphan in sorted(orphans, key=lambda e: e.seq):
        node.children.append(_leaf_node(orphan))
    return node


def _build_turn_node(
    agent_id: str, entry: TimelineEntry, leaves: list[TimelineEntry]
) -> NavNode:
    node = NavNode(
        node_id=entry.entry_id,
        kind=TURN_NODE,
        label=entry.label,
        status=entry.status,
        agent_id=agent_id,
        entry_id=entry.entry_id,
        wave=entry.wave,
        turn=entry.turn,
    )
    node.children = [_leaf_node(leaf) for leaf in leaves]
    return node


def _leaf_node(entry: TimelineEntry) -> NavNode:
    return NavNode(
        node_id=entry.entry_id,
        kind=LEAF_NODE,
        label=entry.label,
        status=entry.status,
        agent_id=entry.agent_id,
        entry_id=entry.entry_id,
        wave=entry.wave,
        turn=entry.turn,
    )


def _wave_status(state: RunViewState, wave: int, entries: list[TimelineEntry]) -> str:
    global_entry = state.entry_by_id(f"wave:{wave}")
    if global_entry is not None:
        return global_entry.status
    if any(e.status == NODE_ACTIVE for e in entries):
        return NODE_ACTIVE
    statuses = [e.status for e in entries]
    if statuses and all(s in _TERMINAL_NODE_STATUSES for s in statuses):
        return NODE_COMPLETED
    return NODE_PENDING


def flatten_navigation(
    roots: list[NavNode], expanded: set[str]
) -> list[tuple[int, NavNode]]:
    """Flatten the tree to visible rows, honoring the expanded-node set."""
    rows: list[tuple[int, NavNode]] = []

    def walk(nodes: list[NavNode], depth: int) -> None:
        for node in nodes:
            rows.append((depth, node))
            if node.children and node.node_id in expanded:
                walk(node.children, depth + 1)

    walk(roots, 0)
    return rows


def find_node(roots: list[NavNode], node_id: str | None) -> NavNode | None:
    if node_id is None:
        return None
    stack = list(roots)
    while stack:
        node = stack.pop()
        if node.node_id == node_id:
            return node
        stack.extend(node.children)
    return None


def active_agent_id(state: RunViewState) -> str:
    """The agent the auto-follow cursor should track."""
    for status in (AGENT_ACTIVE, AGENT_EVALUATING):
        for summary in state.ordered_agents():
            if summary.status == status:
                return summary.agent_id
    if state.winner_agent:
        return state.winner_agent
    agents = state.ordered_agents()
    if agents:
        return agents[0].agent_id
    return ""


def active_leaf_entry(state: RunViewState, agent_id: str) -> str | None:
    """Newest recorded entry for an agent, used while following live."""
    for entry in reversed(state.timeline):
        if entry.agent_id == agent_id and entry.kind in _LEAF_KINDS:
            return entry.entry_id
    for entry in reversed(state.timeline):
        if entry.agent_id == agent_id:
            return entry.entry_id
    return None


def parent_ids(roots: list[NavNode]) -> dict[str, str]:
    """Map every non-root node id to its parent node id."""
    parents: dict[str, str] = {}
    stack = list(roots)
    while stack:
        node = stack.pop()
        for child in node.children:
            parents[child.node_id] = node.node_id
            stack.append(child)
    return parents


__all__ = [
    "AGENT_NODE",
    "LEAF_NODE",
    "TURN_NODE",
    "WAVE_NODE",
    "NavNode",
    "active_agent_id",
    "active_leaf_entry",
    "build_agent_navigation",
    "find_node",
    "flatten_navigation",
    "parent_ids",
]
