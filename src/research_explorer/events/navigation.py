"""Derived navigation trees over the projected run view state.

The default dashboard hierarchy is wave-first: ``Setup``, one root per wave with
Research/Evaluation/Decision children, ``Final result``, and ``Debug`` (see
:func:`build_wave_navigation`). :func:`build_agent_navigation` remains as the
agent-grouped view of the same timeline for agent-scoped navigation and
regression coverage. Both are pure derivations of the projected
:class:`~research_explorer.events.models.RunViewState`; neither introduces
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
    PHASE_DECISION,
    PHASE_EVALUATION,
    PHASE_RESEARCH,
    TERMINAL_STATUSES,
    WAVE_PHASE_ORDER,
    AgentSummary,
    RunViewState,
    TimelineEntry,
)

AGENT_NODE = "agent"
WAVE_NODE = "wave"
TURN_NODE = "turn"
LEAF_NODE = "leaf"
SETUP_NODE = "setup"
PHASE_NODE = "phase"
FINAL_NODE = "final"
DEBUG_NODE = "debug"

_PHASE_OF_KIND = {
    "turn": PHASE_RESEARCH,
    "paper": PHASE_RESEARCH,
    "discovery": PHASE_RESEARCH,
    "frontier": PHASE_RESEARCH,
    "reference_mapping": PHASE_RESEARCH,
    "evaluation": PHASE_EVALUATION,
    "warning": PHASE_DECISION,
}

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
    detail: dict = Field(default_factory=dict)
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


def build_wave_navigation(state: RunViewState) -> list[NavNode]:
    """Build the default wave-first hierarchy.

    Roots are ``Setup``, one node per wave (with Research/Evaluation/Decision
    children), ``Final result``, and ``Debug``. Waves are never repeated beneath
    individual agents; agent, paper, mapping, and raw-event detail is reachable
    from the selected phase or Debug.
    """
    roots: list[NavNode] = [_build_setup_node(state)]
    for wave in state.ordered_waves():
        roots.append(_build_wave_first_node(state, wave))
    roots.append(_build_final_node(state))
    roots.append(_build_debug_node(state))
    return roots


def _build_setup_node(state: RunViewState) -> NavNode:
    entries = [e for e in state.timeline if e.wave == 0]
    node = NavNode(
        node_id="setup",
        kind=SETUP_NODE,
        label="Setup",
        status=_phase_or_entry_status(state, 0, entries),
    )
    node.children = [_leaf_node(e) for e in sorted(entries, key=lambda e: e.seq)]
    return node


def _build_wave_first_node(state: RunViewState, wave: int) -> NavNode:
    global_entry = state.entry_by_id(f"wave:{wave}")
    label = f"Wave {wave}"
    if state.legacy_projection:
        label += " (legacy)"
    node = NavNode(
        node_id=f"wave:{wave}",
        kind=WAVE_NODE,
        label=label,
        status=global_entry.status if global_entry is not None else _entries_status(
            [e for e in state.timeline if e.wave == wave]
        ),
        wave=wave,
    )
    for phase in WAVE_PHASE_ORDER:
        node.children.append(_build_phase_node(state, wave, phase))
    decisions = state.phase(wave, PHASE_DECISION)
    if decisions is not None:
        node.detail = {
            "leader": decisions.leader,
            "q_delta": decisions.q_delta,
            "stop_reason": decisions.stop_reason,
            "continue_reason": decisions.continue_reason,
        }
    return node


def _build_phase_node(state: RunViewState, wave: int, phase: str) -> NavNode:
    record = state.phase(wave, phase)
    entries = [
        e
        for e in state.timeline
        if e.wave == wave and _PHASE_OF_KIND.get(e.kind) == phase
    ]
    status = record.status if record is not None else _entries_status(entries)
    node = NavNode(
        node_id=f"wave:{wave}:{phase}",
        kind=PHASE_NODE,
        label=phase.title(),
        status=status,
        wave=wave,
    )
    if record is not None:
        node.detail = {
            "selected": record.selected,
            "completed": record.completed,
            "failed": record.failed,
            "skipped": record.skipped,
            "papers_attempted": record.papers_attempted,
            "papers_integrated": record.papers_integrated,
            "evidence_added": record.evidence_added,
            "best_q": record.best_q,
            "leader": record.leader,
            "q_delta": record.q_delta,
            "budget_used": record.budget_used,
            "continue_reason": record.continue_reason,
            "stop_reason": record.stop_reason,
        }
    node.children = [_leaf_node(e) for e in sorted(entries, key=lambda e: e.seq)]
    return node


def _build_final_node(state: RunViewState) -> NavNode:
    status = NODE_COMPLETED if state.status in TERMINAL_STATUSES else NODE_PENDING
    node = NavNode(node_id="final", kind=FINAL_NODE, label="Final result", status=status)
    node.detail = {
        "outcome": state.outcome,
        "stop_reason": state.stop_reason or state.terminal_reason,
        "winner": state.winner_agent,
        "best_quality": state.best_quality,
    }
    return node


def _build_debug_node(state: RunViewState) -> NavNode:
    node = NavNode(
        node_id="debug",
        kind=DEBUG_NODE,
        label="Debug",
        status=NODE_COMPLETED if state.events else NODE_PENDING,
    )
    warnings = [e for e in state.timeline if e.kind == "warning"]
    node.children = [_leaf_node(e) for e in sorted(warnings, key=lambda e: e.seq)]
    node.detail = {
        "events_seen": state.events_seen_total,
        "events_dropped": state.events_dropped,
        "warnings": len(state.warnings),
        "failures": len(state.failures),
    }
    return node


def _entries_status(entries: list[TimelineEntry]) -> str:
    if any(e.status == NODE_ACTIVE for e in entries):
        return NODE_ACTIVE
    statuses = [e.status for e in entries]
    if statuses and all(s in _TERMINAL_NODE_STATUSES for s in statuses):
        return NODE_COMPLETED
    if any(e.status == NODE_FAILED for e in entries):
        return NODE_FAILED
    return NODE_PENDING


def _phase_or_entry_status(
    state: RunViewState, wave: int, entries: list[TimelineEntry]
) -> str:
    if wave == 0:
        if entries and all(e.status in _TERMINAL_NODE_STATUSES for e in entries):
            return NODE_COMPLETED
        return NODE_ACTIVE if state.status not in TERMINAL_STATUSES else NODE_COMPLETED
    return _entries_status(entries)


def active_wave_phase(state: RunViewState) -> tuple[int, str]:
    """The wave/phase the follow cursor should track while a run is live."""
    wave = state.current_wave
    phase = state.current_phase
    if not phase:
        phase = PHASE_RESEARCH
    return wave, phase


def active_phase_node_id(state: RunViewState) -> str:
    wave, phase = active_wave_phase(state)
    if state.status in TERMINAL_STATUSES:
        return "final"
    if not wave:
        return "setup"
    if state.phase(wave, phase) is None and not any(
        e.wave == wave and _PHASE_OF_KIND.get(e.kind) == phase for e in state.timeline
    ):
        return f"wave:{wave}"
    return f"wave:{wave}:{phase}"


__all__ = [
    "AGENT_NODE",
    "DEBUG_NODE",
    "FINAL_NODE",
    "LEAF_NODE",
    "PHASE_NODE",
    "SETUP_NODE",
    "TURN_NODE",
    "WAVE_NODE",
    "NavNode",
    "active_agent_id",
    "active_leaf_entry",
    "active_phase_node_id",
    "active_wave_phase",
    "build_agent_navigation",
    "build_wave_navigation",
    "find_node",
    "flatten_navigation",
    "parent_ids",
]
