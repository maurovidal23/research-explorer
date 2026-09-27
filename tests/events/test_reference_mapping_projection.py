"""Projection and navigation coverage for paper-level reference-mapping events.

The design pass nested the mapping step under its agent/turn, made partial and
incomplete mappings visible as degraded completions, and marked reused mappings.
These tests pin that contract (PRD §8) so a regression cannot silently render a
partial mapping as a clean, empty result.
"""

from __future__ import annotations

from research_explorer.events.models import (
    NODE_ACTIVE,
    NODE_COMPLETED,
    NODE_FAILED,
    EventType,
    RunEvent,
)
from research_explorer.events.navigation import (
    active_leaf_entry,
    build_agent_navigation,
    find_node,
)
from research_explorer.events.projection import RunProjection

PAPER = "arxiv:2106.09685"
JOB = "map-abc123"


def _mapping_state(event_type: str = EventType.REFERENCE_MAPPING_COMPLETED, **payload):
    events = [
        RunEvent(
            seq=1,
            type="orchestrator_start",
            payload={"run_id": "r", "colony_size": 2, "K": 1},
        ),
        RunEvent(
            seq=2,
            type="colony_initialized",
            payload={"size": 2, "agents": ["a0", "a1"]},
        ),
        RunEvent(seq=3, type="oleada_start", payload={"oleada": 1, "active": ["a0"]}),
        RunEvent(
            seq=4,
            type="agent_turn_start",
            payload={"agent_id": "a0", "oleada": 1, "turn": 0},
        ),
        RunEvent(seq=5, type=event_type, payload={"agent_id": "a0", **payload}),
    ]
    return RunProjection.from_events(events).state


def test_completed_mapping_nests_under_turn_with_counts() -> None:
    state = _mapping_state(
        paper_id=PAPER,
        job_id=JOB,
        status="completed",
        observed=62,
        mapped=62,
        resolved=12,
        provisional=50,
        failed=0,
    )
    entry = state.entry_by_id(f"reference:{JOB}")
    assert entry is not None
    assert entry.kind == "reference_mapping"
    assert entry.agent_id == "a0"
    assert entry.parent_id == "turn:1:a0:0"
    assert entry.status == NODE_COMPLETED
    assert "observed=62" in entry.label
    assert "mapped=62" in entry.label
    assert "resolved=12" in entry.label
    assert "[degraded]" not in entry.label
    assert "[reused]" not in entry.label
    assert state.warnings == []

    turn = find_node(build_agent_navigation(state), "turn:1:a0:0")
    assert turn is not None
    assert any("map references" in child.label for child in turn.children)
    assert active_leaf_entry(state, "a0") == f"reference:{JOB}"


def test_partial_mapping_is_degraded_and_visible() -> None:
    state = _mapping_state(
        paper_id=PAPER,
        job_id=JOB,
        status="partial",
        observed=62,
        mapped=62,
        resolved=12,
        provisional=50,
        failed=0,
    )
    entry = state.entry_by_id(f"reference:{JOB}")
    assert entry is not None
    assert entry.status == NODE_COMPLETED
    assert "[degraded]" in entry.label
    assert any("reference mapping partial" in w for w in state.warnings)
    assert state.failures == []


def test_incomplete_mapping_is_degraded_not_clean_success() -> None:
    state = _mapping_state(
        paper_id=PAPER,
        job_id=JOB,
        status="incomplete",
        observed=62,
        mapped=10,
        resolved=0,
        provisional=10,
        failed=0,
    )
    entry = state.entry_by_id(f"reference:{JOB}")
    assert entry is not None
    assert entry.status == NODE_COMPLETED
    assert "[degraded]" in entry.label
    assert any("reference mapping incomplete" in w for w in state.warnings)


def test_failed_mapping_is_a_failure_not_a_warning() -> None:
    state = _mapping_state(
        event_type=EventType.REFERENCE_MAPPING_FAILED,
        paper_id=PAPER,
        status="failed",
        observed=62,
        mapped=0,
        resolved=0,
        provisional=0,
        failed=62,
        error_code="builder_error",
    )
    # No job id was supplied, so the entry falls back to the paper id.
    entry = state.entry_by_id(f"reference:{PAPER}")
    assert entry is not None
    assert entry.status == NODE_FAILED
    assert any(
        "reference mapping failed" in f and "builder_error" in f for f in state.failures
    )
    assert state.warnings == []


def test_reused_mapping_is_labeled_distinctly() -> None:
    state = _mapping_state(
        event_type=EventType.REFERENCE_MAPPING_REUSED,
        paper_id=PAPER,
        job_id=JOB,
        status="completed",
        observed=62,
        mapped=62,
        resolved=12,
        provisional=50,
        failed=0,
        reused=True,
    )
    entry = state.entry_by_id(f"reference:{JOB}")
    assert entry is not None
    assert entry.status == NODE_COMPLETED
    assert "[reused]" in entry.label
    assert "[degraded]" not in entry.label


def test_started_mapping_is_active_before_completion() -> None:
    state = _mapping_state(
        event_type=EventType.REFERENCE_MAPPING_STARTED,
        paper_id=PAPER,
        job_id=JOB,
        observed=62,
    )
    entry = state.entry_by_id(f"reference:{JOB}")
    assert entry is not None
    assert entry.status == NODE_ACTIVE


def test_completed_mapping_with_zero_observed_does_not_warn() -> None:
    state = _mapping_state(
        paper_id="arxiv:empty",
        job_id="map-empty",
        status="completed",
        observed=0,
        mapped=0,
        resolved=0,
        provisional=0,
        failed=0,
    )
    assert state.warnings == []
    assert state.failures == []
