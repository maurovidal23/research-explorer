"""Failure, transit, and status edge cases for the projection.

The happy-path semantics live in ``test_projection.py``; this module covers the
degraded branches the PRD calls out: discovery/frontier/paper/evaluation
failures, fatal warnings, transit nodes, the status alias, and the small
projection helpers used by replay.
"""

from __future__ import annotations

from research_explorer.events.models import (
    NODE_COMPLETED,
    NODE_FAILED,
    NODE_SKIPPED,
    RunEvent,
)
from research_explorer.events.projection import RunProjection


def _run_started() -> RunEvent:
    return RunEvent(
        seq=1,
        type="orchestrator_start",
        payload={"run_id": "r1", "colony_size": 2, "K": 1},
    )


def _turn_started(seq: int = 2, agent: str = "a0") -> RunEvent:
    return RunEvent(
        seq=seq,
        type="agent_turn_start",
        payload={"agent_id": agent, "oleada": 1, "turn": 0},
    )


def test_discovery_failed_marks_node_and_records_redacted_failure() -> None:
    state = RunProjection.from_events(
        [
            _run_started(),
            _turn_started(),
            RunEvent(
                seq=3,
                type="neighbor_discovery_failed",
                payload={
                    "agent_id": "a0",
                    "paper_id": "p1",
                    "oleada": 1,
                    "turn": 0,
                    "error": "discovery boom api_key=sk-sentinel-disc",
                },
            ),
        ]
    ).state
    nodes = [e for e in state.timeline if e.kind == "discovery"]
    assert len(nodes) == 1
    assert nodes[0].status == NODE_FAILED
    assert nodes[0].parent_id == "turn:1:a0:0"
    assert any("discovery boom" in failure for failure in state.failures)
    assert "sk-sentinel-disc" not in " ".join(state.failures)


def test_frontier_failed_marks_open_node_and_records_failure() -> None:
    state = RunProjection.from_events(
        [
            _run_started(),
            _turn_started(),
            RunEvent(
                seq=3,
                type="frontier_reference_evaluation_started",
                payload={"agent_id": "a0", "count": 3, "oleada": 1, "turn": 0},
            ),
            RunEvent(
                seq=4,
                type="frontier_reference_evaluation_failed",
                payload={"agent_id": "a0", "oleada": 1, "turn": 0, "error": "timeout"},
            ),
        ]
    ).state
    nodes = [e for e in state.timeline if e.kind == "frontier"]
    assert len(nodes) == 1
    assert nodes[0].status == NODE_FAILED
    assert "candidates" in nodes[0].label
    assert any("timeout" in failure for failure in state.failures)


def test_frontier_failed_without_started_creates_failed_node() -> None:
    state = RunProjection.from_events(
        [
            _run_started(),
            _turn_started(),
            RunEvent(
                seq=3,
                type="frontier_reference_evaluation_failed",
                payload={"agent_id": "a0", "oleada": 1, "turn": 0, "error": "boom"},
            ),
        ]
    ).state
    nodes = [e for e in state.timeline if e.kind == "frontier"]
    assert len(nodes) == 1
    assert nodes[0].status == NODE_FAILED
    assert nodes[0].label == "frontier evaluation failed"


def test_frontier_completed_marks_degraded_label() -> None:
    state = RunProjection.from_events(
        [
            _run_started(),
            _turn_started(),
            RunEvent(
                seq=3,
                type="frontier_reference_evaluation_started",
                payload={"agent_id": "a0", "count": 2, "oleada": 1, "turn": 0},
            ),
            RunEvent(
                seq=4,
                type="frontier_reference_evaluation_completed",
                payload={"agent_id": "a0", "count": 2, "oleada": 1, "turn": 0, "degraded": True},
            ),
        ]
    ).state
    nodes = [e for e in state.timeline if e.kind == "frontier"]
    assert len(nodes) == 1
    assert nodes[0].status == NODE_COMPLETED
    assert "[degraded]" in nodes[0].label


def test_frontier_completed_without_started_still_creates_a_node() -> None:
    state = RunProjection.from_events(
        [
            _run_started(),
            _turn_started(),
            RunEvent(
                seq=3,
                type="frontier_reference_evaluation_completed",
                payload={"agent_id": "a0", "count": 1, "oleada": 1, "turn": 0},
            ),
        ]
    ).state
    nodes = [e for e in state.timeline if e.kind == "frontier"]
    assert len(nodes) == 1
    assert nodes[0].status == NODE_COMPLETED
    assert "1 candidates" in nodes[0].label


def test_paper_fetch_failed_projects_failed_paper_node() -> None:
    state = RunProjection.from_events(
        [
            _run_started(),
            _turn_started(),
            RunEvent(
                seq=3,
                type="paper_fetch_failed",
                payload={
                    "agent_id": "a0",
                    "paper_id": "openalex:W9",
                    "title": "Missing",
                    "oleada": 1,
                    "turn": 0,
                },
            ),
        ]
    ).state
    nodes = [e for e in state.timeline if e.kind == "paper"]
    assert len(nodes) == 1
    assert nodes[0].status == NODE_FAILED
    assert nodes[0].entry_id.startswith("paper:fetch:")


def test_paper_step_completion_updates_existing_node_without_duplication() -> None:
    state = RunProjection.from_events(
        [
            _run_started(),
            _turn_started(),
            RunEvent(
                seq=3,
                type="paper_fetch_started",
                payload={"agent_id": "a0", "paper_id": "p1", "mode": "ref", "oleada": 1, "turn": 0},
            ),
            RunEvent(
                seq=4,
                type="paper_fetch_completed",
                payload={
                    "agent_id": "a0",
                    "paper_id": "p1",
                    "title": "Paper One",
                    "year": 2019,
                    "authors": ["Ada", "Bob"],
                    "src": "10.1/seed",
                    "analysis": {"summary": "done"},
                    "oleada": 1,
                    "turn": 0,
                },
            ),
        ]
    ).state
    nodes = [e for e in state.timeline if e.kind == "paper"]
    assert len(nodes) == 1
    assert nodes[0].status == NODE_COMPLETED
    assert nodes[0].detail["title"] == "Paper One"
    assert nodes[0].detail["year"] == 2019
    assert nodes[0].detail["authors"] == ["Ada", "Bob"]
    assert nodes[0].detail["source"] == "10.1/seed"
    assert nodes[0].detail["analysis"] == {"summary": "done"}
    assert state.paper_analyses["a0"]["p1"] == {"summary": "done"}


def test_evaluation_failed_records_failure_and_marks_agent() -> None:
    state = RunProjection.from_events(
        [
            _run_started(),
            _turn_started(),
            RunEvent(
                seq=3,
                type="quality_evaluation_started",
                payload={"agent_id": "a0", "oleada": 1, "turn": 0},
            ),
            RunEvent(
                seq=4,
                type="quality_evaluation_failed",
                payload={
                    "agent_id": "a0",
                    "oleada": 1,
                    "turn": 0,
                    "error": "judge down token=sk-sentinel-eval",
                },
            ),
        ]
    ).state
    assert any("judge down" in failure for failure in state.failures)
    assert "sk-sentinel-eval" not in " ".join(state.failures)
    assert state.agents["a0"].status == "failed"
    assert state.status == "running"


def test_warning_with_fatal_classification_is_a_failure_node() -> None:
    state = RunProjection.from_events(
        [
            _run_started(),
            _turn_started(),
            RunEvent(
                seq=3,
                type="warning",
                payload={
                    "classification": "fatal",
                    "reason": "seed discovery failed",
                    "agent_id": "a0",
                    "oleada": 1,
                    "turn": 0,
                },
            ),
        ]
    ).state
    assert any("seed discovery failed" in failure for failure in state.failures)
    assert state.warnings == []
    nodes = [e for e in state.timeline if e.kind == "warning"]
    assert nodes and nodes[0].status == NODE_FAILED


def test_metadata_transit_projects_a_skipped_paper_node() -> None:
    state = RunProjection.from_events(
        [
            _run_started(),
            _turn_started(),
            RunEvent(
                seq=3,
                type="metadata_transit",
                payload={"agent_id": "a0", "paper_id": "p9", "oleada": 1, "turn": 0},
            ),
        ]
    ).state
    nodes = [e for e in state.timeline if e.kind == "paper"]
    assert len(nodes) == 1
    assert nodes[0].status == NODE_SKIPPED
    assert nodes[0].label == "transit p9"
    assert nodes[0].parent_id == "turn:1:a0:0"


def test_status_event_updates_run_status() -> None:
    state = RunProjection.from_events(
        [
            _run_started(),
            RunEvent(seq=2, type="status", payload={"status": "exhausted"}),
        ]
    ).state
    assert state.status == "exhausted"


def test_apply_many_and_to_dict_round_trip() -> None:
    projection = RunProjection()
    projection.apply_many(
        [
            _run_started(),
            RunEvent(seq=2, type="status", payload={"status": "exhausted"}),
        ]
    )
    assert projection.to_dict()["run_id"] == "r1"
    assert projection.state.status == "exhausted"


def test_select_entry_and_empty_index_are_safe() -> None:
    projection = RunProjection.from_events([_run_started(), _turn_started()])
    projection.select_entry("turn:1:a0:0")
    assert projection.selected_entry() is not None
    assert projection.selected_entry().entry_id == "turn:1:a0:0"
    RunProjection().select_by_index(0)


def test_hydrate_narrative_sets_agent_and_latest_fallback() -> None:
    projection = RunProjection()
    projection.hydrate_narrative("", "orphan body")
    assert projection.state.narratives["__latest__"] == "orphan body"
    projection.hydrate_narrative("a0", "agent body")
    assert projection.state.narratives["a0"] == "agent body"
    assert projection.state.narratives["__latest__"] == "orphan body"
    projection.hydrate_narrative("a0", "overwritten")
    assert projection.state.narratives["a0"] == "agent body"
