"""Shared deterministic fixtures for the Textual TUI tests."""

from __future__ import annotations

from typing import Any

import pytest

from research_explorer.events.models import RunEvent, RunViewState
from research_explorer.events.projection import RunProjection

ANALYSIS: dict[str, Any] = {
    "summary": "A concise summary",
    "key_concepts": ["alpha", "beta"],
    "methods": "Methods text",
    "findings": "Findings text",
    "relevance": "Relevance text",
    "limitations": "Limitations text",
    "key_references": ["ref-1"],
}


def _evaluation_detail() -> dict[str, Any]:
    return {
        "agent_id": "a0",
        "oleada": 1,
        "turn": 0,
        "q": 0.7,
        "old_quality": 0.5,
        "delta_q": 0.2,
        "weights": {"S": 0.25, "P": 0.25, "J": 0.25, "R": 0.25},
        "self_assessment": {"score": 0.7, "reasoning": "self good"},
        "peers": {
            "votes": [{"voter_id": "agent-1", "score": 0.6, "reasoning": "peer ok"}],
            "aggregated_score": 0.6,
            "num_votes": 1,
        },
        "virgin_judge": {"score": 0.7, "coverage": "covers", "gaps": "gaps here"},
        "structural": {
            "coverage": 0.5,
            "diversity": 0.4,
            "depth": 0.3,
            "coherence": 0.2,
            "r": 0.5,
        },
    }


def build_events() -> list[RunEvent]:
    budget: dict[str, Any] = {
        "used": 3,
        "max_fetches": 100,
        "token_usage": 500,
        "cost": 0.02,
    }
    return [
        RunEvent(
            seq=1,
            type="orchestrator_start",
            payload={
                "run_id": "run-1",
                "seed": "10.1/seed",
                "query": "How do things work?",
                "pipeline": "aco",
                "colony_size": 3,
                "K": 2,
                "k_per_turn": 4,
                "max_fetches": 100,
                "explorer_model": "explorer-x",
                "judge_model": "judge-y",
            },
        ),
        RunEvent(seq=2, type="oleada_start", payload={"oleada": 1, "active": ["a0", "a1", "a2"]}),
        RunEvent(
            seq=3,
            type="agent_turn_start",
            payload={"agent_id": "a0", "caste": "fundaciones", "oleada": 1, "turn": 0},
        ),
        RunEvent(
            seq=4,
            type="paper_fetch_started",
            payload={"agent_id": "a0", "paper_id": "openalex:W1", "mode": "ref", "turn": 0},
        ),
        RunEvent(
            seq=5,
            type="paper_integration_completed",
            payload={
                "agent_id": "a0",
                "paper_id": "openalex:W1",
                "title": "Paper One",
                "mode": "ref",
                "turn": 0,
                "provider": "openalex",
                "analysis": ANALYSIS,
            },
        ),
        RunEvent(
            seq=6,
            type="candidate_score",
            payload={
                "paper_id": "openalex:W1",
                "agent_id": "a0",
                "provider": "openalex",
                "components": {"sim": 0.5, "citations": 0.4},
                "weights": {"w_sim": 0.6},
                "eta": 0.6,
                "source": "10.1/seed",
                "mode": "ref",
            },
        ),
        RunEvent(
            seq=7,
            type="candidate_score",
            payload={
                "paper_id": "openalex:W2",
                "agent_id": "a0",
                "provider": "openalex",
                "components": {"sim": 0.9},
                "weights": {"w_sim": 0.6},
                "eta": 0.9,
                "source": "10.1/seed",
                "mode": "cites",
            },
        ),
        RunEvent(
            seq=8,
            type="candidate_selected",
            payload={
                "agent_id": "a0",
                "paper_id": "openalex:W1",
                "src": "10.1/seed",
                "mode": "ref",
                "caste": "fundaciones",
                "dir_modifier": 0.7,
                "tau": 1.0,
                "alpha": 1.0,
                "beta": 3.0,
                "eta": 0.6,
                "final_weight": 0.216,
                "probability": 0.8,
                "epsilon_branch": False,
                "chosen": True,
            },
        ),
        RunEvent(seq=9, type="evaluation_detail", payload={"detail": _evaluation_detail()}),
        RunEvent(
            seq=10,
            type="agent_turn_complete",
            payload={"agent": "a0", "oleada": 1, "turn": 0, "Q": 0.7, "delta_q": 0.2, "budget": 9},
        ),
        RunEvent(seq=11, type="new_best", payload={"agent": "a0", "Q": 0.7, "oleada": 1}),
        RunEvent(
            seq=12,
            type="artifact_saved",
            payload={
                "artifact_id": "art-1",
                "name": "narrative_a0_t0.md",
                "kind": "narrative",
                "content": "# Narrative\nBody text",
            },
        ),
        RunEvent(seq=13, type="budget_snapshot", payload=budget),
    ]


@pytest.fixture
def run_events() -> list[RunEvent]:
    return build_events()


@pytest.fixture
def view_state(run_events: list[RunEvent]) -> RunViewState:
    return RunProjection.from_events(run_events).state
