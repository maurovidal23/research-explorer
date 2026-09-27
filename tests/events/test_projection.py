"""Deterministic projection tests for the run view state."""

from __future__ import annotations

from research_explorer.events.models import (
    AGENT_ACTIVE,
    AGENT_COMPLETED,
    AGENT_EVALUATING,
    AGENT_EXHAUSTED,
    AGENT_FAILED,
    AGENT_WAITING,
    NODE_SKIPPED,
    STATUS_COMPLETED,
    STATUS_EVALUATING,
    EventType,
    RunEvent,
)
from research_explorer.events.projection import RunProjection
from research_explorer.tui import text as render


def _run_started() -> RunEvent:
    return RunEvent(
        seq=1,
        type="orchestrator_start",
        payload={
            "run_id": "r1",
            "seed": "seed-1",
            "query": "q",
            "colony_size": 3,
            "K": 2,
            "k_per_turn": 4,
            "max_fetches": 100,
            "explorer_model": "explorer",
            "judge_model": "judge",
        },
    )


def test_run_and_budget_progression() -> None:
    projection = RunProjection()
    projection.apply(_run_started())
    assert projection.state.status == "running"
    assert projection.state.max_fetches == 100
    assert projection.state.k_per_turn == 4
    assert projection.state.colony_size == 3

    projection.apply(
        RunEvent(
            seq=2,
            type="budget_snapshot",
            payload={"used": 12, "max_fetches": 100, "token_usage": 500, "cost": 0.02},
        )
    )
    assert projection.state.fetches_used == 12
    assert projection.state.token_usage == 500
    assert projection.state.cost == 0.02


def test_token_usage_unavailable_not_zero() -> None:
    projection = RunProjection.from_events([_run_started()])
    assert projection.state.token_usage is None
    assert projection.state.cost is None
    header = render.render_header(projection.state)
    assert "tokens unavailable" in header


def test_agent_ordering_and_status_transitions() -> None:
    projection = RunProjection()
    projection.apply(_run_started())
    projection.apply(
        RunEvent(seq=2, type="agent_turn_queued", payload={"agent_id": "agent-1", "caste": "impacto"})
    )
    projection.apply(
        RunEvent(seq=3, type="agent_turn_start", payload={"agent_id": "agent-0", "caste": "fundaciones"})
    )
    assert projection.state.agent_order == ["agent-1", "agent-0"]
    assert projection.state.agents["agent-1"].caste == "impacto"

    projection.apply(
        RunEvent(
            seq=4,
            type="agent_turn_complete",
            payload={"agent": "agent-0", "Q": 0.6, "delta_q": 0.6, "budget": 5},
        )
    )
    assert projection.state.agents["agent-0"].quality == 0.6
    assert projection.state.agents["agent-1"].status == AGENT_WAITING
    assert projection.state.agents["agent-0"].status == AGENT_WAITING


def test_sequential_execution_representation() -> None:
    projection = RunProjection()
    projection.apply(_run_started())
    projection.apply(RunEvent(seq=2, type="oleada_start", payload={"oleada": 1, "active": ["a0", "a1"]}))
    projection.apply(RunEvent(seq=3, type="agent_turn_start", payload={"agent_id": "a0", "oleada": 1, "turn": 0}))
    projection.apply(RunEvent(seq=4, type="agent_turn_start", payload={"agent_id": "a1", "oleada": 1, "turn": 0}))
    assert projection.state.agents["a0"].status == AGENT_ACTIVE
    assert projection.state.agents["a1"].status == AGENT_ACTIVE
    projection.apply(RunEvent(seq=5, type="agent_turn_complete", payload={"agent": "a0", "oleada": 1, "turn": 0, "budget": 3}))
    assert projection.state.agents["a0"].status == AGENT_WAITING


def test_grouping_by_wave_and_turn() -> None:
    projection = RunProjection.from_events(
        [
            _run_started(),
            RunEvent(seq=2, type="oleada_start", payload={"oleada": 1, "active": ["a0"]}),
            RunEvent(seq=3, type="agent_turn_start", payload={"agent_id": "a0", "oleada": 1, "turn": 0}),
            RunEvent(
                seq=4,
                type="paper_integration_completed",
                payload={"agent_id": "a0", "paper_id": "p1", "title": "P1", "turn": 0, "mode": "ref"},
            ),
            RunEvent(
                seq=5,
                type="evaluation_complete",
                payload={"agent_id": "a0", "oleada": 1, "turn": 0, "Q": 0.5},
            ),
            RunEvent(seq=6, type="oleada_complete", payload={"oleada": 1, "best_Q": 0.5, "total_fetches": 4, "max_fetches": 100}),
        ]
    )
    waves = [e for e in projection.state.timeline if e.kind == "wave"]
    turns = [e for e in projection.state.timeline if e.kind == "turn"]
    papers = [e for e in projection.state.timeline if e.kind == "paper"]
    evals = [e for e in projection.state.timeline if e.kind == "evaluation"]
    assert len(waves) == 1 and waves[0].status == "completed"
    assert len(turns) == 1 and turns[0].parent_id == "wave:1"
    assert papers and papers[0].parent_id == turns[0].entry_id
    assert evals and evals[0].parent_id == turns[0].entry_id
    assert projection.state.fetches_used == 4


def test_best_agent_changes() -> None:
    projection = RunProjection.from_events(
        [
            _run_started(),
            RunEvent(seq=2, type="new_best", payload={"agent": "a0", "Q": 0.4}),
            RunEvent(seq=3, type="new_best", payload={"agent": "a1", "Q": 0.8}),
        ]
    )
    assert projection.state.winner_agent == "a1"
    assert projection.state.best_quality == 0.8
    assert projection.state.agents["a1"].is_winner
    assert not projection.state.agents["a0"].is_winner


def test_skipped_evaluation_semantics() -> None:
    projection = RunProjection.from_events(
        [
            _run_started(),
            RunEvent(seq=2, type="oleada_start", payload={"oleada": 1}),
            RunEvent(seq=3, type="agent_turn_start", payload={"agent_id": "a0", "oleada": 1, "turn": 0}),
            RunEvent(
                seq=4,
                type="evaluation_skipped",
                payload={"agent_id": "a0", "oleada": 1, "turn": 0, "reason": "no_new_evidence"},
            ),
        ]
    )
    entries = [e for e in projection.state.timeline if e.kind == "evaluation"]
    assert entries and entries[0].status == NODE_SKIPPED
    assert projection.state.evaluations.get("a0") in (None, [])
    assert not projection.state.failures


def test_transient_and_fatal_failures() -> None:
    projection = RunProjection.from_events(
        [
            _run_started(),
            RunEvent(
                seq=2,
                type="provider_failure",
                payload={"paper_id": "p1", "reason": "timeout", "classification": "transient"},
            ),
            RunEvent(seq=3, type="agent_turn_failed", payload={"agent_id": "a0", "error": "boom"}),
            RunEvent(seq=4, type="run_failed", payload={"error": "fatal"}),
        ]
    )
    assert projection.state.status == "failed"
    assert any("boom" in f for f in projection.state.failures)
    assert any("fatal" in f for f in projection.state.failures)
    assert any("timeout" in w for w in projection.state.warnings)
    assert projection.state.agents["a0"].status == AGENT_FAILED


def test_historical_selection_versus_follow_live() -> None:
    projection = RunProjection.from_events(
        [
            _run_started(),
            RunEvent(seq=2, type="oleada_start", payload={"oleada": 1}),
            RunEvent(seq=3, type="agent_turn_start", payload={"agent_id": "a0", "oleada": 1, "turn": 0}),
        ]
    )
    assert projection.state.follow_live
    newest = projection.state.timeline[-1].entry_id
    assert projection.state.selected_entry_id == newest

    projection.select_by_index(0)
    assert not projection.state.follow_live
    assert projection.state.selected_entry_id == projection.state.timeline[0].entry_id

    projection.apply(RunEvent(seq=4, type="new_best", payload={"agent": "a0", "Q": 0.5}))
    assert not projection.state.follow_live
    assert projection.state.selected_entry_id == projection.state.timeline[0].entry_id

    projection.restore_follow()
    assert projection.state.follow_live
    assert projection.state.selected_entry_id == projection.state.timeline[-1].entry_id


def test_duplicate_delivery_idempotence() -> None:
    events = [
        _run_started(),
        RunEvent(seq=2, type="oleada_start", payload={"oleada": 1}),
        RunEvent(seq=3, type="agent_turn_start", payload={"agent_id": "a0", "oleada": 1, "turn": 0}),
    ]
    projection = RunProjection.from_events(events)
    timeline_before = len(projection.state.timeline)
    for event in events:
        projection.apply(event)
    assert len(projection.state.timeline) == timeline_before
    assert len(projection.state.events) == len(events)


def test_unknown_event_compatibility() -> None:
    projection = RunProjection.from_events(
        [
            _run_started(),
            RunEvent(seq=2, type="future_event_type", payload={"payload": {"a": 1}}),
        ]
    )
    assert projection.state.status == "running"
    assert any(e.type == "future_event_type" for e in projection.state.events)
    assert len(projection.state.timeline) == 0


def test_partial_paper_analysis_degrades_field_by_field() -> None:
    projection = RunProjection()
    projection.apply(_run_started())
    projection.apply(RunEvent(seq=2, type="oleada_start", payload={"oleada": 1}))
    projection.apply(RunEvent(seq=3, type="agent_turn_start", payload={"agent_id": "a0", "oleada": 1, "turn": 0}))
    projection.apply(
        RunEvent(
            seq=4,
            type="paper_integration_completed",
            payload={
                "agent_id": "a0",
                "paper_id": "p1",
                "title": "P1",
                "turn": 0,
                "analysis": {"summary": "only summary", "methods": 123, "key_concepts": None},
            },
        )
    )
    state = projection.state
    assert state.paper_analyses["a0"]["p1"]["summary"] == "only summary"
    body = render.render_paper_full(state, "a0", "p1")
    assert "only summary" in body
    assert "key_concepts: unavailable" in body
    assert "findings: unavailable" in body
    detail = render.render_detail(state, state.timeline[-1])
    assert "only summary" in detail or "no structured analysis" not in detail


def test_evaluation_detail_and_compact_dedupe() -> None:
    compact = RunEvent(
        seq=2,
        type="evaluation_complete",
        payload={"agent_id": "a0", "oleada": 1, "turn": 0, "Q": 0.7, "delta_q": 0.7},
    )
    detail = RunEvent(
        seq=3,
        type="evaluation_detail",
        payload={
            "detail": {
                "agent_id": "a0",
                "oleada": 1,
                "turn": 0,
                "q": 0.7,
                "self_assessment": {"score": 0.7, "reasoning": "r"},
            }
        },
    )
    projection = RunProjection.from_events([_run_started(), compact, detail])
    records = projection.state.evaluations["a0"]
    assert len(records) == 1
    assert records[0].self_assessment.reasoning == "r"


def test_secrets_redacted_in_failures() -> None:
    projection = RunProjection.from_events(
        [
            _run_started(),
            RunEvent(
                seq=2,
                type="run_failed",
                payload={"error": "request failed api_key=sk-sentinel-12345"},
            ),
        ]
    )
    assert "sk-sentinel-12345" not in " ".join(projection.state.failures)
    assert "***" in " ".join(projection.state.failures)


def test_secrets_redacted_in_event_view() -> None:
    sentinel = "sk-sentinel-99999"
    projection = RunProjection.from_events(
        [
            _run_started(),
            RunEvent(
                seq=2,
                type="provider_failure",
                payload={"paper_id": "p1", "reason": f"429 https://api.x.test?api_key={sentinel}"},
            ),
            RunEvent(seq=3, type="future_event", payload={"note": f"token={sentinel}"}),
        ]
    )
    state = projection.state
    body = render.render_events(state)
    assert sentinel not in body
    assert sentinel not in str([e.payload for e in state.events])


def test_cancelled_status_distinct() -> None:
    projection = RunProjection.from_events(
        [
            _run_started(),
            RunEvent(seq=2, type=EventType.RUN_CANCELLED, payload={"run_id": "r1"}),
        ]
    )
    assert projection.state.status == "cancelled"
    assert projection.state.status != STATUS_COMPLETED


def test_finalize_marks_agents_completed() -> None:
    projection = RunProjection.from_events(
        [
            _run_started(),
            RunEvent(seq=2, type="agent_turn_start", payload={"agent_id": "a0", "oleada": 1, "turn": 0}),
            RunEvent(seq=3, type="orchestrator_complete", payload={"winner": "a0", "status": "completed"}),
        ]
    )
    assert projection.state.agents["a0"].status == AGENT_COMPLETED


def test_seed_routing_and_colony_init_populate_agents() -> None:
    projection = RunProjection.from_events(
        [
            _run_started(),
            RunEvent(seq=2, type="seed_routing_started", payload={"seed": "s"}),
            RunEvent(seq=3, type="colony_initialized", payload={"size": 2, "agents": ["a0", "a1"]}),
        ]
    )
    assert projection.state.colony_size == 2
    assert projection.state.agent_order == ["a0", "a1"]
    assert all(agent.status == AGENT_WAITING for agent in projection.state.agents.values())


def test_historical_waves_remain_available() -> None:
    projection = RunProjection.from_events(
        [
            _run_started(),
            RunEvent(seq=2, type="oleada_start", payload={"oleada": 1, "active": ["a0"]}),
            RunEvent(seq=3, type="agent_turn_start", payload={"agent_id": "a0", "oleada": 1, "turn": 0}),
            RunEvent(
                seq=4,
                type="oleada_complete",
                payload={"oleada": 1, "best_Q": 0.3, "total_fetches": 2, "max_fetches": 100},
            ),
            RunEvent(seq=5, type="oleada_start", payload={"oleada": 2, "active": ["a0"]}),
        ]
    )
    waves = [e for e in projection.state.timeline if e.kind == "wave"]
    assert [w.wave for w in waves] == [1, 2]
    assert waves[0].status == "completed"
    assert waves[1].status == "active"
    assert projection.state.current_wave == 2


def test_turn_budget_reaches_exhaustion() -> None:
    projection = RunProjection.from_events(
        [
            _run_started(),
            RunEvent(seq=2, type="agent_turn_start", payload={"agent_id": "a0", "oleada": 1, "turn": 0}),
            RunEvent(
                seq=3,
                type="agent_turn_complete",
                payload={"agent": "a0", "oleada": 1, "turn": 0, "budget": 0},
            ),
        ]
    )
    assert projection.state.agents["a0"].status == AGENT_EXHAUSTED


def test_evaluation_started_sets_evaluating_state() -> None:
    projection = RunProjection.from_events(
        [
            _run_started(),
            RunEvent(seq=2, type="agent_turn_start", payload={"agent_id": "a0", "oleada": 1, "turn": 0}),
            RunEvent(
                seq=3,
                type="quality_evaluation_started",
                payload={"agent_id": "a0", "oleada": 1, "turn": 0},
            ),
        ]
    )
    assert projection.state.status == STATUS_EVALUATING
    assert projection.state.agents["a0"].status == AGENT_EVALUATING


def test_malformed_evaluation_detail_does_not_crash() -> None:
    projection = RunProjection.from_events(
        [
            _run_started(),
            RunEvent(
                seq=2,
                type="evaluation_detail",
                payload={"detail": {"agent_id": "a0", "q": "not-a-number"}},
            ),
            RunEvent(seq=3, type="future_event", payload={"x": 1}),
        ]
    )
    assert projection.state.status == "running"
    assert not projection.state.evaluations.get("a0")
    assert any(e.type == "future_event" for e in projection.state.events)


def test_cancelled_run_finalizes_agents() -> None:
    projection = RunProjection.from_events(
        [
            _run_started(),
            RunEvent(seq=2, type="agent_turn_start", payload={"agent_id": "a0", "oleada": 1, "turn": 0}),
            RunEvent(seq=3, type="run_cancelled", payload={"run_id": "r1"}),
        ]
    )
    assert projection.state.status == "cancelled"
    assert projection.state.agents["a0"].status == AGENT_COMPLETED


def test_llm_operation_telemetry_projects_model_elapsed_and_tokens() -> None:
    projection = RunProjection.from_events(
        [
            _run_started(),
            RunEvent(
                seq=2,
                type=EventType.LLM_STARTED,
                payload={"purpose": "paper_integration", "model": "explorer-x"},
            ),
            RunEvent(
                seq=3,
                type=EventType.LLM_COMPLETED,
                payload={
                    "purpose": "paper_integration",
                    "model": "explorer-x",
                    "elapsed": 1.5,
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            ),
            RunEvent(
                seq=4,
                type=EventType.LLM_COMPLETED,
                payload={
                    "purpose": "virgin_judge",
                    "model": "judge-y",
                    "elapsed": 0.5,
                    "total_tokens": 30,
                },
            ),
        ]
    )
    state = projection.state
    assert state.current_operation == "virgin_judge"
    assert state.current_operation_model == "judge-y"
    assert state.operation_elapsed_seconds == 0.5
    assert state.token_usage == 150


def test_llm_failure_is_recorded_and_redacted() -> None:
    projection = RunProjection.from_events(
        [
            _run_started(),
            RunEvent(
                seq=2,
                type=EventType.LLM_FAILED,
                payload={"purpose": "paper_integration", "error": "401 api_key=sk-sentinel-77"},
            ),
        ]
    )
    assert any("paper_integration" in f for f in projection.state.failures)
    assert "sk-sentinel-77" not in " ".join(projection.state.failures)


def test_discovery_and_frontier_events_become_timeline_nodes() -> None:
    projection = RunProjection.from_events(
        [
            _run_started(),
            RunEvent(seq=2, type="oleada_start", payload={"oleada": 1, "active": ["a0"]}),
            RunEvent(seq=3, type="agent_turn_start", payload={"agent_id": "a0", "oleada": 1, "turn": 0}),
            RunEvent(
                seq=4,
                type="neighbor_discovery_started",
                payload={"agent_id": "a0", "paper_id": "p1", "turn": 0},
            ),
            RunEvent(
                seq=5,
                type="neighbor_discovery_completed",
                payload={"agent_id": "a0", "paper_id": "p1", "turn": 0, "refs": 4, "cits": 2},
            ),
            RunEvent(
                seq=6,
                type="frontier_reference_evaluation_started",
                payload={"agent_id": "a0", "count": 4, "turn": 0},
            ),
            RunEvent(
                seq=7,
                type="frontier_reference_evaluation_completed",
                payload={"agent_id": "a0", "count": 4, "turn": 0},
            ),
        ]
    )
    timeline = projection.state.timeline
    discovery = [e for e in timeline if e.kind == "discovery"]
    frontier = [e for e in timeline if e.kind == "frontier"]
    assert len(discovery) == 1
    assert len(frontier) == 1
    assert discovery[0].status == "completed"
    assert "refs=4" in discovery[0].label
    assert frontier[0].status == "completed"
    parent = "turn:1:a0:0"
    assert discovery[0].parent_id == parent
    assert frontier[0].parent_id == parent


def test_turn_completed_projects_frontier_state() -> None:
    projection = RunProjection.from_events(
        [
            _run_started(),
            RunEvent(seq=2, type="agent_turn_start", payload={"agent_id": "a0", "oleada": 1, "turn": 0}),
            RunEvent(
                seq=3,
                type="agent_turn_complete",
                payload={"agent": "a0", "oleada": 1, "turn": 0, "budget": 8, "frontier": 12},
            ),
        ]
    )
    assert projection.state.agents["a0"].frontier == 12
    assert projection.state.frontier_size == 12


def test_paper_step_projects_year_authors_and_source() -> None:
    projection = RunProjection.from_events(
        [
            _run_started(),
            RunEvent(seq=2, type="agent_turn_start", payload={"agent_id": "a0", "oleada": 1, "turn": 0}),
            RunEvent(
                seq=3,
                type="paper_fetch_completed",
                payload={
                    "agent_id": "a0",
                    "paper_id": "openalex:W1",
                    "title": "Paper One",
                    "year": 2019,
                    "authors": ["Ada", "Bob"],
                    "src": "10.1/seed",
                    "provider": "openalex",
                    "mode": "ref",
                    "turn": 0,
                },
            ),
        ]
    )
    summary = projection.state.agents["a0"]
    assert summary.current_paper_year == 2019
    assert summary.current_paper_authors == ["Ada", "Bob"]
    assert summary.current_paper_source == "10.1/seed"
