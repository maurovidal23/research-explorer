"""Wave-phase projection, legacy handling, and wave-first navigation tests."""

from __future__ import annotations

from research_explorer.events.models import (
    EVAL_COMPLETE,
    NODE_COMPLETED,
    OUTCOME_DEGRADED,
    PHASE_DECISION,
    PHASE_EVALUATION,
    PHASE_RESEARCH,
    REASON_EMPTY_WINNER_NARRATIVE,
    RunEvent,
)
from research_explorer.events.navigation import (
    FINAL_NODE,
    SETUP_NODE,
    WAVE_NODE,
    active_phase_node_id,
    build_wave_navigation,
    find_node,
)
from research_explorer.events.projection import RunProjection
from research_explorer.tui import text as render


def _detail(q: float, unavailable: dict[str, str]) -> dict:
    return {
        "agent_id": "a0",
        "oleada": 1,
        "turn": 0,
        "q": q,
        "delta_q": q,
        "self_assessment": {"score": 0.0, "reasoning": "empty", "available": False,
                             "unavailable_reason": "empty_narrative"},
        "peers": {"aggregated_score": 0.0, "num_votes": 0, "available": False,
                   "unavailable_reason": "no_peers"},
        "virgin_judge": {"score": 0.0, "coverage": "", "gaps": "empty", "available": False,
                          "unavailable_reason": "empty_narrative"},
        "structural": {"coverage": 0.4, "r": 0.32},
        "unavailable": unavailable,
    }


def _new_trace_events() -> list[RunEvent]:
    return [
        RunEvent(seq=1, type="orchestrator_start", payload={
            "run_id": "1c7cc5e0b6da", "seed": "arxiv:1905.07697",
            "colony_size": 3, "K": 2, "k_per_turn": 2, "max_fetches": 10,
            "pipeline": "aco", "explorer_model": "x", "judge_model": "y",
        }),
        RunEvent(seq=2, type="wave_phase_started", payload={"oleada": 0, "phase": "setup"}),
        RunEvent(seq=3, type="seed_routed", payload={
            "seed": "arxiv:1905.07697", "kind": "arxiv", "provider": "arxiv"}),
        RunEvent(seq=4, type="neighbor_discovery_completed", payload={
            "agent_id": "a0", "paper_id": "arxiv:seed", "oleada": 0, "turn": 0,
            "refs": 3, "cits": 1}),
        RunEvent(seq=5, type="reference_mapping_completed", payload={
            "paper_id": "arxiv:seed", "status": "partial", "observed": 3,
            "mapped": 2, "resolved": 1, "provisional": 1, "failed": 1}),
        RunEvent(seq=6, type="wave_phase_completed", payload={
            "oleada": 0, "phase": "setup", "elapsed": 1.0}),
        RunEvent(seq=7, type="oleada_start", payload={
            "oleada": 1, "active": ["a0"], "phase": "research"}),
        RunEvent(seq=8, type="agent_turn_start", payload={
            "agent_id": "a0", "caste": "mixto", "oleada": 1, "turn": 0}),
        RunEvent(seq=9, type="paper_integration_completed", payload={
            "agent_id": "a0", "paper_id": "arxiv:W1", "title": "Paper One",
            "oleada": 1, "turn": 0}),
        RunEvent(seq=10, type="agent_turn_complete", payload={
            "agent_id": "a0", "agent": "a0", "oleada": 1, "turn": 0,
            "fetches": 1, "budget": 3, "frontier": 1}),
        RunEvent(seq=11, type="wave_phase_completed", payload={
            "oleada": 1, "phase": "research", "selected": ["a0"], "completed": ["a0"],
            "failed": [], "skipped": [], "papers_attempted": 1,
            "papers_integrated": 1, "evidence_added": 1, "elapsed": 0.5}),
        RunEvent(seq=12, type="wave_phase_started", payload={
            "oleada": 1, "phase": "evaluation", "selected": ["a0"]}),
        RunEvent(seq=13, type="quality_evaluation_started", payload={
            "agent_id": "a0", "oleada": 1, "turn": 0}),
        RunEvent(seq=14, type="evaluation_detail", payload={
            "detail": _detail(0.0809, {
                "S": "empty_narrative", "P": "no_peers", "J": "empty_narrative"})}),
        RunEvent(seq=15, type="evaluation_settled", payload={
            "agent_id": "a0", "oleada": 1, "turn": 0, "status": EVAL_COMPLETE,
            "reason": "", "Q": 0.0809, "delta_q": 0.0809,
            "S": 0.0, "P": 0.0, "J": 0.0, "R": 0.32,
            "unavailable": {"S": "empty_narrative", "P": "no_peers",
                             "J": "empty_narrative"},
            "evidence_papers": 1}),
        RunEvent(seq=16, type="wave_phase_completed", payload={
            "oleada": 1, "phase": "evaluation", "selected": ["a0"],
            "completed": ["a0"], "failed": [], "skipped": [],
            "best_Q": 0.0809, "elapsed": 0.2}),
        RunEvent(seq=17, type="wave_phase_started", payload={
            "oleada": 1, "phase": "decision", "selected": ["a0"]}),
        RunEvent(seq=18, type="wave_phase_completed", payload={
            "oleada": 1, "phase": "decision", "selected": ["a0"],
            "leader": "a0", "best_Q": 0.0809, "q_delta": 0.0809,
            "ranking": [["a0", 0.0809]], "budget_used": 1,
            "stop_reason": "budget_exhausted", "elapsed": 0.1}),
        RunEvent(seq=19, type="oleada_complete", payload={
            "oleada": 1, "best_Q": 0.0809, "total_fetches": 1, "max_fetches": 10,
            "leader": "a0", "q_delta": 0.0809, "stop_reason": "budget_exhausted"}),
        RunEvent(seq=20, type="warning", payload={
            "classification": "warning", "outcome": OUTCOME_DEGRADED,
            "reason_code": REASON_EMPTY_WINNER_NARRATIVE,
            "reason": "the winning agent produced no usable narrative"}),
        RunEvent(seq=21, type="orchestrator_complete", payload={
            "run_id": "1c7cc5e0b6da", "status": "completed",
            "outcome": OUTCOME_DEGRADED,
            "reason_code": REASON_EMPTY_WINNER_NARRATIVE,
            "reason": "the winning agent produced no usable narrative",
            "winner": "a0", "peak_Q": 0.0809, "stop_reason": "budget_exhausted",
            "total_fetches": 1, "total_waves": 1, "elapsed": 14.0}),
    ]


def test_projection_reconstructs_setup_and_wave_phases() -> None:
    state = RunProjection.from_events(_new_trace_events()).state

    setup = state.phase(0, "setup")
    assert setup is not None and setup.status == NODE_COMPLETED
    research = state.phase(1, PHASE_RESEARCH)
    evaluation = state.phase(1, PHASE_EVALUATION)
    decision = state.phase(1, PHASE_DECISION)
    assert research is not None and research.completed == ["a0"]
    assert research.papers_integrated == 1 and research.evidence_added == 1
    assert evaluation is not None and evaluation.best_q == 0.0809
    assert decision is not None and decision.leader == "a0"
    assert decision.stop_reason == "budget_exhausted"
    assert state.stop_reason == "budget_exhausted"
    assert state.legacy_projection is False


def test_terminal_evaluation_state_distinguishes_unavailable_from_zero() -> None:
    state = RunProjection.from_events(_new_trace_events()).state
    record = state.evaluation_state(1, "a0")
    assert record is not None and record.status == EVAL_COMPLETE
    assert record.q == 0.0809
    assert record.components["R"] == 0.32
    assert record.components["S"] is None
    assert record.unavailable["S"] == "empty_narrative"
    assert record.unavailable["P"] == "no_peers"


def test_wave_first_navigation_has_no_per_agent_wave_duplication() -> None:
    state = RunProjection.from_events(_new_trace_events()).state
    roots = build_wave_navigation(state)
    kinds = [root.kind for root in roots]
    assert kinds[0] == SETUP_NODE
    assert WAVE_NODE in kinds and kinds[-2] == FINAL_NODE and kinds[-1] == "debug"
    assert "agent" not in kinds
    wave = next(root for root in roots if root.node_id == "wave:1")
    assert [child.node_id for child in wave.children] == [
        "wave:1:research", "wave:1:evaluation", "wave:1:decision"
    ]
    assert find_node(roots, "wave:1:research") is not None


def test_terminal_navigation_selects_final_result() -> None:
    state = RunProjection.from_events(_new_trace_events()).state
    assert active_phase_node_id(state) == "final"
    roots = build_wave_navigation(state)
    final = find_node(roots, "final")
    assert final is not None and final.kind == FINAL_NODE


def test_empty_narrative_regression_is_unmistakable_in_tui() -> None:
    state = RunProjection.from_events(_new_trace_events()).state
    assert state.status == "completed"
    assert state.outcome == OUTCOME_DEGRADED
    assert state.reason_code == REASON_EMPTY_WINNER_NARRATIVE

    mapping = state.entry_by_id("reference:arxiv:seed")
    assert mapping is not None and mapping.kind == "reference_mapping"
    assert "observed=3" in mapping.label

    final_body = render.render_final_result(state)
    assert "No winning narrative" in final_body
    assert "no usable narrative" in final_body
    assert "degraded" in render.status_label(state)
    assert "degraded" in render.render_header(state)
    tree = "\n".join(
        str(render.render_tree_label(state, node, selected=False, expanded=True, depth=0))
        for node in build_wave_navigation(state)
    )
    assert "Final result" in tree


def test_wave_and_evaluation_summaries_survive_event_bursts() -> None:
    projection = RunProjection(event_window=8)
    for event in _new_trace_events():
        projection.apply(event)
    assert projection.state.phase(1, PHASE_RESEARCH) is not None
    assert projection.state.evaluation_state(1, "a0") is not None

    for seq in range(100, 400):
        projection.apply(
            RunEvent(seq=seq, type="candidate_selected",
                     payload={"agent_id": "a0", "paper_id": f"p{seq}"})
        )
    for seq in range(400, 520):
        projection.apply(
            RunEvent(seq=seq, type="paper_fetch_completed",
                     payload={"agent_id": "a0", "paper_id": f"p{seq}"})
        )
    for seq in range(520, 600):
        projection.apply(
            RunEvent(seq=seq, type="reference_mapping_completed",
                     payload={"paper_id": f"p{seq}", "status": "complete",
                              "observed": 1, "mapped": 1, "resolved": 1,
                              "provisional": 0, "failed": 0})
        )

    state = projection.state
    assert state.events_dropped > 0
    assert state.phase(1, PHASE_RESEARCH) is not None
    evaluation = state.phase(1, PHASE_EVALUATION)
    assert evaluation is not None and evaluation.best_q == 0.0809
    record = state.evaluation_state(1, "a0")
    assert record is not None and record.q == 0.0809
    assert record.unavailable["S"] == "empty_narrative"
    roots = build_wave_navigation(state)
    assert find_node(roots, "wave:1:evaluation") is not None


def test_legacy_trace_does_not_invent_evaluation_states() -> None:
    events = [
        RunEvent(seq=1, type="orchestrator_start", payload={
            "run_id": "legacy", "colony_size": 1, "K": 1}),
        RunEvent(seq=2, type="colony_initialized", payload={
            "size": 1, "agents": ["a0"]}),
        RunEvent(seq=3, type="oleada_start", payload={"oleada": 1, "active": ["a0"]}),
        RunEvent(seq=4, type="agent_turn_start", payload={
            "agent_id": "a0", "oleada": 1, "turn": 0}),
        RunEvent(seq=5, type="evaluation_complete", payload={
            "agent_id": "a0", "oleada": 1, "Q": 0.4}),
        RunEvent(seq=6, type="orchestrator_complete", payload={
            "run_id": "legacy", "winner": "a0", "status": "completed",
            "peak_Q": 0.4}),
    ]
    state = RunProjection.from_events(events).state
    assert state.legacy_projection is True
    assert state.evaluation_states == {}
    assert state.phase(1, PHASE_EVALUATION) is None
    roots = build_wave_navigation(state)
    wave = find_node(roots, "wave:1")
    assert wave is not None and wave.label == "Wave 1 (legacy)"
    phase = find_node(roots, "wave:1:evaluation")
    assert phase is not None
