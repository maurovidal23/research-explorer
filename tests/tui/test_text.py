"""Unit tests for the Textual-free renderers in ``research_explorer.tui.text``.

These lock down what each pane promises so projection or renderer regressions
surface without needing a terminal.
"""

from __future__ import annotations

from research_explorer.events.models import (
    AGENT_ACTIVE,
    AGENT_COMPLETED,
    AGENT_EVALUATING,
    AGENT_EXHAUSTED,
    AGENT_FAILED,
    AGENT_WAITING,
    RunEvent,
    RunViewState,
)
from research_explorer.events.projection import RunProjection
from research_explorer.tui import text as render


def test_ellipsize_boundaries() -> None:
    assert render.ellipsize("abc", 5) == "abc"
    assert render.ellipsize("abcdef", 4) == "abc\u2026"
    assert render.ellipsize("abcdef", 1) == "a"
    assert render.ellipsize("abcdef", 0) == ""


def test_format_duration_units() -> None:
    assert render.format_duration(5) == "5s"
    assert render.format_duration(65) == "1m05s"
    assert render.format_duration(3661) == "1h01m01s"
    assert render.format_duration(-4) == "0s"


def test_caste_label_normalizes_stored_spanish_values() -> None:
    assert render.caste_label("fundaciones") == "foundations"
    assert render.caste_label("fundamentos") == "foundations"
    assert render.caste_label("impacto") == "impact"
    assert render.caste_label("mixto") == "mixed"
    assert render.caste_label("otro") == "otro"
    assert render.caste_label("") == "mixed"


def test_lifecycle_markers_are_distinct_text_not_color() -> None:
    marks = {
        s: render.status_mark(s)
        for s in (
            AGENT_ACTIVE,
            AGENT_EVALUATING,
            AGENT_WAITING,
            AGENT_EXHAUSTED,
            AGENT_COMPLETED,
            AGENT_FAILED,
        )
    }
    assert len(set(marks.values())) == 6
    assert all(m and m != "unknown" for m in marks.values())
    assert render.status_mark("") == "unknown"
    assert render.status_mark("bespoke") == "bespoke"


def test_header_shows_authoritative_identity_budget_and_models(view_state: RunViewState) -> None:
    header = render.render_header(view_state)
    assert "run-1" in header
    assert "elapsed 0s" in header
    assert "fetch 3/100" in header
    assert "wave 1" in header
    assert "best 0.700" in header
    assert "A01" in header
    assert "explorer-x" in header
    assert "judge-y" in header
    assert "10.1/seed" in header
    assert "colony 3" in header
    assert "slots 2" in header
    assert "papers/eval 4" in header
    assert "tokens 500" in header
    assert "How do things work?" in header


def test_header_marks_unknown_tokens_unavailable_not_zero() -> None:
    events = [
        RunEvent(
            seq=1,
            type="orchestrator_start",
            payload={"run_id": "r", "colony_size": 1, "K": 1},
        )
    ]
    header = render.render_header(RunProjection.from_events(events).state)
    assert "tokens unavailable" in header
    assert "tokens 0" not in header


def test_budget_bar_reports_exact_fraction(view_state: RunViewState) -> None:
    budget = render.render_budget(view_state)
    assert "3/100 fetches" in budget
    assert "=" in budget or "." in budget


def test_budget_time_and_convergence_stopping_rules() -> None:
    time_state = RunViewState(
        budget_type="time",
        max_time_seconds=60,
        elapsed_seconds=5,
        fetches_used=3,
        max_fetches=100,
    )
    time_budget = render.render_budget(time_state)
    assert "time stopping rule" in time_budget
    assert "5s/1m00s" in time_budget
    assert "3/100" in time_budget

    convergence = render.render_budget(
        RunViewState(budget_type="convergence", best_quality=0.7, fetches_used=3, max_fetches=100)
    )
    assert "convergence stopping rule" in convergence
    assert "best Q 0.700" in convergence
    assert "3/100" in convergence


def test_tabs_expose_caste_score_delta_state_and_winner(view_state: RunViewState) -> None:
    tabs = render.render_tabs(view_state, 0)
    assert "A01" in tabs
    assert "foundations" in tabs
    assert "Q=0.700" in tabs
    assert "d=+0.200" in tabs
    assert "*" in tabs
    assert ">" in tabs
    assert "waiting" in tabs


def test_tabs_empty_state_explains_pending_colony() -> None:
    tabs = render.render_tabs(RunViewState(), 0)
    assert "waiting for colony initialization" in tabs


def test_timeline_empty_state() -> None:
    timeline = render.render_timeline(RunViewState(), None)
    assert "waiting for the first wave" in timeline


def test_timeline_groups_wave_turn_and_steps(view_state: RunViewState) -> None:
    selected = view_state.timeline[-1].entry_id
    timeline = render.render_timeline(view_state, selected)
    assert "Wave 1" in timeline
    assert "A01 fundaciones" in timeline
    assert "read Paper One [ref]" in timeline
    assert "evaluate Q=0.700" in timeline
    assert any(line.startswith(">") for line in timeline.splitlines())


def test_detail_exposes_all_four_sections(view_state: RunViewState) -> None:
    detail = render.render_detail(view_state, view_state.timeline[-1])
    for heading in (
        "-- Current activity --",
        "-- Selection rationale --",
        "-- Paper contribution --",
        "-- Research line --",
    ):
        assert heading in detail
    assert "live head" in detail
    assert "mode ref" in detail
    assert "provider openalex" in detail
    assert "exploit" in detail
    assert "pheromone 1.0000" in detail
    assert (
        "eta components: sim=0.5 citations=0.4 recency=unavailable "
        "confidence=unavailable llm=unavailable" in detail
    )
    assert "A concise summary" in detail
    assert "Body text" in detail


def test_detail_marks_historical_snapshot(view_state: RunViewState) -> None:
    projection = RunProjection(view_state.model_copy(deep=True))
    projection.select_by_index(0)
    detail = render.render_detail(projection.state, projection.selected_entry())
    assert "historical snapshot" in detail


def test_detail_without_selection_reports_no_decision() -> None:
    state = RunProjection.from_events(
        [
            RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "r"}),
            RunEvent(seq=2, type="oleada_start", payload={"oleada": 1}),
            RunEvent(seq=3, type="agent_turn_start", payload={"agent_id": "a0", "oleada": 1}),
        ]
    ).state
    detail = render.render_detail(state, state.timeline[-1])
    assert "no recorded decision for the selected step" in detail


def test_detail_marks_missing_eta_components_unavailable() -> None:
    state = RunProjection.from_events(
        [
            RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "r"}),
            RunEvent(seq=2, type="oleada_start", payload={"oleada": 1}),
            RunEvent(seq=3, type="agent_turn_start", payload={"agent_id": "a0", "oleada": 1, "turn": 0}),
            RunEvent(
                seq=4,
                type="candidate_selected",
                payload={"agent_id": "a0", "paper_id": "px", "mode": "ref", "chosen": True},
            ),
        ]
    ).state
    detail = render.render_detail(state, state.timeline[-1])
    assert "eta components: unavailable" in detail


def test_evaluation_view_exposes_full_breakdown(view_state: RunViewState) -> None:
    body = render.render_evaluation(view_state, "a0")
    assert "=== Evaluation: a0 wave=1 turn=0 ===" in body
    assert "Q=0.7000" in body
    assert "old=0.5000" in body
    assert "delta=+0.2000" in body
    assert "weights S=0.25; P=0.25; J=0.25; R=0.25" in body
    assert "S=0.7000" in body
    assert "P=0.6000" in body
    assert "J=0.7000" in body
    assert "R=0.5000" in body
    assert "self rationale: self good" in body
    assert "peer agent-1: 0.6000 -- peer ok" in body
    assert "virgin coverage: covers" in body
    assert "virgin gaps: gaps here" in body
    assert "structural coverage=0.500 diversity=0.400" in body


def test_evaluation_empty_state(view_state: RunViewState) -> None:
    body = render.render_evaluation(view_state, "nobody")
    assert "no detailed evaluation recorded" in body


def test_frontier_orders_by_recorded_eta_and_marks_the_chosen(view_state: RunViewState) -> None:
    body = render.render_frontier(view_state, "a0")
    assert "(2 candidates)" in body
    assert body.index("openalex:W2") < body.index("openalex:W1")
    assert "[chosen]" in body
    assert "[candidate]" in body
    assert "pheromone=1.0000" in body
    assert "weight=0.21600" in body
    assert "probability=0.8000" in body


def test_frontier_empty_state(view_state: RunViewState) -> None:
    body = render.render_frontier(view_state, "nobody")
    assert "no recorded frontier candidates" in body


def test_paper_full_renders_structured_fields(view_state: RunViewState) -> None:
    body = render.render_paper_full(view_state, "a0", "openalex:W1")
    for field in (
        "summary:",
        "key_concepts:",
        "methods:",
        "findings:",
        "relevance:",
        "limitations:",
        "key_references:",
    ):
        assert field in body
    assert "A concise summary" in body


def test_paper_full_missing_analysis(view_state: RunViewState) -> None:
    body = render.render_paper_full(view_state, "a0", "openalex:missing")
    assert "no structured analysis recorded" in body


def test_narrative_full_and_empty(view_state: RunViewState) -> None:
    body = render.render_narrative_full(view_state, "a0")
    assert "Body text" in body
    empty = render.render_narrative_full(view_state, "nobody")
    assert "no research line recorded" in empty


def test_events_view_filters_by_agent_and_omits_content(view_state: RunViewState) -> None:
    all_events = render.render_events(view_state)
    assert "orchestrator_start" in all_events
    assert "Body text" not in all_events
    filtered = render.render_events(view_state, "a0")
    assert "filtered to a0" in filtered
    assert "agent_turn_start" in filtered
    assert "no events recorded for this filter" in render.render_events(view_state, "nobody")


def test_metadata_lists_resolved_configuration(view_state: RunViewState) -> None:
    metadata = render.render_metadata(view_state)
    assert "run_id: run-1" in metadata
    assert "explorer_model: explorer-x" in metadata
    assert "judge_model: judge-y" in metadata
    assert "k_per_turn (papers per evaluation): 4" in metadata
    assert "question: How do things work?" in metadata


def test_metadata_redacts_failure_strings() -> None:
    state = RunViewState(failures=["boom api_key=sk-sentinel-1"], warnings=["token=sk-sentinel-2"])
    metadata = render.render_metadata(state)
    assert "sk-sentinel-1" not in metadata
    assert "sk-sentinel-2" not in metadata


def test_footer_reflects_terminal_state() -> None:
    assert "[completed] press q to exit" in render.render_footer(RunViewState(status="completed"))
    assert "[failed] press q to exit" in render.render_footer(RunViewState(status="failed"))
    running = render.render_footer(RunViewState(status="running"))
    assert "left/right agent" in running


def test_detail_renders_required_activity_metadata_and_operation_state() -> None:
    state = RunProjection.from_events(
        [
            RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "r"}),
            RunEvent(seq=2, type="oleada_start", payload={"oleada": 1, "active": ["a0"]}),
            RunEvent(seq=3, type="agent_turn_start", payload={"agent_id": "a0", "oleada": 1, "turn": 0}),
            RunEvent(
                seq=4,
                type="paper_fetch_completed",
                payload={
                    "agent_id": "a0",
                    "paper_id": "openalex:W1",
                    "title": "Paper One",
                    "year": 2021,
                    "authors": ["Ada", "Bob"],
                    "src": "10.1/seed",
                    "provider": "openalex",
                    "mode": "ref",
                    "turn": 0,
                },
            ),
            RunEvent(
                seq=5,
                type="llm_operation_completed",
                payload={"purpose": "paper_integration", "model": "explorer-x", "elapsed": 2.0},
            ),
            RunEvent(
                seq=6,
                type="agent_turn_complete",
                payload={"agent": "a0", "oleada": 1, "turn": 0, "budget": 9, "frontier": 7},
            ),
        ]
    ).state
    detail = render.render_detail(state, state.timeline[-1])
    assert "year 2021" in detail
    assert "authors Ada, Bob" in detail
    assert "source 10.1/seed" in detail
    assert "operation paper_integration" in detail
    assert "model explorer-x" in detail
    assert "elapsed 2s" in detail
    assert "budget remaining 9  frontier 7" in detail


def test_detail_marks_unavailable_activity_fields_explicitly() -> None:
    state = RunProjection.from_events(
        [
            RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "r"}),
            RunEvent(seq=2, type="oleada_start", payload={"oleada": 1}),
            RunEvent(seq=3, type="agent_turn_start", payload={"agent_id": "a0", "oleada": 1, "turn": 0}),
        ]
    ).state
    detail = render.render_detail(state, state.timeline[-1])
    assert "year unavailable" in detail
    assert "authors unavailable" in detail
    assert "source unavailable" in detail
    assert "mode unavailable" in detail
    assert "provider unavailable" in detail
    assert "operation unavailable" in detail
    assert "elapsed unavailable" in detail
    assert "frontier 0" in detail


def test_events_view_filters_by_outcome() -> None:
    state = RunProjection.from_events(
        [
            RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "r"}),
            RunEvent(
                seq=2,
                type="provider_failure",
                payload={"paper_id": "p", "reason": "429 timeout", "classification": "transient"},
            ),
            RunEvent(seq=3, type="run_failed", payload={"error": "boom"}),
            RunEvent(seq=4, type="agent_turn_start", payload={"agent_id": "a0", "turn": 0}),
        ]
    ).state
    errors = render.render_events(state, outcome="error")
    assert "outcome=error" in errors
    assert "run_failed" in errors
    assert "provider_failure" not in errors
    warnings = render.render_events(state, outcome="warning")
    assert "provider_failure" in warnings
    assert "run_failed" not in warnings
    assert render.classify_event_outcome(state.events[-1]) == "info"
    assert render.next_event_outcome("") == "error"
    assert render.next_event_outcome("info") == ""


def test_help_lists_every_required_binding() -> None:
    help_text = render.render_help()
    for phrase in (
        "Left/Right",
        "Up/Down",
        "Enter",
        "Home",
        "detailed evaluation",
        "frontier / candidate ranking",
        "selected-paper analysis",
        "agent narrative",
        "events filtered",
        "metadata and resolved configuration",
        "toggle timeline/detail",
        "Ctrl+C",
    ):
        assert phrase in help_text
