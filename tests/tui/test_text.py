"""Semantic renderer tests for the Convoy-style dashboard.

These target stable text produced from the projected run state rather than
brittle full-screen snapshots. Textual interaction is covered by the pilot
tests in ``test_app.py``.
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
from research_explorer.events.navigation import build_agent_navigation, find_node
from research_explorer.events.projection import RunProjection
from research_explorer.tui import text as render
from research_explorer.tui.session import (
    TAB_EVALUATION,
    TAB_EVENTS,
    UISession,
    next_tab,
)


def test_ellipsize_and_duration_units() -> None:
    assert render.ellipsize("abcdef", 4) == "abc\u2026"
    assert render.ellipsize("abcdef", 0) == ""
    assert render.format_duration(5) == "5s"
    assert render.format_duration(65) == "1m05s"
    assert render.format_duration(3661) == "1h01m01s"


def test_caste_and_lifecycle_markers_are_text_not_color() -> None:
    assert render.caste_label("fundaciones") == "foundations"
    assert render.caste_label("impacto") == "impact"
    assert render.caste_label("") == "mixed"
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


def test_context_header_shows_brand_identity_and_models(view_state: RunViewState) -> None:
    header = str(render.render_dashboard_header(view_state, width=120))
    assert "Research Explorer" in header
    assert "run-1" in header
    assert "explorer-x" in header
    assert "judge-y" in header
    assert "wave 1" in header
    assert "best 0.700" in header
    assert "fetch 3/100" in header
    assert "papers/eval 4" in header


def test_header_marks_unknown_tokens_unavailable() -> None:
    state = RunProjection.from_events(
        [RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "r"})]
    ).state
    header = str(render.render_dashboard_header(state))
    assert "tokens unavailable" in header
    assert "tokens 0" not in header


def test_historical_header_marks_snapshot(view_state: RunViewState) -> None:
    header = str(render.render_dashboard_header(view_state, follow_live=False))
    assert "historical snapshot" in header


def test_agent_tree_label_reports_caste_quality_delta_winner_and_state(
    view_state: RunViewState,
) -> None:
    roots = build_agent_navigation(view_state)
    label = str(
        render.render_tree_label(
            view_state, roots[0], selected=True, expanded=True, depth=0
        )
    )
    assert "A01" in label
    assert "foundations" in label
    assert "Q=0.700" in label
    assert "\u0394+0.200" in label
    assert "\u2605" in label
    assert render.theme.SELECTION_MARK in label


def test_agent_tree_nests_waves_turns_and_papers(view_state: RunViewState) -> None:
    roots = build_agent_navigation(view_state)
    wave = find_node(roots, "agent:a0:wave:1")
    assert wave is not None and wave.children
    turn = find_node(roots, "turn:1:a0:0")
    assert turn is not None
    assert any(child.label.startswith("read") for child in turn.children)
    evaluation = find_node(roots, "eval:1:a0:0")
    assert evaluation is not None


def test_agent_roster_marks_winner_and_status(view_state: RunViewState) -> None:
    roster = str(render.render_agent_roster(view_state))
    assert "A01" in roster
    assert "\u2605" in roster


def test_activity_card_reports_action_paper_operation_and_budget(
    view_state: RunViewState,
) -> None:
    entry = view_state.timeline[-1]
    card = str(render.render_activity_card(view_state, "a0", entry, width=60))
    assert "action" in card
    assert "Paper One" in card
    assert "budget" in card
    assert "frontier" in card
    assert "[live]" in card


def test_tab_bar_marks_one_active_tab(view_state: RunViewState) -> None:
    session = UISession()
    bar = str(render.render_tab_bar(session))
    assert "Research" in bar and "Paper" in bar and "Evaluation" in bar and "Events" in bar
    assert render.theme.SELECTION_MARK in bar
    session.select_tab(TAB_EVENTS)
    assert "Events" in str(render.render_tab_bar(session))


def test_tab_cycling_wraps() -> None:
    assert next_tab("research", 1) == "paper"
    assert next_tab("research", -1) == "events"


def test_footer_reflects_terminal_state_and_compact_hints(
    view_state: RunViewState,
) -> None:
    session = UISession()
    running = str(render.render_footer_text(view_state, session))
    assert "Research Explorer" in running
    assert "quit" in running
    terminal = str(render.render_footer_text(RunViewState(status="completed"), session))
    assert "completed" in terminal
    assert "exit" in terminal
    session.narrowed = True
    compact = str(render.render_footer_text(view_state, session))
    assert "toggle panes" in compact


def test_research_tab_renders_narrative_and_delta(view_state: RunViewState) -> None:
    body = render.render_research_tab(view_state, "a0", view_state.timeline[-1])
    assert "Research line" in body
    assert "Body text" in body


def test_paper_tab_renders_metadata_analysis_rationale_and_frontier(
    view_state: RunViewState,
) -> None:
    entry = next(
        e for e in view_state.timeline if e.kind == "paper" and e.detail.get("analysis")
    )
    body = render.render_paper_tab(view_state, "a0", entry)
    assert "Paper" in body
    assert "title: Paper One" in body
    assert "Selection rationale" in body
    assert "Structured analysis" in body
    assert "A concise summary" in body
    assert "Decision-time frontier" in body
    assert "openalex:W2" in body


def test_evaluation_tab_exposes_full_breakdown(view_state: RunViewState) -> None:
    body = render.render_evaluation_tab(view_state, "a0")
    assert "Q=0.7000" in body
    assert "S=0.7000" in body and "P=0.6000" in body
    assert "self rationale: self good" in body
    assert "peer agent-1" in body
    assert "judge coverage: covers" in body
    assert "diversity=0.400" in body


def test_evaluation_tab_empty_explains_skipped_semantics(view_state: RunViewState) -> None:
    body = render.render_evaluation_tab(view_state, "nobody")
    assert "No detailed evaluation" in body
    assert "not a zero score" in body


def test_events_tab_is_reverse_chronological_and_filterable(
    view_state: RunViewState,
) -> None:
    body = render.render_events_tab(view_state)
    assert body.index("budget_snapshot") < body.index("orchestrator_start")
    assert "Body text" not in body
    errors = render.render_events_tab(view_state, outcome="error")
    assert "No events recorded" in errors


def test_tab_body_dispatch_matches_active_tab(view_state: RunViewState) -> None:
    session = UISession()
    assert "Research line" in render.render_tab_body(view_state, session, None)
    session.select_tab(TAB_EVALUATION)
    assert "Evaluation" in render.render_tab_body(view_state, session, None)


def test_failure_summary_redacts_secrets() -> None:
    state = RunViewState(
        run_id="r", status="failed", failures=["boom api_key=sk-sentinel-1"]
    )
    summary = render.render_failure_summary(state)
    assert "Run r" in summary
    assert "sk-sentinel-1" not in summary


def _evaluation_detail(q: float, turn: int) -> dict:
    return {
        "agent_id": "a0",
        "oleada": 1,
        "turn": turn,
        "q": q,
        "old_quality": q - 0.1,
        "delta_q": 0.1,
        "self_assessment": {"score": q, "reasoning": "self"},
    }


def test_evaluation_tab_shows_q_history_for_multiple_records() -> None:
    state = RunProjection.from_events(
        [
            RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "r"}),
            RunEvent(seq=2, type="evaluation_detail", payload={"detail": _evaluation_detail(0.5, 0)}),
            RunEvent(seq=3, type="evaluation_detail", payload={"detail": _evaluation_detail(0.7, 1)}),
        ]
    ).state
    body = render.render_evaluation_tab(state, "a0")
    assert "### Q history" in body
    assert "wave 1 turn 0" in body
    assert "wave 1 turn 1" in body
    assert "Q=0.7000" in body


def test_activity_card_handles_empty_and_compact_states(view_state: RunViewState) -> None:
    empty = str(render.render_activity_card(view_state, "nobody", None))
    assert "waiting for colony initialization" in empty
    compact = str(
        render.render_activity_card(
            view_state, "a0", view_state.timeline[-1], width=60, compact=True
        )
    )
    assert "[live]" in compact
    assert "Q " in compact
    assert "budget" in compact


def test_footer_reports_settling_cancellation(view_state: RunViewState) -> None:
    text = str(render.render_footer_text(view_state, UISession(), settling=True))
    assert "settling" in text


def test_research_tab_falls_back_to_latest_narrative() -> None:
    state = RunViewState(narratives={"__latest__": "# Latest line"})
    body = render.render_research_tab(state, "missing", None)
    assert "Latest line" in body


def test_paper_tab_empty_explains_absence() -> None:
    body = render.render_paper_tab(RunViewState(), "nobody", None)
    assert "_No paper selected yet._" in body


def test_budget_renderer_reports_time_and_convergence_rules() -> None:
    time_state = RunViewState(
        budget_type="time",
        max_time_seconds=60,
        elapsed_seconds=30,
        fetches_used=4,
        max_fetches=10,
    )
    assert "time stopping rule" in render.render_budget(time_state)
    convergence = RunViewState(budget_type="convergence", best_quality=0.8)
    assert "convergence stopping rule" in render.render_budget(convergence)


def test_legacy_events_renderer_filters_and_empty_state(view_state: RunViewState) -> None:
    body = render.render_events(view_state, outcome="error")
    assert "no events recorded" in body
    assert "budget_snapshot" in render.render_events(view_state)
