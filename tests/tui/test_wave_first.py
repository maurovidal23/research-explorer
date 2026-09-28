"""Wave-first TUI hierarchy, live summary, and final-result defaults."""

from __future__ import annotations

from research_explorer.events.models import (
    OUTCOME_DEGRADED,
    REASON_EMPTY_WINNER_NARRATIVE,
    RunEvent,
)
from research_explorer.events.projection import RunProjection
from research_explorer.tui import build_app


def _events(terminal: bool) -> list[RunEvent]:
    events = [
        RunEvent(seq=1, type="orchestrator_start", payload={
            "run_id": "wave-run", "seed": "arxiv:1", "colony_size": 2, "K": 2,
            "k_per_turn": 1, "max_fetches": 10,
        }),
        RunEvent(seq=2, type="colony_initialized", payload={"size": 2, "agents": ["a0", "a1"]}),
        RunEvent(seq=3, type="oleada_start", payload={"oleada": 1, "active": ["a0", "a1"]}),
        RunEvent(seq=4, type="agent_turn_start", payload={
            "agent_id": "a0", "oleada": 1, "turn": 0}),
        RunEvent(seq=5, type="paper_integration_completed", payload={
            "agent_id": "a0", "paper_id": "p1", "title": "Paper One",
            "oleada": 1, "turn": 0}),
        RunEvent(seq=6, type="agent_turn_complete", payload={
            "agent_id": "a0", "agent": "a0", "oleada": 1, "turn": 0,
            "fetches": 1, "budget": 3, "frontier": 1}),
        RunEvent(seq=7, type="wave_phase_completed", payload={
            "oleada": 1, "phase": "research", "selected": ["a0", "a1"],
            "completed": ["a0"], "failed": [], "skipped": ["a1"],
            "papers_attempted": 1, "papers_integrated": 1, "evidence_added": 1}),
        RunEvent(seq=8, type="wave_phase_started", payload={
            "oleada": 1, "phase": "evaluation", "selected": ["a0", "a1"]}),
        RunEvent(seq=9, type="evaluation_settled", payload={
            "agent_id": "a0", "oleada": 1, "turn": 0, "status": "complete",
            "Q": 0.4, "delta_q": 0.4, "S": 0.3, "P": 0.4, "J": 0.5, "R": 0.4,
            "unavailable": {}, "evidence_papers": 1}),
        RunEvent(seq=10, type="evaluation_settled", payload={
            "agent_id": "a1", "oleada": 1, "turn": 0, "status": "skipped",
            "reason": "no_new_evidence", "evidence_papers": 0}),
        RunEvent(seq=11, type="wave_phase_completed", payload={
            "oleada": 1, "phase": "evaluation", "selected": ["a0", "a1"],
            "completed": ["a0"], "failed": [], "skipped": ["a1"], "best_Q": 0.4}),
        RunEvent(seq=12, type="wave_phase_started", payload={
            "oleada": 1, "phase": "decision", "selected": ["a0", "a1"]}),
        RunEvent(seq=13, type="wave_phase_completed", payload={
            "oleada": 1, "phase": "decision", "selected": ["a0", "a1"],
            "leader": "a0", "best_Q": 0.4, "q_delta": 0.4,
            "ranking": [["a0", 0.4], ["a1", 0.0]], "budget_used": 1,
            "continue_reason": "budget_remaining"}),
        RunEvent(seq=14, type="oleada_complete", payload={
            "oleada": 1, "best_Q": 0.4, "total_fetches": 1, "max_fetches": 10,
            "leader": "a0", "q_delta": 0.4}),
    ]
    if terminal:
        events.append(
            RunEvent(seq=15, type="orchestrator_complete", payload={
                "run_id": "wave-run", "status": "completed",
                "outcome": OUTCOME_DEGRADED,
                "reason_code": REASON_EMPTY_WINNER_NARRATIVE,
                "reason": "the winning agent produced no usable narrative",
                "winner": "a0", "peak_Q": 0.4, "stop_reason": "budget_exhausted",
            })
        )
    return events


async def test_live_summary_shows_phase_progress_leader_and_budget() -> None:
    app = build_app(RunProjection.from_events(_events(terminal=False)))
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        header = app.query_one("#header").plain_text
        assert "phase" in header
        assert "leader" in header
        assert "best 0.400" in header
        assert "fetch 1/10" in header
        rows = app.tree_row_texts()
        assert any("Setup" in row for row in rows)
        assert any("Wave 1" in row for row in rows)
        assert any("Final result" in row for row in rows)
        assert any("Debug" in row for row in rows)
        wave_row = next(row for row in rows if "Wave 1" in row)
        assert "research" in wave_row
        assert "eval" in wave_row
        assert "Δ" in wave_row
        assert "budget_remaining" in wave_row


async def test_terminal_defaults_to_final_result_and_keeps_wave_drilldown() -> None:
    app = build_app(RunProjection.from_events(_events(terminal=True)))
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        assert app.session.selected_node_id == "final"
        content = str(app.query_one("#content").source)
        assert "Final result" in content
        assert "No winning narrative" in content
        assert "degraded" in app.query_one("#footer").plain_text
        rows = app.tree_row_texts()
        assert any("Wave 1" in row for row in rows)
        assert any("Setup" in row for row in rows)
        assert any("Debug" in row for row in rows)
