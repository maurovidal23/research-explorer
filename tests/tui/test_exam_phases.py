"""Terminal examination phases in the live Textual app (TUI-1).

Drives the real live consumer with a frozen, network-free exam lifecycle and
asserts the phases are visible, reviewable, and free of private keys.
"""

from __future__ import annotations

from research_explorer.events.models import RunEvent
from research_explorer.events.projection import RunProjection
from research_explorer.tui import build_app


def _exam_events() -> list[RunEvent]:
    return [
        RunEvent(seq=1, type="orchestrator_start", payload={
            "run_id": "exam-run", "seed": "arxiv:seed", "colony_size": 2,
            "K": 2, "k_per_turn": 1, "max_fetches": 8,
            "explorer_model": "explorer-x", "judge_model": "judge-y"}),
        RunEvent(seq=2, type="colony_initialized", payload={"size": 2, "agents": ["a0", "a1"]}),
        RunEvent(seq=3, type="evidence_pack_frozen", payload={
            "pack_id": "pack-1", "pack_hash": "h1", "source_count": 3,
            "excluded_count": 1}),
        RunEvent(seq=4, type="exam_generated", payload={
            "item_count": 11, "exam_version": "exam/1"}),
        RunEvent(seq=5, type="exam_validated", payload={
            "accepted": 10, "rejected": 1,
            "rejection_reasons": ["multiple_defensible_options"]}),
        RunEvent(seq=6, type="exam_partitioned", payload={
            "partition_seed": 7, "selection_count": 6, "holdout_count": 4}),
        RunEvent(seq=7, type="candidate_test_completed", payload={
            "agent_id": "a0", "partition": "selection"}),
        RunEvent(seq=8, type="survivor_selected", payload={
            "survivor_id": "a0", "terminal_score": 0.8, "selection_accuracy": 0.7,
            "process_score": 0.6, "grounding_score": 0.5, "ranking": ["a0", "a1"]}),
        RunEvent(seq=9, type="survivor_frozen", payload={
            "survivor_id": "a0", "state_hash": "abc123"}),
        RunEvent(seq=10, type="baseline_completed", payload={
            "survivor_accuracy": 0.75, "naive_accuracy": 0.25, "uplift": 0.5}),
        RunEvent(seq=11, type="benchmark_completed", payload={
            "outcome": "completed_benchmarked", "reason_code": "", "reason": "",
            "survivor": "a0", "survivor_accuracy": 0.75, "naive_accuracy": 0.25,
            "uplift": 0.5}),
        RunEvent(seq=12, type="orchestrator_complete", payload={
            "run_id": "exam-run", "status": "completed",
            "outcome": "completed_benchmarked", "winner": "a0", "peak_Q": 0.4,
            "stop_reason": "budget_exhausted"}),
    ]


_FORBIDDEN = ("correct_option_id", "rationale", "answer_key", "evidence_refs")


async def test_tui_exposes_exam_phases_and_keeps_keys_hidden() -> None:
    app = build_app(RunProjection.from_events(_exam_events()))
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        rows = app.tree_row_texts()
        for label in ("Exam build", "Selection", "Survivor", "Benchmark"):
            assert any(label in row for row in rows), rows

        assert app.session.selected_node_id == "final"
        content = str(app.query_one("#content").source)
        assert "Research uplift" in content
        for forbidden in _FORBIDDEN:
            assert forbidden not in content

        benchmark = next(node for node in app._tree_nodes if node.node_id == "exam:benchmark")
        app._on_tree_select(benchmark)
        await pilot.pause()
        detail = str(app.query_one("#content").source)
        assert "## Examination · Benchmark" in detail
        assert "0.7500" in detail and "0.2500" in detail
        for forbidden in _FORBIDDEN:
            assert forbidden not in detail
