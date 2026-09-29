"""Terminal examination phases in the wave-first TUI hierarchy (TUI-1).

PRD §4 cases 19 and 21: private answer keys never reach TUI state, and the exam
phases project live while replay reaches the equivalent terminal navigation.
"""

from __future__ import annotations

from research_explorer.events.models import RunEvent
from research_explorer.events.navigation import EXAM_NODE, build_wave_navigation
from research_explorer.events.projection import RunProjection
from research_explorer.events.sink import CallbackSink
from research_explorer.replay.trace import RunTracer, RunTraceStore
from research_explorer.tui import text as render


def _exam_events() -> list[RunEvent]:
    return [
        RunEvent(seq=1, type="orchestrator_start", payload={
            "run_id": "exam-run", "seed": "arxiv:seed", "colony_size": 2}),
        RunEvent(seq=2, type="evidence_pack_frozen", payload={
            "pack_id": "pack-1", "pack_hash": "h1", "source_count": 3,
            "excluded_count": 1, "exclusion_reasons": ["metadata_only"]}),
        RunEvent(seq=3, type="exam_generated", payload={
            "item_count": 11, "exam_version": "exam/1"}),
        RunEvent(seq=4, type="exam_validated", payload={
            "accepted": 10, "rejected": 1,
            "rejection_reasons": ["multiple_defensible_options"]}),
        RunEvent(seq=5, type="exam_partitioned", payload={
            "partition_seed": 7, "selection_count": 6, "holdout_count": 4}),
        RunEvent(seq=6, type="candidate_test_completed", payload={
            "agent_id": "a0", "partition": "selection"}),
        RunEvent(seq=7, type="survivor_selected", payload={
            "survivor_id": "a0", "terminal_score": 0.8, "selection_accuracy": 0.7,
            "process_score": 0.6, "grounding_score": 0.5, "ranking": ["a0", "a1"]}),
        RunEvent(seq=8, type="survivor_frozen", payload={
            "survivor_id": "a0", "state_hash": "abc123"}),
        RunEvent(seq=9, type="baseline_completed", payload={
            "survivor_accuracy": 0.75, "naive_accuracy": 0.25, "uplift": 0.5}),
        RunEvent(seq=10, type="benchmark_completed", payload={
            "outcome": "completed_benchmarked", "reason_code": "", "reason": "",
            "survivor": "a0", "survivor_accuracy": 0.75, "naive_accuracy": 0.25,
            "uplift": 0.5}),
    ]


_FORBIDDEN = ("correct_option_id", "rationale", "answer_key", "evidence_refs")


def _assert_no_keys(blob: str) -> None:
    for forbidden in _FORBIDDEN:
        assert forbidden not in blob


def test_exam_phases_appear_in_order_in_the_wave_navigation() -> None:
    state = RunProjection.from_events(_exam_events()).state
    roots = build_wave_navigation(state)
    exam_nodes = [node for node in roots if node.kind == EXAM_NODE]
    assert [node.label for node in exam_nodes] == [
        "Exam build", "Selection", "Survivor", "Benchmark"
    ]
    assert [node.status for node in exam_nodes] == [
        "completed", "completed", "completed", "completed"
    ]
    selection = next(n for n in exam_nodes if n.label == "Selection")
    assert selection.detail["completed"] == ["a0"]
    benchmark = next(n for n in exam_nodes if n.label == "Benchmark")
    assert benchmark.detail["phase"] == "benchmark"


def test_tui_state_and_exam_views_never_expose_private_keys() -> None:
    state = RunProjection.from_events(_exam_events()).state
    _assert_no_keys(state.model_dump_json())
    for node in build_wave_navigation(state):
        _assert_no_keys(node.model_dump_json())
        rendered = render.render_exam_tab(state, node) if node.kind == EXAM_NODE else ""
        _assert_no_keys(rendered)
    benchmark_node = next(
        n for n in build_wave_navigation(state) if n.label == "Benchmark"
    )
    rendered = render.render_exam_tab(state, benchmark_node)
    assert "research uplift" in rendered
    assert "0.75" in rendered and "0.25" in rendered


def test_keys_absent_while_selection_is_active() -> None:
    partial = [e for e in _exam_events() if e.seq <= 6]
    state = RunProjection.from_events(partial).state
    _assert_no_keys(state.model_dump_json())
    selection = next(
        n for n in build_wave_navigation(state) if n.label == "Selection"
    )
    rendered = render.render_exam_tab(state, selection)
    _assert_no_keys(rendered)
    assert "selection items: 6" in rendered
    assert "holdout items: 4" in rendered


def test_live_and_replay_exam_navigation_match(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "replay.db")
    run_id = store.create_run("arxiv:seed", "scope")
    live_events: list[RunEvent] = []
    tracer = RunTracer(store, run_id, sink=CallbackSink(live_events.append))
    for event in _exam_events():
        tracer.emit(event.type, **event.payload)

    live = RunProjection.from_events(live_events).state
    persisted = [
        RunEvent(seq=e["seq"], type=e["type"], payload=e["payload"], ts=e["ts"])
        for e in store.list_events(run_id)
    ]
    replayed = RunProjection.from_events(persisted).state

    def signature(state) -> list[tuple[str, str, str]]:
        return [
            (node.node_id, node.label, node.status)
            for node in build_wave_navigation(state)
        ]

    assert signature(live) == signature(replayed)
    replayed_ids = {node_id: (label, status) for node_id, label, status in signature(replayed)}
    assert replayed_ids["exam:benchmark"] == ("Benchmark", "completed")
    store.close()
