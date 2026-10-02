"""Examination projection: live and replay reach the same terminal state (TUI-1)."""

from __future__ import annotations

from research_explorer.events.models import RunEvent
from research_explorer.events.projection import RunProjection
from research_explorer.events.sink import CallbackSink
from research_explorer.examination.events import (
    evidence_pack_frozen_payload,
    exam_payloads,
    survivor_payloads,
)
from research_explorer.replay.trace import RunTracer, RunTraceStore


def _emit_exam_events(tracer: RunTracer) -> None:
    from research_explorer.examination.models import EvidenceEntry, EvidencePack, ExamBank
    from research_explorer.memory.models import ContentKind

    pack = EvidencePack(
        pack_id="pack-1",
        seed_paper_id="arxiv:seed",
        sources=[
            EvidenceEntry(
                source_id="arxiv:seed",
                title="Seed",
                content_kind=ContentKind.FULL_TEXT,
                content_hash="h1",
            )
        ],
    ).freeze()
    tracer.emit("evidence_pack_frozen", **evidence_pack_frozen_payload(pack))
    bank = ExamBank(
        selection_ids=["q-1"],
        holdout_ids=["q-2"],
        partition_seed=7,
        accepted_count=2,
        rejected_count=1,
    )
    for event_type, payload in exam_payloads(bank, accepted=2, rejected=1, rejection_reasons=["multiple"]):
        tracer.emit(event_type, **payload)
    tracer.emit("candidate_test_completed", agent_id="a0", partition="selection")

    from research_explorer.examination.benchmark import BenchmarkResult
    from research_explorer.examination.models import (
        CategoryScore,
        ExamScore,
        SurvivorSelection,
    )

    survivor_score = ExamScore(
        correct=1,
        total=1,
        item_outcomes={"q-2": True},
        by_category={"concepts_definitions": CategoryScore(correct=1, total=1)},
        by_difficulty={"easy": CategoryScore(correct=1, total=1)},
    )
    result = BenchmarkResult(
        survivor_id="a0",
        outcome="completed_benchmarked",
        survivor_accuracy=0.75,
        naive_accuracy=0.25,
        uplift=0.5,
        survivor_score=survivor_score,
        selection=SurvivorSelection(
            survivor_id="a0", eligible=True, terminal_score=0.8, selection_accuracy=0.7
        ),
    )
    for event_type, payload in survivor_payloads(result):
        tracer.emit(event_type, **payload)


def test_exam_projection_live_matches_replay(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "replay.db")
    run_id = store.create_run("arxiv:seed", "scope")

    live_events: list[RunEvent] = []
    tracer = RunTracer(store, run_id, sink=CallbackSink(live_events.append))
    tracer.emit("orchestrator_start", run_id=run_id, seed="arxiv:seed", colony_size=2)
    _emit_exam_events(tracer)

    live = RunProjection.from_events(live_events)
    persisted = [
        RunEvent(seq=e["seq"], type=e["type"], payload=e["payload"], ts=e["ts"])
        for e in store.list_events(run_id)
    ]
    replayed = RunProjection.from_events(persisted)

    def stable(projection: RunProjection) -> dict:
        data = projection.state.model_dump(mode="json")
        data.pop("events", None)
        data.pop("started_at", None)
        return data

    assert stable(live) == stable(replayed)
    assert replayed.state.survivor_id == "a0"
    assert replayed.state.survivor_accuracy == 0.75
    assert replayed.state.naive_accuracy == 0.25
    assert replayed.state.uplift == 0.5
    assert replayed.state.exam_selection_count == 1
    assert replayed.state.exam_holdout_count == 1
    assert replayed.state.exam_rejected_count == 1
    assert replayed.state.current_phase == "benchmark"
    assert replayed.state.exam_phases["exam_build"].status == "completed"
    assert replayed.state.exam_phases["survivor"].leader == "a0"
    assert replayed.state.benchmark_item_outcomes == {"q-2": True}
    assert replayed.state.benchmark_by_category == {
        "concepts_definitions": {"correct": 1, "total": 1}
    }
    assert replayed.state.benchmark_by_difficulty == {"easy": {"correct": 1, "total": 1}}
    from research_explorer.tui import text as render

    rendered = render.render_final_result(replayed.state)
    assert "Public category outcomes" in rendered
    assert "concepts definitions: 1/1" in rendered
    assert "q-2: correct" in rendered
    for forbidden in ("correct_option_id", "rationale", "answer_key", "evidence_refs"):
        assert forbidden not in rendered
    store.close()


def test_legacy_projection_without_exam_events_is_unchanged() -> None:
    events = [
        RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "r", "seed": "s"}),
        RunEvent(seq=2, type="orchestrator_complete", payload={"status": "completed"}),
    ]
    state = RunProjection.from_events(events).state
    assert state.survivor_id == ""
    assert state.benchmark_outcome == ""
    assert state.exam_phases == {}
