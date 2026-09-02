"""RunTraceStore tests: ordered events, evaluation records, safe artifacts."""

from __future__ import annotations

import pytest

from research_explorer.replay.models import (
    DetailedEvaluation,
    PeerVoteDetail,
    PeerVotesDetail,
    SelfAssessmentDetail,
    StructuralComponentsDetail,
    VirginJudgeDetail,
)
from research_explorer.replay.trace import (
    RunTracer,
    RunTraceStore,
    safe_artifact_path,
    safe_filename,
)


@pytest.fixture
def store(tmp_path) -> RunTraceStore:
    s = RunTraceStore(tmp_path / "replay.db")
    yield s
    s.close()


def _detail(agent: str = "agent-000-abc", reasoning: str = "r") -> DetailedEvaluation:
    return DetailedEvaluation(
        agent_id=agent,
        oleada=1,
        turn=1,
        q=0.5,
        self_assessment=SelfAssessmentDetail(score=0.5, reasoning=reasoning),
        peers=PeerVotesDetail(
            votes=[PeerVoteDetail(voter_id="p", score=0.5, reasoning="peer r")],
            aggregated_score=0.5,
            num_votes=1,
        ),
        virgin_judge=VirginJudgeDetail(score=0.5, coverage="cov", gaps="gap"),
        structural=StructuralComponentsDetail(coverage=0.5, r=0.5),
    )


def test_create_and_get_run(store):
    run_id = store.create_run("10.1038/nrn3241", "neuroscience")
    run = store.get_run(run_id)
    assert run["run_id"] == run_id
    assert run["seed_paper_id"] == "10.1038/nrn3241"
    assert run["seed_query"] == "neuroscience"
    assert run["status"] == "running"
    assert run["event_count"] == 0

    store.finish_run(run_id, "completed", best_quality=0.9)
    run = store.get_run(run_id)
    assert run["status"] == "completed"
    assert run["completed_at"] is not None
    assert run["best_quality"] == 0.9


def test_missing_run(store):
    assert store.get_run("nope") is None
    assert store.list_events("nope") == []


def test_ordered_events(store):
    run_id = store.create_run("seed", "q")
    s1 = store.append_event(run_id, "orchestrator_start", {"seed": "s"})
    s2 = store.append_event(run_id, "oleada_start", {"oleada": 1})
    s3 = store.append_event(run_id, "agent_step", {"paper": "p"})
    assert (s1, s2, s3) == (1, 2, 3)

    events = store.list_events(run_id)
    assert [e["seq"] for e in events] == [1, 2, 3]
    assert [e["type"] for e in events] == ["orchestrator_start", "oleada_start", "agent_step"]
    assert events[0]["payload"] == {"seed": "s"}


def test_evaluation_roundtrip(store):
    run_id = store.create_run("seed", "q")
    detail = _detail()
    store.save_evaluation(run_id, detail)
    store.save_evaluation(run_id, _detail(agent="agent-000-def", reasoning="x"))

    records = store.list_evaluations(run_id)
    assert len(records) == 2
    assert records[0].agent_id == "agent-000-abc"
    assert records[0].self_assessment.reasoning == "r"
    assert records[1].agent_id == "agent-000-def"

    summary = store.evaluations_summary(run_id)
    assert summary[0]["S"] == 0.5
    assert summary[0]["q"] == 0.5


def test_tracer_records_evaluation_and_event(store):
    run_id = store.create_run("seed", "q")
    tracer = RunTracer(store, run_id)
    tracer.record_evaluation(_detail())

    assert len(store.list_evaluations(run_id)) == 1
    events = store.list_events(run_id)
    assert events[0]["type"] == "evaluation_complete"
    assert events[0]["payload"]["agent_id"] == "agent-000-abc"


def test_tracer_records_artifact(store):
    run_id = store.create_run("seed", "q")
    tracer = RunTracer(store, run_id)
    artifact_id = tracer.record_artifact("narrative_agent-000-abc.md", "narrative", "text")
    assert store.get_artifact(artifact_id)["kind"] == "narrative"
    events = store.list_events(run_id)
    assert events[0]["type"] == "artifact_saved"
    latest = store.get_latest_artifact(run_id, "narrative")
    assert latest["content"] == "text"


def test_artifact_path_traversal_blocked(store, tmp_path):
    run_id = store.create_run("seed", "q")
    with pytest.raises(ValueError):
        store.save_artifact(run_id, "../../../etc/passwd", "narrative", "boom")
    assert store.list_artifacts(run_id) == []
    with pytest.raises(KeyError):
        store.export_artifact("missing", tmp_path)


def test_safe_filename_rejects_traversal():
    for bad in ["..", "../x", "a/../../b", "\\..\\evil", ""]:
        with pytest.raises(ValueError):
            safe_filename(bad)
    assert safe_filename("/etc/passwd") == "passwd"
    assert safe_filename("sub/dir/file.md") == "file.md"
    assert safe_filename("weird\\name.txt") == "name.txt"


def test_safe_artifact_path_stays_inside(tmp_path):
    out = tmp_path / "out"
    target = safe_artifact_path(out, "narrative.md")
    assert target.parent == out.resolve()
    assert target.name == "narrative.md"
    with pytest.raises(ValueError):
        safe_artifact_path(out, "sub/../../escape.md")


def test_export_narrative_with_safe_name(store, tmp_path):
    run_id = store.create_run("seed", "q")
    store.save_artifact(run_id, "my_narrative.md", "narrative", "the content")
    target = store.export_narrative(run_id, tmp_path)
    assert target.name == "my_narrative.md"
    assert target.read_text() == "the content"
    with pytest.raises(ValueError):
        store.export_narrative(run_id, tmp_path, filename="../../evil.md")


def test_export_narrative_missing(store, tmp_path):
    run_id = store.create_run("seed", "q")
    with pytest.raises(KeyError):
        store.export_narrative(run_id, tmp_path)
