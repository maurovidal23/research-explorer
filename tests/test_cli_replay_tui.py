"""CLI tests for `research-explorer replay tui`."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from research_explorer.cli import app
from research_explorer.replay.trace import RunTracer, RunTraceStore

runner = CliRunner()

VOLATILE = {"events", "started_at"}


def _seed_run(db: Path) -> str:
    store = RunTraceStore(db)
    run_id = store.create_run("10.1/seed", "how?")
    tracer = RunTracer(store, run_id)
    tracer.emit(
        "orchestrator_start",
        run_id=run_id,
        seed="10.1/seed",
        query="how?",
        colony_size=1,
        K=1,
        k_per_turn=2,
        max_fetches=50,
    )
    tracer.emit("colony_initialized", size=1, agents=["a0"])
    tracer.emit("oleada_start", oleada=1, active=["a0"])
    tracer.emit("agent_turn_start", agent_id="a0", agent="a0", caste="mixto", oleada=1, turn=0)
    tracer.emit(
        "paper_integration_completed",
        agent_id="a0",
        paper_id="openalex:W1",
        title="P1",
        oleada=1,
        turn=0,
    )
    tracer.emit(
        "evaluation_complete",
        agent_id="a0",
        oleada=1,
        turn=0,
        Q=0.7,
        delta_q=0.7,
    )
    from research_explorer.replay.models import DetailedEvaluation, SelfAssessmentDetail

    tracer.record_evaluation(
        DetailedEvaluation(
            agent_id="a0",
            oleada=1,
            turn=0,
            q=0.7,
            delta_q=0.7,
            self_assessment=SelfAssessmentDetail(score=0.7, reasoning="self"),
        )
    )
    tracer.record_artifact("narrative_a0_t0.md", "narrative", "# Narrative\nBody")
    tracer.emit("orchestrator_complete", run_id=run_id, winner="a0", peak_Q=0.7, status="completed")
    store.finish_run(run_id, "completed", best_quality=0.7)
    store.close()
    return run_id


def test_replay_tui_missing_run_exits_nonzero(tmp_path) -> None:
    db = tmp_path / "replay.db"
    result = runner.invoke(app, ["replay", "tui", "does-not-exist", "--db", str(db)])
    assert result.exit_code != 0
    combined = result.output + str(result.exception or "")
    assert "Run not found" in combined


def test_replay_tui_opens_read_only_dashboard(tmp_path, monkeypatch) -> None:
    db = tmp_path / "replay.db"
    run_id = _seed_run(db)
    captured: dict = {}

    class _FakeApp:
        def __init__(self, projection, read_only=False):
            captured["projection"] = projection
            captured["read_only"] = read_only

        def run(self) -> None:
            return None

    monkeypatch.setattr("research_explorer.tui.build_app", _FakeApp)
    result = runner.invoke(app, ["replay", "tui", run_id, "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert captured["read_only"] is True
    state = captured["projection"].state
    assert state.run_id == run_id
    assert state.winner_agent == "a0"
    # Evaluation detail was hydrated from the evaluation_results table.
    assert state.evaluations["a0"][-1].self_assessment.reasoning == "self"
    # Narrative content was hydrated from the artifacts table.
    assert "Body" in state.narratives["a0"]


def test_replay_tui_hydration_does_not_duplicate_existing_detail(tmp_path, monkeypatch) -> None:
    db = tmp_path / "replay.db"
    run_id = _seed_run(db)
    captured: dict = {}

    class _FakeApp:
        def __init__(self, projection, read_only=False):
            captured["projection"] = projection

        def run(self) -> None:
            return None

    monkeypatch.setattr("research_explorer.tui.build_app", _FakeApp)
    runner.invoke(app, ["replay", "tui", run_id, "--db", str(db)])
    records = captured["projection"].state.evaluations["a0"]
    assert len(records) == 1


def test_replay_tui_projects_failed_and_cancelled_status(tmp_path, monkeypatch) -> None:
    captured: dict = {}

    class _FakeApp:
        def __init__(self, projection, read_only=False):
            captured["projection"] = projection

        def run(self) -> None:
            return None

    monkeypatch.setattr("research_explorer.tui.build_app", _FakeApp)
    for status, terminal_type in (("failed", "run_failed"), ("cancelled", "run_cancelled")):
        db = tmp_path / f"{status}.db"
        store = RunTraceStore(db)
        run_id = store.create_run("seed", "q")
        tracer = RunTracer(store, run_id)
        tracer.emit("orchestrator_start", run_id=run_id, colony_size=1, K=1)
        payload = {"error": "boom"} if status == "failed" else {"run_id": run_id}
        tracer.emit(terminal_type, **payload)
        store.finish_run(run_id, status)
        store.close()

        result = runner.invoke(app, ["replay", "tui", run_id, "--db", str(db)])
        assert result.exit_code == 0, result.output
        assert captured["projection"].state.status == status
