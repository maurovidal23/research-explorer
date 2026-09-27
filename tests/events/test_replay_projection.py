"""Live-vs-replay equivalence and trace integration for the projection."""

from __future__ import annotations

from types import SimpleNamespace

from research_explorer.agents.llm_client import LLMClient
from research_explorer.events.models import RunEvent
from research_explorer.events.projection import RunProjection
from research_explorer.events.sink import CallbackSink
from research_explorer.replay.models import (
    DetailedEvaluation,
    PeerVoteDetail,
    PeerVotesDetail,
    SelfAssessmentDetail,
    StructuralComponentsDetail,
    VirginJudgeDetail,
)
from research_explorer.replay.trace import RunTracer, RunTraceStore

VOLATILE = {"events", "started_at"}


def _stable_state(state) -> dict:
    data = state.model_dump(mode="json")
    for key in VOLATILE:
        data.pop(key, None)
    return data


def _drive(tracer: RunTracer, run_id: str) -> None:
    tracer.emit(
        "orchestrator_start",
        run_id=run_id,
        seed="seed-1",
        query="how?",
        colony_size=2,
        K=2,
        k_per_turn=3,
        max_fetches=50,
        explorer_model="explorer",
        judge_model="judge",
    )
    tracer.emit("oleada_start", oleada=1, active=["a0", "a1"], max_fetches=50)
    tracer.emit("agent_turn_start", agent_id="a0", agent="a0", caste="mixto", oleada=1, turn=0)
    tracer.emit(
        "paper_integration_completed",
        agent_id="a0",
        paper_id="openalex:W1",
        title="Paper One",
        mode="ref",
        turn=0,
        analysis={"summary": "s"},
    )
    tracer.emit(
        "neighbor_discovery_started", agent_id="a0", paper_id="openalex:W1", turn=0
    )
    tracer.emit(
        "neighbor_discovery_completed",
        agent_id="a0",
        paper_id="openalex:W1",
        turn=0,
        refs=2,
        cits=1,
    )
    tracer.emit(
        "frontier_reference_evaluation_started", agent_id="a0", count=2, turn=0
    )
    tracer.emit(
        "frontier_reference_evaluation_completed", agent_id="a0", count=2, turn=0
    )
    tracer.record_evaluation(
        DetailedEvaluation(
            agent_id="a0",
            oleada=1,
            turn=0,
            q=0.7,
            delta_q=0.7,
            self_assessment=SelfAssessmentDetail(score=0.7, reasoning="self"),
            peers=PeerVotesDetail(
                votes=[PeerVoteDetail(voter_id="a1", score=0.6, reasoning="peer")],
                aggregated_score=0.6,
                num_votes=1,
            ),
            virgin_judge=VirginJudgeDetail(score=0.7, coverage="c", gaps="g"),
            structural=StructuralComponentsDetail(r=0.5),
        )
    )
    tracer.emit("new_best", agent="a0", Q=0.7, oleada=1)
    tracer.emit("oleada_complete", oleada=1, best_Q=0.7, total_fetches=4, max_fetches=50)
    tracer.record_artifact("narrative_a0_t0.md", "narrative", "# Narrative\nBody")
    tracer.emit(
        "orchestrator_complete",
        run_id=run_id,
        winner="a0",
        peak_Q=0.7,
        status="completed",
        total_fetches=4,
        elapsed=3.0,
    )


def test_persisted_stream_reconstructs_live_projection(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "replay.db")
    run_id = store.create_run("seed-1", "how?")

    live_events: list[RunEvent] = []
    tracer = RunTracer(store, run_id, sink=CallbackSink(live_events.append))
    _drive(tracer, run_id)

    live = RunProjection.from_events(live_events)
    persisted = [
        RunEvent(seq=e["seq"], type=e["type"], payload=e["payload"], ts=e["ts"])
        for e in store.list_events(run_id)
    ]
    replayed = RunProjection.from_events(persisted)

    assert _stable_state(live.state) == _stable_state(replayed.state)
    assert replayed.state.status == "completed"
    assert replayed.state.winner_agent == "a0"
    assert replayed.state.evaluations["a0"][-1].self_assessment.reasoning == "self"
    assert "Body" in replayed.state.narratives["a0"]
    store.close()


def test_required_semantic_events_persist(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "replay.db")
    run_id = store.create_run("seed-1", "how?")
    tracer = RunTracer(store, run_id)
    _drive(tracer, run_id)
    types = [e["type"] for e in store.list_events(run_id)]
    for required in (
        "orchestrator_start",
        "oleada_start",
        "agent_turn_start",
        "paper_integration_completed",
        "evaluation_complete",
        "evaluation_detail",
        "new_best",
        "oleada_complete",
        "artifact_saved",
        "orchestrator_complete",
    ):
        assert required in types, required
    store.close()


def test_unrecognized_event_survives_replay(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "replay.db")
    run_id = store.create_run("seed-1", "how?")
    tracer = RunTracer(store, run_id)
    tracer.emit("orchestrator_start", run_id=run_id)
    tracer.emit("brand_new_event", future_field=1)
    persisted = [
        RunEvent(seq=e["seq"], type=e["type"], payload=e["payload"], ts=e["ts"])
        for e in store.list_events(run_id)
    ]
    projection = RunProjection.from_events(persisted)
    assert any(e.type == "brand_new_event" for e in projection.state.events)
    store.close()


def test_discovery_and_frontier_events_replay_to_timeline_nodes(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "replay.db")
    run_id = store.create_run("seed-1", "how?")
    tracer = RunTracer(store, run_id)
    _drive(tracer, run_id)
    persisted = [
        RunEvent(seq=e["seq"], type=e["type"], payload=e["payload"], ts=e["ts"])
        for e in store.list_events(run_id)
    ]
    projection = RunProjection.from_events(persisted)
    discovery = [e for e in projection.state.timeline if e.kind == "discovery"]
    frontier = [e for e in projection.state.timeline if e.kind == "frontier"]
    assert len(discovery) == 1 and discovery[0].status == "completed"
    assert "refs=2" in discovery[0].label
    assert len(frontier) == 1 and frontier[0].status == "completed"
    store.close()


class _FakeCompletions:
    async def create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        )


async def test_llm_client_producer_telemetry_persists_and_replays(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "replay.db")
    run_id = store.create_run("seed-1", "how?")
    live_events: list[RunEvent] = []
    tracer = RunTracer(store, run_id, sink=CallbackSink(live_events.append))

    client = LLMClient(api_key="sk-test")
    client.tracer = tracer
    client.client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=_FakeCompletions())
    )

    content = await client.chat(
        [{"role": "user", "content": "hi"}],
        model="explorer-x",
        purpose="paper_integration",
    )
    assert content == "ok"

    types = [e["type"] for e in store.list_events(run_id)]
    assert "llm_operation_started" in types
    assert "llm_operation_completed" in types

    persisted = [
        RunEvent(seq=e["seq"], type=e["type"], payload=e["payload"], ts=e["ts"])
        for e in store.list_events(run_id)
    ]
    live = RunProjection.from_events(live_events)
    replayed = RunProjection.from_events(persisted)
    assert _stable_state(live.state) == _stable_state(replayed.state)
    assert replayed.state.current_operation == "paper_integration"
    assert replayed.state.current_operation_model == "explorer-x"
    assert replayed.state.token_usage == 120
    assert replayed.state.operation_elapsed_seconds is not None
    store.close()
