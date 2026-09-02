"""Scheduler tracer integration: oleada run events + evaluation records."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from research_explorer.aco.scheduler import Scheduler
from research_explorer.agents.state import AgentState
from research_explorer.config import Config
from research_explorer.replay.models import (
    DetailedEvaluation,
    PeerVoteDetail,
    PeerVotesDetail,
    SelfAssessmentDetail,
    StructuralComponentsDetail,
    VirginJudgeDetail,
)
from research_explorer.replay.trace import RunTracer, RunTraceStore


class FakePheromone:
    def __init__(self):
        self.calls = 0

    def update(self, *args, **kwargs):
        self.calls += 1


class FakeColony:
    def __init__(self, agents: list[SimpleNamespace]):
        self.llm = object()
        self.agents = agents
        self.seed_query = "research line"
        self.graph = SimpleNamespace(get_paper=lambda nid: None)
        self.shared_frontier = []
        self.agent_states = [a.state for a in agents]
        self.best_quality = 0.0
        self.best_snapshot_agent = ""

    def update_best(self, oleada: int) -> None:
        peak = max(a.state.quality for a in self.agents)
        if peak > self.best_quality:
            self.best_quality = peak
            self.best_snapshot_agent = max(self.agents, key=lambda a: a.state.quality).state.id


def _agent(agent_id: str, budget: int = 4, narrative: str = "n") -> SimpleNamespace:
    state = AgentState(
        id=agent_id,
        pos="seed",
        narrative=narrative,
        budget=budget,
        turn_count=1,
        caste="mixto",
    )
    return SimpleNamespace(state=state, take_turn=_async_empty_turn)


async def _async_empty_turn(k: int) -> list:
    return []


async def test_scheduler_records_lifecycle_and_evaluations(tmp_path):
    store = RunTraceStore(tmp_path / "replay.db")
    run_id = store.create_run("seed", "research line")

    agents = [_agent("agent-000-aa"), _agent("agent-000-bb")]
    colony = FakeColony(agents)
    cfg = Config()
    scheduler = Scheduler(colony, cfg, FakePheromone(), SimpleNamespace())
    scheduler.tracer = RunTracer(store, run_id)
    scheduler._pick_top_k = lambda: agents

    records: list[DetailedEvaluation] = []

    async def fake_assess_detail(agent, active_agents, seed_query, new_papers=None, oleada=0):
        rec = DetailedEvaluation(
            agent_id=agent.state.id,
            oleada=oleada,
            turn=agent.state.turn_count,
            q=0.72,
            old_quality=agent.state.quality,
            delta_q=0.72 - agent.state.quality,
            self_assessment=SelfAssessmentDetail(score=0.8, reasoning=f"{agent.state.id} self r"),
            peers=PeerVotesDetail(
                votes=[PeerVoteDetail(voter_id="peer", score=0.7, reasoning="peer r")],
                aggregated_score=0.7,
                num_votes=1,
            ),
            virgin_judge=VirginJudgeDetail(score=0.6, coverage="covers", gaps="gaps"),
            structural=StructuralComponentsDetail(coverage=0.4, r=0.7),
        )
        records.append(rec)
        return rec

    scheduler.assessor.assess_detail = fake_assess_detail

    async def fake_take_turn(k):
        return [("seed", "paper-1", "ref")]

    for a in agents:
        a.take_turn = fake_take_turn

    await scheduler.run_oleada()

    events = store.list_events(run_id)
    types = [e["type"] for e in events]
    assert types[0] == "oleada_start"
    assert types[1] == "agent_turn_start"
    assert "evaluation_complete" in types
    assert "agent_turn_complete" in types
    assert "new_best" in types
    assert types[-1] == "oleada_complete"

    evals = store.list_evaluations(run_id)
    assert len(evals) == 2
    assert {e.agent_id for e in evals} == {"agent-000-aa", "agent-000-bb"}
    assert evals[0].self_assessment.reasoning.startswith("agent-000-")
    assert evals[0].virgin_judge.gaps == "gaps"
    assert records[0].peers.votes[0].reasoning == "peer r"

    eval_events = [e for e in events if e["type"] == "evaluation_complete"]
    assert {e["payload"]["agent_id"] for e in eval_events} == {"agent-000-aa", "agent-000-bb"}
    for e in eval_events:
        assert e["payload"]["Q"] == pytest.approx(0.72)

    assert scheduler.oleada_count == 1
    assert scheduler.history and scheduler.history[0]["oleada"] == 1

    narrative_artifacts = [
        a for a in store.list_artifacts(run_id) if a["kind"] == "narrative"
    ]
    assert len(narrative_artifacts) == 2
    names = {a["name"] for a in narrative_artifacts}
    assert "narrative_agent-000-aa_t1.md" in names
    assert "narrative_agent-000-bb_t1.md" in names

    store.close()


class FakeGraph:
    def __init__(self):
        self._papers = {}

    def cache_paper(self, paper):
        self._papers[paper.id] = paper

    def close(self):
        pass


class FakeOrchColony:
    def __init__(self):
        self.agents = []
        self.seed_query = "q"
        self.shared_frontier = []
        self.agent_states = []
        self.best_quality = 0.0
        self.best_snapshot_agent = ""
        self.best_snapshot_oleada = 0
        self.best_narrative = ""
        self._best_agent = None

    @property
    def best_agent(self):
        return self._best_agent

    def initialize(self, seed, query):
        pass

    def active_candidates(self):
        return []

    def update_best(self, oleada):
        pass

    def pheromone_concentration(self):
        return 0.0


async def test_orchestrator_emits_events_before_finish_run(tmp_path):
    db = tmp_path / "replay.db"
    trace = RunTraceStore(db)
    run_id = trace.create_run("seed", "q")
    tracer = RunTracer(trace, run_id)

    tracer.emit("orchestrator_start", seed="seed")
    tracer.emit("orchestrator_complete", winner="agent-000", best_Q=0.8)
    tracer.record_artifact("narrative_agent-000.md", "narrative", "winning text")
    trace.finish_run(run_id, "completed", best_quality=0.8)

    events = trace.list_events(run_id)
    types = [e["type"] for e in events]
    assert types[-1] != "finish_run"
    complete_idx = types.index("orchestrator_complete")
    artifact_idx = types.index("artifact_saved")
    assert artifact_idx > complete_idx

    run = trace.get_run(run_id)
    assert run["status"] == "completed"

    narrative = trace.get_latest_artifact(run_id, "narrative")
    assert narrative is not None
    assert narrative["content"] == "winning text"
    trace.close()


async def test_orchestrator_no_winner_emits_event(tmp_path):
    db = tmp_path / "replay.db"
    trace = RunTraceStore(db)
    run_id = trace.create_run("seed", "q")
    tracer = RunTracer(trace, run_id)

    tracer.emit("orchestrator_start", seed="seed")
    tracer.emit("no_winner", elapsed=1.0)
    trace.finish_run(run_id, "completed", best_quality=0.0)

    events = trace.list_events(run_id)
    types = [e["type"] for e in events]
    assert "no_winner" in types
    assert "orchestrator_complete" not in types

    run = trace.get_run(run_id)
    assert run["status"] == "completed"

    narrative = trace.get_latest_artifact(run_id, "narrative")
    assert narrative is None
    trace.close()


async def test_scheduler_skips_evaluation_when_no_edges(tmp_path):
    store = RunTraceStore(tmp_path / "replay.db")
    run_id = store.create_run("seed", "research line")

    agent = _agent("agent-no-evidence", budget=4)
    agent.state.delta_q = 0.7
    other = _agent("agent-with-evidence", budget=4)
    colony = FakeColony([agent, other])
    cfg = Config()
    scheduler = Scheduler(colony, cfg, FakePheromone(), SimpleNamespace())
    scheduler.tracer = RunTracer(store, run_id)
    scheduler._pick_top_k = lambda: [agent]

    assessed: list[str] = []

    async def fake_assess_detail(a, active_agents, seed_query, new_papers=None, oleada=0):
        assessed.append(a.state.id)
        return DetailedEvaluation(
            agent_id=a.state.id,
            oleada=oleada,
            turn=a.state.turn_count,
            q=0.8,
            old_quality=a.state.quality,
            delta_q=0.8,
            self_assessment=SelfAssessmentDetail(score=0.8, reasoning="ok"),
            peers=PeerVotesDetail(aggregated_score=0.7, num_votes=1),
            virgin_judge=VirginJudgeDetail(score=0.6, coverage="", gaps=""),
            structural=StructuralComponentsDetail(r=0.5),
        )

    scheduler.assessor.assess_detail = fake_assess_detail

    await scheduler.run_oleada()

    assert assessed == [], "assess_detail must not be called when edges are empty"

    assert agent.state.quality == 0.0, "agent quality must remain unchanged"
    assert agent.state.delta_q == 0.0

    events = store.list_events(run_id)
    types = [e["type"] for e in events]
    assert "evaluation_skipped" in types
    assert "agent_turn_skipped" in types

    skipped_evts = [e for e in events if e["type"] == "evaluation_skipped"]
    assert len(skipped_evts) == 1
    assert skipped_evts[0]["payload"]["agent_id"] == "agent-no-evidence"
    assert skipped_evts[0]["payload"]["reason"] == "no_new_evidence"

    evals = store.list_evaluations(run_id)
    assert len(evals) == 0

    store.close()
