"""Wave lifecycle barrier, evaluation-invariant, and containment tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from research_explorer.aco.scheduler import Scheduler
from research_explorer.agents.state import AgentState
from research_explorer.config import Config
from research_explorer.events.models import (
    PHASE_DECISION,
    PHASE_EVALUATION,
    PHASE_RESEARCH,
)
from research_explorer.events.projection import RunProjection
from research_explorer.replay.models import (
    DetailedEvaluation,
    PeerVoteDetail,
    PeerVotesDetail,
    SelfAssessmentDetail,
    StructuralComponentsDetail,
    VirginJudgeDetail,
)
from research_explorer.replay.trace import RunTracer, RunTraceStore
from research_explorer.tui import text as render


class FakePheromone:
    def __init__(self) -> None:
        self.calls = 0

    def update(self, *args, **kwargs) -> None:
        self.calls += 1


class _Frontier:
    def release_all(self, agent: str) -> None:
        return None

    def __len__(self) -> int:
        return 0


class FakeColony:
    def __init__(self, agents: list[SimpleNamespace]) -> None:
        self.llm = object()
        self.agents = agents
        self.seed_query = "research line"
        self.graph = SimpleNamespace(get_paper=lambda nid: None)
        self.shared_frontier = _Frontier()
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
    return SimpleNamespace(state=state)


def _record(agent, oleada: int, q: float, evidence: int = 1) -> DetailedEvaluation:
    return DetailedEvaluation(
        agent_id=agent.state.id,
        oleada=oleada,
        turn=agent.state.turn_count,
        q=q,
        old_quality=agent.state.quality,
        delta_q=q - agent.state.quality,
        self_assessment=SelfAssessmentDetail(score=q, reasoning="self"),
        peers=PeerVotesDetail(
            votes=[PeerVoteDetail(voter_id="peer", score=q, reasoning="peer")],
            aggregated_score=q,
            num_votes=1,
        ),
        virgin_judge=VirginJudgeDetail(score=q, coverage="", gaps=""),
        structural=StructuralComponentsDetail(r=q),
        new_papers=[f"paper-{evidence}"],
        status="complete",
    )


def _scheduler(tmp_path, agents, tracer, assess):
    store = tracer.store
    run_id = tracer.run_id
    colony = FakeColony(agents)
    scheduler = Scheduler(colony, Config(), FakePheromone(), SimpleNamespace())
    scheduler.tracer = tracer
    scheduler._pick_top_k = lambda: agents
    scheduler.assessor.assess_detail = assess
    return scheduler, store, run_id


async def test_two_agents_finish_research_before_evaluation_starts(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "replay.db")
    run_id = store.create_run("seed", "research line")
    tracer = RunTracer(store, run_id)
    agents = [_agent("a0"), _agent("a1")]
    for agent in agents:
        async def take_turn(k, agent=agent):
            return [("seed", f"paper-{agent.state.id}", "ref")]

        agent.take_turn = take_turn

    order: list[str] = []

    async def assess(agent, active_agents, seed_query, new_papers=None, oleada=0):
        order.append(f"eval:{agent.state.id}")
        return _record(agent, oleada, 0.5)

    scheduler, store, run_id = _scheduler(tmp_path, agents, tracer, assess)
    await scheduler.run_oleada()

    events = store.list_events(run_id)
    types = [e["type"] for e in events]
    first_eval = types.index("quality_evaluation_started")
    last_turn = max(
        i for i, t in enumerate(types) if t in ("agent_turn_complete", "agent_turn_skipped")
    )
    assert last_turn < first_eval
    assert order == ["eval:a0", "eval:a1"]
    assert "wave_phase_completed" in types
    research_done = next(
        i
        for i, e in enumerate(events)
        if e["type"] == "wave_phase_completed" and e["payload"]["phase"] == "research"
    )
    assert research_done < first_eval
    store.close()


async def test_evaluation_order_cannot_change_inputs_or_rankings(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "replay.db")
    run_id = store.create_run("seed", "research line")
    tracer = RunTracer(store, run_id)
    agents = [_agent("a0"), _agent("a1")]
    for agent in agents:
        async def take_turn(k, agent=agent):
            return [("seed", f"paper-{agent.state.id}", "ref")]

        agent.take_turn = take_turn

    seen_qualities: list[tuple[str, float, float]] = []

    async def assess(agent, active_agents, seed_query, new_papers=None, oleada=0):
        seen_qualities.append(
            (
                agent.state.id,
                agent.state.quality,
                max(a.state.quality for a in active_agents),
            )
        )
        return _record(agent, oleada, 0.8 if agent.state.id == "a0" else 0.4)

    scheduler, store, run_id = _scheduler(tmp_path, agents, tracer, assess)
    await scheduler.run_oleada()

    assert seen_qualities[0][1] == 0.0
    assert seen_qualities[1][1] == 0.0, "peer weights must stay frozen at the research value"
    assert seen_qualities[1][2] == 0.0
    ranking = scheduler.last_decision["ranking"]
    assert [aid for aid, _ in ranking[:2]] == ["a0", "a1"]
    store.close()


async def test_every_selected_agent_gets_one_terminal_evaluation(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "replay.db")
    run_id = store.create_run("seed", "research line")
    tracer = RunTracer(store, run_id)
    agents = [_agent("a0"), _agent("a1"), _agent("a2")]
    for agent in agents:
        async def take_turn(k, agent=agent):
            if agent.state.id == "a2":
                return []
            return [("seed", f"paper-{agent.state.id}", "ref")]

        agent.take_turn = take_turn

    async def assess(agent, active_agents, seed_query, new_papers=None, oleada=0):
        return _record(agent, oleada, 0.5)

    scheduler, store, run_id = _scheduler(tmp_path, agents, tracer, assess)
    await scheduler.run_oleada()

    settled = [e for e in store.list_events(run_id) if e["type"] == "evaluation_settled"]
    assert len(settled) == 3
    by_agent = {e["payload"]["agent_id"]: e["payload"]["status"] for e in settled}
    assert by_agent == {"a0": "complete", "a1": "complete", "a2": "skipped"}
    skipped = next(e for e in settled if e["payload"]["agent_id"] == "a2")
    assert skipped["payload"]["reason"] == "no_new_evidence"
    store.close()


async def test_agent_failure_is_contained_and_only_valid_results_ranked(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "replay.db")
    run_id = store.create_run("seed", "research line")
    tracer = RunTracer(store, run_id)
    agents = [_agent("a0"), _agent("a1")]

    async def failing_turn(k):
        raise RuntimeError("provider exploded")

    async def working_turn(k):
        return [("seed", "paper-ok", "ref")]

    agents[0].take_turn = failing_turn
    agents[1].take_turn = working_turn

    async def assess(agent, active_agents, seed_query, new_papers=None, oleada=0):
        return _record(agent, oleada, 0.9)

    scheduler, store, run_id = _scheduler(tmp_path, agents, tracer, assess)
    await scheduler.run_oleada()

    events = store.list_events(run_id)
    failed = [e for e in events if e["type"] == "agent_turn_failed"]
    assert len(failed) == 1 and failed[0]["payload"]["agent_id"] == "a0"
    settled = {e["payload"]["agent_id"]: e["payload"]["status"] for e in events if e["type"] == "evaluation_settled"}
    assert settled["a0"] == "failed"
    assert settled["a1"] == "complete"
    assert agents[0].state.quality == 0.0
    assert agents[1].state.quality == 0.9
    assert scheduler.last_decision["leader"] == "a1"
    store.close()


async def test_phase_failure_is_durable_and_reraised(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "replay.db")
    run_id = store.create_run("seed", "research line")
    tracer = RunTracer(store, run_id)
    agents = [_agent("a0")]

    async def assess(agent, active_agents, seed_query, new_papers=None, oleada=0):
        return _record(agent, oleada, 0.5)

    scheduler, store, run_id = _scheduler(tmp_path, agents, tracer, assess)

    async def boom(wave, k_agents):
        raise RuntimeError("research exploded")

    scheduler._run_research = boom
    with pytest.raises(RuntimeError):
        await scheduler.run_oleada()

    failed = [
        e for e in store.list_events(run_id) if e["type"] == "wave_phase_failed"
    ]
    assert len(failed) == 1
    assert failed[0]["payload"]["phase"] == "research"
    assert "research exploded" in failed[0]["payload"]["error"]
    store.close()


async def test_decision_mutations_happen_after_evaluation_barrier(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "replay.db")
    run_id = store.create_run("seed", "research line")
    tracer = RunTracer(store, run_id)
    agents = [_agent("a0"), _agent("a1")]
    for agent in agents:
        async def take_turn(k, agent=agent):
            return [("seed", f"paper-{agent.state.id}", "ref")]

        agent.take_turn = take_turn

    async def assess(agent, active_agents, seed_query, new_papers=None, oleada=0):
        return _record(agent, oleada, 0.6)

    scheduler, store, run_id = _scheduler(tmp_path, agents, tracer, assess)
    await scheduler.run_oleada()

    events = store.list_events(run_id)
    types = [e["type"] for e in events]
    eval_settled_idx = max(i for i, t in enumerate(types) if t == "evaluation_settled")
    decision_started = next(
        i
        for i, e in enumerate(events)
        if e["type"] == "wave_phase_started" and e["payload"]["phase"] == "decision"
    )
    assert eval_settled_idx < decision_started
    new_best = types.index("new_best")
    assert decision_started < new_best
    assert agents[0].state.quality == 0.6 and agents[1].state.quality == 0.6
    decision = next(
        e
        for e in events
        if e["type"] == "wave_phase_completed" and e["payload"]["phase"] == "decision"
    )
    assert "ranking" in decision["payload"] and "budget_used" in decision["payload"]
    assert "converged" in decision["payload"]
    assert "pheromone_concentration" in decision["payload"]
    store.close()


async def test_scheduler_durable_events_reconstruct_wave_view(tmp_path) -> None:
    """The scheduler's durable telemetry must replay into the wave-first view.

    This pins the wire contract between the phase/evaluation events the scheduler
    emits and the projection the TUI/replay consume. The lifecycle tests assert
    the emitted payloads and the projection tests assert hand-crafted events, so
    without this test a field rename on either side would stay green.
    """
    store = RunTraceStore(tmp_path / "replay.db")
    run_id = store.create_run("seed", "research line")
    tracer = RunTracer(store, run_id)
    agents = [_agent("a0"), _agent("a1")]
    for agent in agents:
        async def take_turn(k, agent=agent):
            return [("seed", f"paper-{agent.state.id}", "ref")]

        agent.take_turn = take_turn

    async def assess(agent, active_agents, seed_query, new_papers=None, oleada=0):
        record = _record(agent, oleada, 0.4)
        record.self_assessment.available = False
        record.self_assessment.unavailable_reason = "empty_narrative"
        record.peers.available = False
        record.peers.unavailable_reason = "no_peers"
        record.virgin_judge.available = False
        record.virgin_judge.unavailable_reason = "empty_narrative"
        record.unavailable = {
            "S": "empty_narrative",
            "P": "no_peers",
            "J": "empty_narrative",
        }
        return record

    scheduler, store, run_id = _scheduler(tmp_path, agents, tracer, assess)
    await scheduler.run_oleada()

    state = RunProjection.from_events(store.list_events(run_id)).state

    research = state.phase(1, PHASE_RESEARCH)
    assert research is not None
    assert research.completed == ["a0", "a1"]
    assert research.papers_attempted == 2
    assert research.evidence_added == 2

    evaluation = state.phase(1, PHASE_EVALUATION)
    assert evaluation is not None and evaluation.best_q == 0.4
    decision = state.phase(1, PHASE_DECISION)
    assert decision is not None and decision.leader == "a0"
    assert decision.continue_reason == "budget_remaining"

    record = state.evaluation_state(1, "a0")
    assert record is not None and record.status == "complete"
    assert record.components["R"] == 0.4
    assert record.components["S"] is None
    assert record.unavailable["S"] == "empty_narrative"
    assert record.q == 0.4

    # The TUI must distinguish unavailable components from a numeric zero and
    # show the evaluation status per agent in the phase table.
    detail = "\n".join(render.render_evaluation_state(state, "a0", 1))
    assert "S: unavailable (empty_narrative)" in detail
    assert "R: 0.4000" in detail
    table = "\n".join(render.render_phase_agent_table(state, 1))
    assert "A01" in table and "complete" in table and "0.400" in table
    summary = "\n".join(render.render_wave_summary(state, 1))
    assert "best Q 0.400" in summary
    assert "budget_remaining" in summary
    store.close()
