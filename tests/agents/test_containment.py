"""STAB-6: malformed LLM output is contained; a failed turn does not abort the run."""

from __future__ import annotations

from types import SimpleNamespace

from research_explorer.aco.frontier import SharedFrontier
from research_explorer.aco.scheduler import Scheduler
from research_explorer.agents.explorer import ExplorerAgent
from research_explorer.agents.state import AgentState
from research_explorer.config import Config

SENTINEL = "SENTINEL-SECRET-XYZ"


def _agent() -> ExplorerAgent:
    return ExplorerAgent.__new__(ExplorerAgent)


def test_parse_eval_response_ignores_non_list_scores() -> None:
    agent = _agent()
    assert agent._parse_eval_response('{"scores": "oops"}') == []
    assert agent._parse_eval_response("[1, 2]") == []
    assert agent._parse_eval_response("{}") == []


def test_parse_eval_response_ignores_malformed_entries() -> None:
    agent = _agent()
    parsed = agent._parse_eval_response(
        '{"scores": [{"id": "a", "score": 0.5}, "bad", 7, {"id": "b", "score": 0.1}]}'
    )
    assert parsed == [{"id": "a", "score": 0.5}, {"id": "b", "score": 0.1}]


class _FailingAgent:
    def __init__(self) -> None:
        self.state = AgentState(id="agent-1", pos="openalex:seed", budget=5)
        self.calls = 0

    async def take_turn(self, k: int) -> list:
        self.calls += 1
        raise RuntimeError(f"malformed llm output token={SENTINEL}")


async def test_failed_agent_turn_is_contained_and_recorded() -> None:
    frontier = SharedFrontier()
    frontier.add(["openalex:cand"], source="openalex:seed", mode="ref")
    agent = _FailingAgent()
    assert frontier.claim_for("openalex:cand", agent.state.id) is True

    events: list[tuple[str, dict]] = []
    tracer = SimpleNamespace(emit=lambda kind, **payload: events.append((kind, payload)))
    colony = SimpleNamespace(
        llm=object(),
        seed_query="q",
        shared_frontier=frontier,
        agents=[agent],
        agent_states=[agent.state],
        active_candidates=lambda: [agent],
        update_best=lambda oleada: None,
        best_quality=0.0,
    )
    scheduler = Scheduler(
        colony=colony,
        config=Config(),
        pheromone=SimpleNamespace(update=lambda *a, **kw: None),
        structural=object(),
    )
    scheduler.tracer = tracer

    await scheduler.run_oleada()  # must not raise

    assert agent.calls == 1
    kinds = [kind for kind, _ in events]
    assert "agent_turn_failed" in kinds
    payload = dict(events[kinds.index("agent_turn_failed")][1])
    assert SENTINEL not in payload["error"]
    # Claims acquired before the exception are released.
    assert not frontier.is_claimed("openalex:cand")
