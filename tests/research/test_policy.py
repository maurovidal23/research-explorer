"""Greedy policy determinism, tie-breaking, and reason codes."""

from __future__ import annotations

from research_explorer.research.models import (
    AgentNotebook,
    BudgetState,
    CandidateAction,
    ResearchEvaluation,
    ResearchObjective,
    ResearchState,
    SlotState,
)
from research_explorer.research.policy import GreedyPolicy


def _state(visited: list[str] | None = None) -> ResearchState:
    objective = ResearchObjective(
        run_id="r", seed_paper_id="seed", question="Q", budget=BudgetState(max_fetches=5)
    )
    return ResearchState(
        objective=objective,
        notebook=AgentNotebook(agent_id="a"),
        visited=visited or [],
    )


def _candidates() -> list[CandidateAction]:
    return [
        CandidateAction(paper_id="b", score=0.5, source="s", mode="ref"),
        CandidateAction(paper_id="a", score=0.5, source="s", mode="ref"),
        CandidateAction(paper_id="c", score=0.9, source="s", mode="cites"),
    ]


async def test_greedy_picks_highest_score() -> None:
    policy = GreedyPolicy(seed=1, max_actions=1)
    actions = await policy.select_actions(
        _state(), _candidates(), BudgetState(max_fetches=5), SlotState()
    )
    assert [a.paper_id for a in actions] == ["c"]


async def test_tie_break_is_seed_deterministic() -> None:
    policy_a = GreedyPolicy(seed=7, max_actions=2)
    policy_b = GreedyPolicy(seed=7, max_actions=2)
    state = _state()
    budget = BudgetState(max_fetches=5)
    first = await policy_a.select_actions(state, _candidates(), budget, SlotState())
    second = await policy_b.select_actions(state, _candidates(), budget, SlotState())
    assert [a.paper_id for a in first] == [a.paper_id for a in second]
    # c is top; the remaining tie (a, b) is resolved consistently.
    assert next(a.paper_id for a in first) == "c"


async def test_reason_codes_recorded_for_all_candidates() -> None:
    policy = GreedyPolicy(seed=0)
    await policy.select_actions(
        _state(visited=["c"]),
        [*_candidates(), CandidateAction(paper_id="d", score=0.0)],
        BudgetState(max_fetches=5),
        SlotState(),
    )
    assert policy.last_reasons["c"] == "already_visited"
    assert policy.last_reasons["d"] == "non_positive_score"
    assert policy.last_reasons["a"] == "eligible"
    assert len(policy.last_considered) == 4


async def test_budget_and_slots_limit_capacity() -> None:
    policy = GreedyPolicy(seed=0, max_actions=3)
    actions = await policy.select_actions(
        _state(), _candidates(), BudgetState(max_fetches=1), SlotState(total_slots=1)
    )
    assert len(actions) == 1


async def test_observe_records_evaluations() -> None:
    policy = GreedyPolicy()
    evaluation = ResearchEvaluation(overall=0.4)
    await policy.observe([], [evaluation])
    assert policy.observations == [evaluation]
