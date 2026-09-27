"""Greedy policy determinism, tie-breaking, and reason codes."""

from __future__ import annotations

from research_explorer.research.models import (
    ActionKind,
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


async def test_tie_break_is_independent_of_input_order() -> None:
    forward = [
        CandidateAction(paper_id="a", score=0.5, source="s"),
        CandidateAction(paper_id="b", score=0.5, source="s"),
        CandidateAction(paper_id="c", score=0.5, source="s"),
    ]
    budget = BudgetState(max_fetches=5)
    slots = SlotState(total_slots=3)
    first = await GreedyPolicy(seed=3, max_actions=3).select_actions(
        _state(), forward, budget, slots
    )
    second = await GreedyPolicy(seed=3, max_actions=3).select_actions(
        _state(), list(reversed(forward)), budget, slots
    )
    assert [a.paper_id for a in first] == [a.paper_id for a in second]
    assert sorted(a.paper_id for a in first) == ["a", "b", "c"]


async def test_selection_records_predicted_value_and_cost() -> None:
    actions = await GreedyPolicy(seed=0).select_actions(
        _state(), _candidates(), BudgetState(max_fetches=5), SlotState()
    )
    assert actions[0].predicted_value == 0.9
    assert actions[0].predicted_cost == 1


async def test_exhausted_token_budget_blocks_selection() -> None:
    actions = await GreedyPolicy(seed=0).select_actions(
        _state(), _candidates(), BudgetState(max_tokens=0), SlotState()
    )
    assert actions == []


async def test_empty_frontier_schedules_search_for_missing_knowledge() -> None:
    state = _state()
    state.latest_evaluation = ResearchEvaluation(missing_knowledge=["gap"])
    policy = GreedyPolicy(seed=0, search_enabled=True, max_search_queries=2)
    actions = await policy.select_actions(
        state, [], BudgetState(max_fetches=5), SlotState()
    )
    assert len(actions) == 1
    assert actions[0].kind is ActionKind.SEARCH
    assert actions[0].query == "gap"


async def test_search_not_scheduled_without_evaluator_gaps() -> None:
    policy = GreedyPolicy(seed=0)
    actions = await policy.select_actions(
        _state(), [], BudgetState(max_fetches=5), SlotState()
    )
    assert actions == []


async def test_search_skips_equivalent_issued_query() -> None:
    state = _state()
    state.search_queries = ["gap"]
    state.latest_evaluation = ResearchEvaluation(missing_knowledge=["  Gap "])
    policy = GreedyPolicy(seed=0, search_enabled=True, max_search_queries=5)
    actions = await policy.select_actions(
        state, [], BudgetState(max_fetches=5), SlotState()
    )
    assert actions and actions[0].kind is ActionKind.SEARCH
    assert actions[0].query == "Q"
