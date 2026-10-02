"""Replaceable exploration policy interface and the deterministic greedy baseline.

The policy depends only on the research domain models: it never fetches
providers and never writes evidence. That keeps controller, policy, and
persistence independently replaceable (e.g. by a future contextual-UCB or ACO
policy) without touching the provider layer.
"""

from __future__ import annotations

import random
from typing import Protocol

from research_explorer.research.models import (
    ActionKind,
    BudgetState,
    CandidateAction,
    ResearchAction,
    ResearchEvaluation,
    ResearchState,
    SlotState,
    normalize_query,
)


class ExplorationPolicy(Protocol):
    """Select one or more actions and observe their evaluations."""

    async def select_actions(
        self,
        state: ResearchState,
        candidates: list[CandidateAction],
        budget: BudgetState,
        slots: SlotState,
    ) -> list[ResearchAction]: ...

    async def observe(
        self,
        actions: list[ResearchAction],
        evaluations: list[ResearchEvaluation],
    ) -> None: ...


class GreedyPolicy:
    """Deterministic greedy selection with explicit, seeded tie-breaking.

    Candidates are ranked by descending score. Exact ties are broken by a
    per-seed stable shuffle of paper ids, so the same seed always yields the
    same order. All considered candidates and their reason codes are retained
    on the instance for the controller to record.
    """

    def __init__(
        self,
        seed: int = 0,
        max_actions: int = 1,
        search_enabled: bool = True,
        max_search_queries: int = 3,
    ) -> None:
        self.seed = seed
        self.max_actions = max_actions
        self.search_enabled = search_enabled
        self.max_search_queries = max_search_queries
        self.last_considered: list[CandidateAction] = []
        self.last_reasons: dict[str, str] = {}
        self.observations: list[ResearchEvaluation] = []

    def _tie_rank(self, candidates: list[CandidateAction]) -> dict[str, int]:
        ids = sorted({c.paper_id for c in candidates})
        rng = random.Random(self.seed)
        rng.shuffle(ids)
        return {pid: rank for rank, pid in enumerate(ids)}

    def _pending_queries(self, state: ResearchState) -> list[str]:
        """Return unused search queries derived from the question and evaluator."""
        evaluation = state.latest_evaluation
        issues: list[str] = []
        if evaluation is not None:
            issues.extend(evaluation.missing_knowledge)
            issues.extend(evaluation.recommended_questions)
        if not any(issue.strip() for issue in issues):
            return []
        issued = set(state.search_queries) | set(state.blocked_search_queries)
        pending: list[str] = []
        seen: set[str] = set()
        for raw in [*issues, state.objective.question]:
            normalized = normalize_query(raw)
            if not normalized or normalized in issued or normalized in seen:
                continue
            seen.add(normalized)
            pending.append(raw.strip())
        return pending

    async def select_actions(
        self,
        state: ResearchState,
        candidates: list[CandidateAction],
        budget: BudgetState,
        slots: SlotState,
    ) -> list[ResearchAction]:
        self.last_considered = list(candidates)
        reasons: dict[str, str] = {}
        eligible: list[CandidateAction] = []
        for candidate in candidates:
            if candidate.paper_id in state.visited:
                reasons[candidate.paper_id] = "already_visited"
            elif candidate.score <= 0.0:
                reasons[candidate.paper_id] = "non_positive_score"
            else:
                reasons[candidate.paper_id] = "eligible"
                eligible.append(candidate)
        self.last_reasons = reasons

        rank = self._tie_rank(eligible)
        ordered = sorted(
            eligible,
            key=lambda c: (-round(c.score, 9), rank.get(c.paper_id, 0), c.paper_id),
        )

        capacity = min(
            self.max_actions,
            max(slots.available, 0) or self.max_actions,
            budget.fetches_remaining,
        )
        if budget.tokens_remaining <= 0:
            capacity = 0
        selected = ordered[:capacity]
        if selected:
            return [
                ResearchAction(
                    kind=ActionKind.READ_EVIDENCE,
                    paper_id=candidate.paper_id,
                    reason=candidate.reason or "greedy_top_score",
                    predicted_value=candidate.score,
                    predicted_cost=1,
                )
                for candidate in selected
            ]

        if (
            self.search_enabled
            and not eligible
            and budget.fetches_remaining > 0
            and budget.tokens_remaining > 0
            and len(state.search_queries) < self.max_search_queries
        ):
            pending = self._pending_queries(state)
            if pending:
                query = pending[0]
                self.last_reasons = {
                    normalize_query(query): "frontier_empty_search"
                }
                return [
                    ResearchAction(
                        kind=ActionKind.SEARCH,
                        query=query,
                        reason="frontier_empty_search",
                        predicted_value=0.4,
                        predicted_cost=1,
                    )
                ]
        return []

    async def observe(
        self,
        actions: list[ResearchAction],
        evaluations: list[ResearchEvaluation],
    ) -> None:
        self.observations.extend(evaluations)
