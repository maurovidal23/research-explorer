"""Peer vote component (P) of the quality function Q.

Other active agents vote on the value of an agent's recent contribution.
Aggregation: median (robust) or reputation-weighted.
"""

from __future__ import annotations

import asyncio
import statistics

from research_explorer.agents.explorer import ExplorerAgent
from research_explorer.agents.llm_client import LLMClient
from research_explorer.agents.prompts import peer_vote
from research_explorer.config import Config
from research_explorer.graph.models import Paper
from research_explorer.logging_setup import get_logger
from research_explorer.replay.models import PeerVoteDetail, PeerVotesDetail

log = get_logger("eval.peers")

VOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}


class PeerVoting:
    """Peer vote (P) — other active agents vote on an agent's contribution."""

    def __init__(self, llm: LLMClient, config: Config):
        self.llm = llm
        self.cfg = config

    async def vote(
        self,
        target: ExplorerAgent,
        active_agents: list[ExplorerAgent],
        seed_query: str,
        new_papers: list[Paper] | None = None,
    ) -> float:
        """Aggregate peer votes for the target agent.

        Args:
            target: The agent being evaluated.
            active_agents: All active agents (including target, who is excluded).
            seed_query: The research line query.
            new_papers: Papers the target added this turn (for focused voting).

        Returns:
            P ∈ [0, 1] — aggregated peer score.
        """
        voters = [a for a in active_agents if a.state.id != target.state.id]
        if not voters:
            return 0.0

        new_papers = new_papers or []
        tasks = [
            self._single_vote(v, target, seed_query, new_papers) for v in voters
        ]
        scores = await asyncio.gather(*tasks, return_exceptions=True)

        valid_scores: list[float] = []
        voter_weights: list[float] = []
        for v, s in zip(voters, scores, strict=False):
            if isinstance(s, BaseException):
                continue
            valid_scores.append(float(s))
            voter_weights.append(v.state.quality)

        if not valid_scores:
            return 0.0

        if self.cfg.quality.peer_aggregation == "reputation" and sum(voter_weights) > 0:
            return sum(s * w for s, w in zip(valid_scores, voter_weights, strict=False)) / sum(
                voter_weights
            )
        return statistics.median(valid_scores)

    async def vote_detail(
        self,
        target: ExplorerAgent,
        active_agents: list[ExplorerAgent],
        seed_query: str,
        new_papers: list[Paper] | None = None,
    ) -> PeerVotesDetail:
        """Aggregate peer votes, retaining each individual vote's reasoning."""
        voters = [a for a in active_agents if a.state.id != target.state.id]
        method = self.cfg.quality.peer_aggregation
        if not voters:
            return PeerVotesDetail(
                votes=[],
                aggregated_score=0.0,
                aggregation_method=method,
                num_votes=0,
            )

        new_papers = new_papers or []
        tasks = [
            self._single_vote_detail(v, target, seed_query, new_papers) for v in voters
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        votes = [r for r in results if isinstance(r, PeerVoteDetail)]
        if not votes:
            return PeerVotesDetail(
                votes=[],
                aggregated_score=0.0,
                aggregation_method=method,
                num_votes=0,
            )

        scores = [v.score for v in votes]
        if method == "reputation":
            weights = [
                v.state.quality
                for v, r in zip(voters, results, strict=False)
                if isinstance(r, PeerVoteDetail)
            ]
            if sum(weights) > 0:
                aggregated = sum(v.score * w for v, w in zip(votes, weights, strict=False)) / sum(weights)
            else:
                aggregated = statistics.median(scores)
        else:
            aggregated = statistics.median(scores)

        return PeerVotesDetail(
            votes=votes,
            aggregated_score=aggregated,
            aggregation_method=method,
            num_votes=len(votes),
        )

    async def _single_vote_detail(
        self,
        voter: ExplorerAgent,
        target: ExplorerAgent,
        seed_query: str,
        new_papers: list[Paper],
    ) -> PeerVoteDetail:
        messages = peer_vote(
            seed_query,
            voter.state.narrative,
            target.state.narrative,
            new_papers,
        )
        result = await self.llm.chat_json(
            messages,
            model=self.cfg.llm.explorer_model,
            schema=VOTE_SCHEMA,
            temperature=0.3,
            max_tokens=1000,
            purpose="peer_vote_detail",
        )
        score = float(result.get("score", 0.5))
        score = max(0.0, min(1.0, score))
        return PeerVoteDetail(
            voter_id=voter.state.id,
            score=score,
            reasoning=str(result.get("reasoning", "")),
        )

    async def _single_vote(
        self,
        voter: ExplorerAgent,
        target: ExplorerAgent,
        seed_query: str,
        new_papers: list[Paper],
    ) -> float:
        messages = peer_vote(
            seed_query,
            voter.state.narrative,
            target.state.narrative,
            new_papers,
        )
        result = await self.llm.chat_json(
            messages,
            model=self.cfg.llm.explorer_model,
            schema=VOTE_SCHEMA,
            temperature=0.3,
            max_tokens=1000,
            purpose="peer_vote",
        )
        score = float(result.get("score", 0.5))
        return max(0.0, min(1.0, score))
