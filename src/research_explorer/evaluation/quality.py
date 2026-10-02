"""Quality aggregator — combines S, P, J, R into the final Q score.

Q = w1*S + w2*P + w3*J + w4*R, with Σwi = 1.
"""

from __future__ import annotations

import asyncio

from research_explorer.agents.explorer import ExplorerAgent
from research_explorer.agents.llm_client import LLMClient
from research_explorer.config import Config
from research_explorer.evaluation.peer_vote import PeerVoting
from research_explorer.evaluation.self_assess import SelfAssessment
from research_explorer.evaluation.structural import StructuralMetrics
from research_explorer.evaluation.virgin_judge import VirginJudge
from research_explorer.graph.models import Paper, normalize_id
from research_explorer.logging_setup import get_logger
from research_explorer.replay.models import DetailedEvaluation

log = get_logger("eval.quality")


class QualityAssessor:
    """Aggregates the four quality components into a single Q score."""

    def __init__(self, llm: LLMClient, config: Config, structural: StructuralMetrics):
        self.llm = llm
        self.cfg = config
        self.structural = structural
        self.self_assess = SelfAssessment(llm, config)
        self.peer_vote = PeerVoting(llm, config)
        self.virgin_judge = VirginJudge(llm, config)

    async def assess(
        self,
        agent: ExplorerAgent,
        active_agents: list[ExplorerAgent],
        seed_query: str,
        new_papers: list[Paper] | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Compute Q for an agent, combining S, P, J, R.

        Args:
            agent: The agent to evaluate.
            active_agents: All active agents (for peer voting).
            seed_query: The research line query.
            new_papers: Papers added this turn (for focused peer voting).

        Returns:
            (Q, breakdown) where Q is in [0, 1] and breakdown is
            {"S": s, "P": p, "J": j, "R": r}.
        """
        detail = await self.assess_detail(agent, active_agents, seed_query, new_papers)
        return detail.q, detail.breakdown

    async def assess_detail(
        self,
        agent: ExplorerAgent,
        active_agents: list[ExplorerAgent],
        seed_query: str,
        new_papers: list[Paper] | None = None,
        oleada: int = 0,
    ) -> DetailedEvaluation:
        """Compute Q and return the full DetailedEvaluation record.

        Retains self reasoning, every individual peer vote with reasoning, the
        virgin judge's coverage/gaps, and the structural component breakdown.
        """
        q = self.cfg.quality
        old_quality = agent.state.quality

        # Run S, P, J concurrently; R is synchronous
        s_task = self.self_assess.score_detail(agent.state.narrative, seed_query)
        p_task = self.peer_vote.vote_detail(agent, active_agents, seed_query, new_papers)
        j_task = self.virgin_judge.judge_detail(agent.state.narrative, seed_query)
        structural = self.structural.compute_detail(agent.state)

        s, p, j = await asyncio.gather(s_task, p_task, j_task)
        weights = {
            "S": q.w_self,
            "P": q.w_peers,
            "J": q.w_virgin,
            "R": q.w_structural,
        }
        component_scores: dict[str, float | None] = {
            "S": s.score if s.available else None,
            "P": p.aggregated_score if p.available else None,
            "J": j.score if j.available else None,
            "R": structural.r if structural.available else None,
        }
        unavailable = {
            "S": s.unavailable_reason,
            "P": p.unavailable_reason,
            "J": j.unavailable_reason,
            "R": structural.unavailable_reason,
        }
        unavailable = {k: v for k, v in unavailable.items() if component_scores[k] is None}
        score = sum(
            weights[k] * (value if value is not None else 0.0)
            for k, value in component_scores.items()
        )
        reason = ""
        if unavailable:
            reason = "unavailable:" + ",".join(sorted(unavailable))
        log.debug(
            "quality_computed",
            agent=agent.state.id,
            S=s.score,
            P=p.aggregated_score,
            J=j.score,
            R=structural.r,
            Q=score,
            unavailable=unavailable,
        )
        clamped = max(0.0, min(1.0, score))
        return DetailedEvaluation(
            agent_id=agent.state.id,
            oleada=oleada,
            turn=agent.state.turn_count,
            q=clamped,
            old_quality=old_quality,
            delta_q=clamped - old_quality,
            weights=weights,
            self_assessment=s,
            peers=p,
            virgin_judge=j,
            structural=structural,
            new_papers=[normalize_id(p.provider, p.id) for p in (new_papers or [])],
            status="complete",
            reason=reason,
            unavailable=unavailable,
        )
