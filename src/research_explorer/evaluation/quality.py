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
from research_explorer.graph.models import Paper
from research_explorer.logging_setup import get_logger

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
        q = self.cfg.quality

        # Run S, P, J concurrently; R is synchronous
        s_task = self.self_assess.score(agent.state.narrative, seed_query)
        p_task = self.peer_vote.vote(agent, active_agents, seed_query, new_papers)
        j_task = self.virgin_judge.judge(agent.state.narrative, seed_query)
        r = self.structural.compute(agent.state)

        s, p, j = await asyncio.gather(s_task, p_task, j_task)

        score = q.w_self * s + q.w_peers * p + q.w_virgin * j + q.w_structural * r
        log.debug(
            "quality_computed",
            agent=agent.state.id,
            S=s,
            P=p,
            J=j,
            R=r,
            Q=score,
        )
        clamped = max(0.0, min(1.0, score))
        return clamped, {"S": s, "P": p, "J": j, "R": r}
