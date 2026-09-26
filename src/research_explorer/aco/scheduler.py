"""Scheduler -- manages K concurrent agent slots with priority-based selection.

Implements multitarea cooperativa: a large colony (N) is served through a
narrow window of K active slots. Agents are selected by priority:
    priority(a) = Q_a + gamma * EV(F_a) + delta * b_a

After each oleada (all K agents complete a turn), agents are persisted and
the dormant queue is re-prioritized.
"""

from __future__ import annotations

import time

from research_explorer.aco.colony import Colony
from research_explorer.agents.explorer import ExplorerAgent
from research_explorer.config import Config
from research_explorer.evaluation.quality import QualityAssessor
from research_explorer.evaluation.structural import StructuralMetrics
from research_explorer.graph.feromone import AgentPath, PheromoneManager
from research_explorer.logging_setup import get_logger
from research_explorer.redaction import redact_secrets
from research_explorer.replay.trace import RunTracer

log = get_logger("scheduler")


class Scheduler:
    """Manages K concurrent agent slots with priority-based activation."""

    def __init__(
        self,
        colony: Colony,
        config: Config,
        pheromone: PheromoneManager,
        structural: StructuralMetrics,
    ):
        self.colony = colony
        self.cfg = config
        self.pheromone = pheromone
        self.assessor = QualityAssessor(colony.llm, config, structural)
        self.oleada_count = 0
        self._total_fetches = 0
        self.history: list[dict] = []
        self.tracer: RunTracer | None = None

    async def run_oleada(self) -> None:
        """Run one oleada: activate K agents, each takes a turn of k fetches."""
        self.oleada_count += 1
        k_agents = self._pick_top_k()
        if not k_agents:
            log.info("no_active_agents")
            return

        oleada_start_time = time.monotonic()

        log.info(
            "oleada_start",
            oleada=self.oleada_count,
            active=[a.state.id for a in k_agents],
        )
        if self.tracer is not None:
            self.tracer.emit(
                "oleada_start",
                oleada=self.oleada_count,
                active=[a.state.id for a in k_agents],
            )

        # Each agent takes its turn (sequentially within the oleada for peer voting)
        agent_paths: list[AgentPath] = []
        for agent in k_agents:
            log.info(
                "agent_turn_start",
                agent=agent.state.id,
                caste=agent.state.caste,
                turn=agent.state.turn_count,
            )
            if self.tracer is not None:
                self.tracer.emit(
                    "agent_turn_start",
                    agent=agent.state.id,
                    caste=agent.state.caste,
                    oleada=self.oleada_count,
                    turn=agent.state.turn_count,
                )

            edges = []
            try:
                edges = await agent.take_turn(self.cfg.aco.k_per_turn)
            except Exception as e:
                # A failed agent turn is contained and recorded; unrelated
                # turns and the run continue (STAB-6).
                error = redact_secrets(str(e))
                log.warning("agent_turn_failed", agent=agent.state.id, error=error)
                self.colony.shared_frontier.release_all(agent.state.id)
                agent.state.delta_q = 0.0
                if self.tracer is not None:
                    self.tracer.emit(
                        "agent_turn_failed",
                        agent_id=agent.state.id,
                        oleada=self.oleada_count,
                        turn=agent.state.turn_count,
                        error=error,
                    )
                agent_paths.append(
                    AgentPath(edges=[], delta_q=0.0, state=agent.state)
                )
                continue

            # Fetch the papers added this turn for peer voting
            new_papers = self._get_new_papers(agent, edges)

            if not edges:
                agent.state.delta_q = 0.0
                log.info(
                    "evaluation_skipped",
                    agent=agent.state.id,
                    oleada=self.oleada_count,
                    turn=agent.state.turn_count,
                    reason="no_new_evidence",
                )
                if self.tracer is not None:
                    self.tracer.emit(
                        "evaluation_skipped",
                        agent_id=agent.state.id,
                        oleada=self.oleada_count,
                        turn=agent.state.turn_count,
                        reason="no_new_evidence",
                    )
            else:
                # Assess quality -- returns a detailed record retaining rationales
                record = await self.assessor.assess_detail(
                    agent, k_agents, self.colony.seed_query, new_papers,
                    oleada=self.oleada_count,
                )
                q_score = record.q
                breakdown = record.breakdown
                agent.state.quality = q_score
                agent.state.delta_q = record.delta_q
                if self.tracer is not None:
                    self.tracer.record_evaluation(record)
                    self.tracer.record_artifact(
                        f"narrative_{agent.state.id}_t{agent.state.turn_count}.md",
                        "narrative",
                        agent.state.narrative,
                    )
            agent_paths.append(
                AgentPath(edges=edges, delta_q=agent.state.delta_q, state=agent.state)
            )
            self._total_fetches += max(0, agent.state.delta_work)

            if edges:
                log.info(
                    "agent_turn_complete",
                    agent=agent.state.id,
                    Q=agent.state.quality,
                    S=breakdown["S"],
                    P=breakdown["P"],
                    J=breakdown["J"],
                    R=breakdown["R"],
                    delta_q=agent.state.delta_q,
                    fetches=len(edges),
                    budget=agent.state.budget,
                    frontier=len(self.colony.shared_frontier),
                )
                if self.tracer is not None:
                    self.tracer.emit(
                        "agent_turn_complete",
                        agent=agent.state.id,
                        oleada=self.oleada_count,
                        Q=round(record.q, 4),
                        S=round(breakdown["S"], 4),
                        P=round(breakdown["P"], 4),
                        J=round(breakdown["J"], 4),
                        R=round(breakdown["R"], 4),
                        delta_q=round(record.delta_q, 4),
                        fetches=len(edges),
                        budget=agent.state.budget,
                        frontier=len(self.colony.shared_frontier),
                    )
            else:
                log.info(
                    "agent_turn_skipped",
                    agent=agent.state.id,
                    fetches=0,
                    budget=agent.state.budget,
                    frontier=len(self.colony.shared_frontier),
                )
                if self.tracer is not None:
                    self.tracer.emit(
                        "agent_turn_skipped",
                        agent=agent.state.id,
                        oleada=self.oleada_count,
                        fetches=0,
                        budget=agent.state.budget,
                        frontier=len(self.colony.shared_frontier),
                    )

        # Update pheromone (per-agent private trails)
        best_path = max(agent_paths, key=lambda p: p.delta_q) if agent_paths else None
        self.pheromone.update(
            agent_paths,
            best_path,
            self.colony.agent_states,
            rho=self.cfg.aco.rho,
            lambda_elite=self.cfg.aco.lambda_elite,
            tau_min=self.cfg.aco.tau_min,
            tau_max=self.cfg.aco.tau_max,
        )

        # Track best agent (snapshot narrative at peak Q)
        prev_best = self.colony.best_quality
        self.colony.update_best(self.oleada_count)
        if self.tracer is not None and self.colony.best_quality > prev_best:
            self.tracer.emit(
                "new_best",
                agent=self.colony.best_snapshot_agent,
                Q=round(self.colony.best_quality, 4),
                oleada=self.oleada_count,
            )

        elapsed = time.monotonic() - oleada_start_time
        ranking = sorted(
            [(a.state.id, a.state.quality) for a in self.colony.agents],
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        log.info(
            "oleada_complete",
            oleada=self.oleada_count,
            best_Q=self.colony.best_quality,
            total_fetches=self._total_fetches,
            max_fetches=self.cfg.budget.max_fetches,
            elapsed=elapsed,
            ranking=ranking,
        )
        if self.tracer is not None:
            self.tracer.emit(
                "oleada_complete",
                oleada=self.oleada_count,
                best_Q=round(self.colony.best_quality, 4),
                total_fetches=self._total_fetches,
                max_fetches=self.cfg.budget.max_fetches,
                elapsed=round(elapsed, 1),
                ranking=ranking,
            )

        self.history.append(
            {
                "oleada": self.oleada_count,
                "agents": [a.state.id for a in k_agents],
                "best_Q": self.colony.best_quality,
                "fetches": self._total_fetches,
                "elapsed": round(elapsed, 1),
                "ranking": ranking,
            }
        )

    def _pick_top_k(self) -> list[ExplorerAgent]:
        """Pick the K agents with the highest priority that aren't exhausted."""
        candidates = self.colony.active_candidates()
        if not candidates:
            return []
        # Sort by priority descending
        ranked = sorted(candidates, key=lambda a: self._priority(a), reverse=True)
        return ranked[: self.cfg.aco.max_concurrent]

    def _priority(self, agent: ExplorerAgent) -> float:
        """priority(a) = Q_a + gamma * EV(F_a) + delta * b_a"""
        ev = self._frontier_value(agent)
        return (
            agent.state.quality
            + self.cfg.scheduler.gamma * ev
            + self.cfg.scheduler.delta * agent.state.budget
        )

    def _frontier_value(self, agent: ExplorerAgent) -> float:
        """EV(F_a) = sum of eta(v) for top frontier candidates (capped)."""
        if not self.colony.shared_frontier:
            return 0.0
        return min(1.0, len(self.colony.shared_frontier) / 20.0)

    def _get_new_papers(
        self, agent: ExplorerAgent, edges: list[tuple[str, str, str]]
    ) -> list:
        """Get the Paper objects for papers added this turn (for peer voting)."""
        from research_explorer.graph.models import Paper

        papers: list[Paper] = []
        for (_, dst, _) in edges:
            p = self.colony.graph.get_paper(dst)
            if p is not None:
                papers.append(p)
        return papers

    @property
    def total_fetches(self) -> int:
        return self._total_fetches
