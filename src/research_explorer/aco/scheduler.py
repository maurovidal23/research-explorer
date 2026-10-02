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
from research_explorer.evaluation.availability import REASON_MODEL_FAILED
from research_explorer.evaluation.quality import QualityAssessor
from research_explorer.evaluation.structural import StructuralMetrics
from research_explorer.events.models import (
    EVAL_COMPLETE,
    EVAL_FAILED,
    EVAL_SKIPPED,
    PHASE_DECISION,
    PHASE_EVALUATION,
    PHASE_RESEARCH,
)
from research_explorer.graph.feromone import AgentPath, PheromoneManager
from research_explorer.logging_setup import get_logger
from research_explorer.redaction import redact_secrets
from research_explorer.replay.models import DetailedEvaluation
from research_explorer.replay.trace import RunTracer

log = get_logger("scheduler")


class Scheduler:
    """Manages K concurrent agent slots with priority-based activation.

    A wave is executed through three explicit barriers: Research (fetch and
    integrate evidence), Evaluation (score the frozen research snapshot), and
    Decision (apply quality, pheromone, ranking, winner, and continuation).
    Research turns stay sequential because the shared graph and per-agent state
    are mutated in place and peer voting must observe one deterministic
    snapshot; the barrier guarantee holds regardless.
    """

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
        self.evaluations: list[DetailedEvaluation] = []
        self.last_decision: dict = {}
        self.last_research: list[dict] = []

    async def run_oleada(self) -> None:
        """Run one wave through the Research, Evaluation, and Decision barriers."""
        self.oleada_count += 1
        wave = self.oleada_count
        k_agents = self._pick_top_k()
        if not k_agents:
            log.info("no_active_agents")
            return

        wave_start_time = time.monotonic()
        selected_ids = [a.state.id for a in k_agents]

        log.info("oleada_start", oleada=wave, active=selected_ids)
        if self.tracer is not None:
            self.tracer.emit(
                "oleada_start",
                oleada=wave,
                active=selected_ids,
                phase=PHASE_RESEARCH,
                max_fetches=self.cfg.budget.max_fetches,
            )
            for queued in self.colony.agents:
                if queued not in k_agents:
                    self.tracer.emit(
                        "agent_turn_queued",
                        agent_id=queued.state.id,
                        oleada=wave,
                        turn=queued.state.turn_count,
                        state="waiting",
                    )

        research_started = time.monotonic()
        try:
            research = await self._run_research(wave, k_agents)
        except Exception as e:
            self._emit_phase_failed(wave, PHASE_RESEARCH, redact_secrets(str(e)))
            raise
        self.last_research = research
        self._emit_phase_completed(
            wave, PHASE_RESEARCH, self._research_summary(research, selected_ids), research_started
        )

        self._emit_phase_started(wave, PHASE_EVALUATION, selected_ids)
        evaluation_started = time.monotonic()
        try:
            eval_summary, records = await self._run_evaluation(wave, k_agents, research)
        except Exception as e:
            self._emit_phase_failed(wave, PHASE_EVALUATION, redact_secrets(str(e)))
            raise
        self._emit_phase_completed(
            wave, PHASE_EVALUATION, eval_summary, evaluation_started
        )

        self._emit_phase_started(wave, PHASE_DECISION, selected_ids)
        decision_started = time.monotonic()
        try:
            decision = self._run_decision(wave, k_agents, research, records)
        except Exception as e:
            self._emit_phase_failed(wave, PHASE_DECISION, redact_secrets(str(e)))
            raise
        self.last_decision = decision
        self._emit_phase_completed(wave, PHASE_DECISION, decision, decision_started)

        if self.tracer is not None:
            clear_context = getattr(self.tracer, "clear_context", None)
            if clear_context is not None:
                clear_context()

        elapsed = time.monotonic() - wave_start_time
        ranking = decision.get("ranking", [])

        log.info(
            "oleada_complete",
            oleada=wave,
            best_Q=self.colony.best_quality,
            total_fetches=self._total_fetches,
            max_fetches=self.cfg.budget.max_fetches,
            elapsed=elapsed,
            ranking=ranking,
            stop_reason=decision.get("stop_reason", ""),
        )
        if self.tracer is not None:
            self.tracer.emit(
                "oleada_complete",
                oleada=wave,
                best_Q=round(self.colony.best_quality, 4),
                total_fetches=self._total_fetches,
                max_fetches=self.cfg.budget.max_fetches,
                elapsed=round(elapsed, 1),
                ranking=ranking,
                leader=decision.get("leader", ""),
                q_delta=decision.get("q_delta", 0.0),
                stop_reason=decision.get("stop_reason", ""),
                continue_reason=decision.get("continue_reason", ""),
            )

        self.history.append(
            {
                "oleada": wave,
                "agents": selected_ids,
                "best_Q": self.colony.best_quality,
                "fetches": self._total_fetches,
                "elapsed": round(elapsed, 1),
                "ranking": ranking,
                "leader": decision.get("leader", ""),
                "q_delta": decision.get("q_delta", 0.0),
                "stop_reason": decision.get("stop_reason", ""),
                "continue_reason": decision.get("continue_reason", ""),
                "evaluation": eval_summary,
            }
        )

    async def _run_research(self, wave: int, k_agents: list[ExplorerAgent]) -> list[dict]:
        """Fetch and integrate up to ``k_per_turn`` papers per selected agent.

        Turns run sequentially so every selected agent observes the same shared
        graph/state snapshot at the Evaluation barrier. Failures are contained
        per agent and never erase another agent's work.
        """
        results: list[dict] = []
        for agent in k_agents:
            agent.state.oleada = wave
            if self.tracer is not None:
                set_context = getattr(self.tracer, "set_context", None)
                if set_context is not None:
                    set_context(
                        agent_id=agent.state.id,
                        oleada=wave,
                        turn=agent.state.turn_count,
                    )
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
                    agent_id=agent.state.id,
                    caste=agent.state.caste,
                    oleada=wave,
                    turn=agent.state.turn_count,
                )

            edges: list = []
            try:
                edges = await agent.take_turn(self.cfg.aco.k_per_turn)
            except Exception as e:
                error = redact_secrets(str(e))
                log.warning("agent_turn_failed", agent=agent.state.id, error=error)
                self.colony.shared_frontier.release_all(agent.state.id)
                agent.state.delta_q = 0.0
                if self.tracer is not None:
                    self.tracer.emit(
                        "agent_turn_failed",
                        agent_id=agent.state.id,
                        oleada=wave,
                        turn=agent.state.turn_count,
                        error=error,
                    )
                results.append(
                    {
                        "agent_id": agent.state.id,
                        "status": EVAL_FAILED,
                        "reason": error,
                        "edges": [],
                        "papers": [],
                        "attempted": 0,
                        "integrated": 0,
                        "evidence": 0,
                    }
                )
                continue

            new_papers = self._get_new_papers(agent, edges)
            self._total_fetches += max(0, agent.state.delta_work)
            if edges:
                if self.tracer is not None:
                    self.tracer.emit(
                        "agent_turn_complete",
                        agent=agent.state.id,
                        agent_id=agent.state.id,
                        oleada=wave,
                        turn=agent.state.turn_count,
                        fetches=len(edges),
                        budget=agent.state.budget,
                        frontier=len(self.colony.shared_frontier),
                    )
                log.info(
                    "agent_turn_complete",
                    agent=agent.state.id,
                    fetches=len(edges),
                    budget=agent.state.budget,
                    frontier=len(self.colony.shared_frontier),
                )
            else:
                if self.tracer is not None:
                    self.tracer.emit(
                        "agent_turn_skipped",
                        agent=agent.state.id,
                        agent_id=agent.state.id,
                        oleada=wave,
                        turn=agent.state.turn_count,
                        fetches=0,
                        budget=agent.state.budget,
                        frontier=len(self.colony.shared_frontier),
                    )
                log.info(
                    "agent_turn_skipped",
                    agent=agent.state.id,
                    fetches=0,
                    budget=agent.state.budget,
                    frontier=len(self.colony.shared_frontier),
                )
            results.append(
                {
                    "agent_id": agent.state.id,
                    "status": EVAL_COMPLETE if edges else EVAL_SKIPPED,
                    "reason": "" if edges else "no_new_evidence",
                    "edges": edges,
                    "papers": new_papers,
                    "attempted": len(edges),
                    "integrated": len(new_papers),
                    "evidence": len(edges),
                }
            )
        return results

    def _research_summary(self, research: list[dict], selected: list[str]) -> dict:
        status_by_id = {r["agent_id"]: r for r in research}
        return {
            "selected": selected,
            "completed": [r["agent_id"] for r in research if r["status"] == EVAL_COMPLETE],
            "failed": [r["agent_id"] for r in research if r["status"] == EVAL_FAILED],
            "skipped": [r["agent_id"] for r in research if r["status"] == EVAL_SKIPPED],
            "papers_attempted": sum(r["attempted"] for r in research),
            "papers_integrated": sum(r["integrated"] for r in research),
            "evidence_added": sum(r["evidence"] for r in research),
            "per_agent": [
                {
                    "agent_id": r["agent_id"],
                    "status": r["status"],
                    "papers_attempted": r["attempted"],
                    "papers_integrated": r["integrated"],
                    "evidence_added": r["evidence"],
                    "reason": r["reason"],
                }
                for r in status_by_id.values()
            ],
        }

    async def _run_evaluation(
        self, wave: int, k_agents: list[ExplorerAgent], research: list[dict]
    ) -> tuple[dict, list[DetailedEvaluation]]:
        """Evaluate every eligible selected agent against the frozen snapshot."""
        by_id = {r["agent_id"]: r for r in research}
        completed: list[str] = []
        failed: list[str] = []
        skipped: list[str] = []
        records: list[DetailedEvaluation] = []
        for agent in k_agents:
            info = by_id.get(agent.state.id, {})
            turn = agent.state.turn_count
            if self.tracer is not None:
                set_context = getattr(self.tracer, "set_context", None)
                if set_context is not None:
                    set_context(agent_id=agent.state.id, oleada=wave, turn=turn)
            if info.get("status") == EVAL_FAILED:
                failed.append(agent.state.id)
                self._emit_settled(agent.state.id, wave, turn, EVAL_FAILED, info.get("reason", ""))
                continue
            if not info.get("edges"):
                agent.state.delta_q = 0.0
                skipped.append(agent.state.id)
                log.info(
                    "evaluation_skipped",
                    agent=agent.state.id,
                    oleada=wave,
                    turn=turn,
                    reason="no_new_evidence",
                )
                if self.tracer is not None:
                    self.tracer.emit(
                        "evaluation_skipped",
                        agent_id=agent.state.id,
                        oleada=wave,
                        turn=turn,
                        reason="no_new_evidence",
                    )
                self._emit_settled(
                    agent.state.id, wave, turn, EVAL_SKIPPED, "no_new_evidence"
                )
                continue
            if self.tracer is not None:
                self.tracer.emit(
                    "quality_evaluation_started",
                    agent_id=agent.state.id,
                    oleada=wave,
                    turn=turn,
                )
            try:
                record = await self.assessor.assess_detail(
                    agent,
                    k_agents,
                    self.colony.seed_query,
                    info.get("papers", []),
                    oleada=wave,
                )
            except Exception as e:
                error = redact_secrets(str(e))
                log.warning("evaluation_failed", agent=agent.state.id, error=error)
                failed.append(agent.state.id)
                if self.tracer is not None:
                    self.tracer.emit(
                        "quality_evaluation_failed",
                        agent_id=agent.state.id,
                        oleada=wave,
                        turn=turn,
                        error=error,
                    )
                self._emit_settled(
                    agent.state.id, wave, turn, EVAL_FAILED, REASON_MODEL_FAILED
                )
                continue
            records.append(record)
            self.evaluations.append(record)
            completed.append(agent.state.id)
            if self.tracer is not None:
                self.tracer.record_evaluation(record)
                self.tracer.record_artifact(
                    f"narrative_{agent.state.id}_t{turn}.md",
                    "narrative",
                    agent.state.narrative,
                    agent_id=agent.state.id,
                    oleada=wave,
                    turn=turn,
                )
            self._emit_settled(
                agent.state.id,
                wave,
                turn,
                EVAL_COMPLETE,
                "",
                record=record,
            )
        best_q = max((r.q for r in records), default=None)
        summary = {
            "selected": [a.state.id for a in k_agents],
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "best_Q": best_q,
        }
        return summary, records

    def _emit_settled(
        self,
        agent_id: str,
        wave: int,
        turn: int,
        status: str,
        reason: str,
        record: DetailedEvaluation | None = None,
    ) -> None:
        if self.tracer is None:
            return
        payload: dict = {
            "agent_id": agent_id,
            "oleada": wave,
            "turn": turn,
            "status": status,
            "reason": reason,
        }
        if record is not None:
            payload.update(
                {
                    "Q": round(record.q, 4),
                    "delta_q": round(record.delta_q, 4),
                    "S": round(record.self_assessment.score, 4),
                    "P": round(record.peers.aggregated_score, 4),
                    "J": round(record.virgin_judge.score, 4),
                    "R": round(record.structural.r, 4),
                    "unavailable": dict(record.unavailable),
                    "evidence_papers": len(record.new_papers),
                }
            )
        else:
            payload["evidence_papers"] = 0
        self.tracer.emit("evaluation_settled", **payload)

    def _run_decision(
        self,
        wave: int,
        k_agents: list[ExplorerAgent],
        research: list[dict],
        records: list[DetailedEvaluation],
    ) -> dict:
        """Apply quality, pheromone, ranking, winner, and continuation state."""
        record_by_id = {r.agent_id: r for r in records}
        for agent in k_agents:
            record = record_by_id.get(agent.state.id)
            if record is not None:
                agent.state.quality = record.q
                agent.state.delta_q = record.delta_q
            else:
                agent.state.delta_q = 0.0

        by_id = {r["agent_id"]: r for r in research}
        agent_paths = [
            AgentPath(
                edges=by_id.get(a.state.id, {}).get("edges", []),
                delta_q=a.state.delta_q,
                state=a.state,
            )
            for a in k_agents
        ]
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

        prev_best = self.colony.best_quality
        self.colony.update_best(wave)
        if self.tracer is not None and self.colony.best_quality > prev_best:
            self.tracer.emit(
                "new_best",
                agent=self.colony.best_snapshot_agent,
                Q=round(self.colony.best_quality, 4),
                oleada=wave,
            )

        ranking = sorted(
            [(a.state.id, a.state.quality) for a in self.colony.agents],
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        leader = ranking[0][0] if ranking else ""
        leader_delta = next(
            (r.delta_q for r in records if r.agent_id == leader), 0.0
        )

        stop_reason = ""
        continue_reason = ""
        active = getattr(self.colony, "active_candidates", None)
        has_active = bool(active()) if callable(active) else True
        if not has_active:
            stop_reason = "all_agents_exhausted"
        elif (
            self.cfg.budget.type == "fetches"
            and self._total_fetches >= self.cfg.budget.max_fetches
        ):
            stop_reason = "budget_exhausted"
        else:
            continue_reason = "budget_remaining"

        concentration = 0.0
        concentration_of = getattr(self.colony, "pheromone_concentration", None)
        if callable(concentration_of):
            try:
                concentration = float(concentration_of())
            except Exception:
                concentration = 0.0

        return {
            "selected": [a.state.id for a in k_agents],
            "leader": leader,
            "best_Q": round(self.colony.best_quality, 4),
            "q_delta": round(leader_delta, 4),
            "ranking": ranking,
            "budget_used": self._total_fetches,
            "continue_reason": continue_reason,
            "stop_reason": stop_reason,
            "converged": bool(stop_reason),
            "pheromone_concentration": round(concentration, 4),
        }

    def _emit_phase_started(self, wave: int, phase: str, selected: list[str]) -> None:
        if self.tracer is None:
            return
        self.tracer.emit(
            "wave_phase_started",
            oleada=wave,
            phase=phase,
            selected=selected,
        )

    def _emit_phase_completed(
        self, wave: int, phase: str, summary: dict, started: float
    ) -> None:
        if self.tracer is None:
            return
        self.tracer.emit(
            "wave_phase_completed",
            oleada=wave,
            phase=phase,
            elapsed=round(time.monotonic() - started, 1),
            **summary,
        )

    def _emit_phase_failed(self, wave: int, phase: str, error: str) -> None:
        if self.tracer is None:
            return
        self.tracer.emit(
            "wave_phase_failed",
            oleada=wave,
            phase=phase,
            error=error,
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
