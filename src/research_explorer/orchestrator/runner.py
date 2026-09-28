"""Orchestrator — the main ACO loop that ties everything together.

Initializes the colony, runs oleadas until convergence, and returns the
winning agent's narrative.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import time

from research_explorer.aco.colony import Colony
from research_explorer.aco.convergence import ConvergenceChecker
from research_explorer.aco.scheduler import Scheduler
from research_explorer.agents.llm_client import LLMClient
from research_explorer.config import Config, get_api_key
from research_explorer.evaluation.structural import StructuralMetrics
from research_explorer.events.models import (
    OUTCOME_COMPLETED,
    OUTCOME_DEGRADED,
    REASON_NO_WINNER,
    STATUS_RUNNING,
    reason_text,
)
from research_explorer.events.sink import EventSink
from research_explorer.graph.embeddings import EmbeddingService
from research_explorer.graph.feromone import PheromoneManager
from research_explorer.graph.models import normalize_id
from research_explorer.graph.store import GraphStore
from research_explorer.logging_setup import configure_logging, get_logger
from research_explorer.providers.base import ResilientProvider
from research_explorer.providers.factory import build_all_providers
from research_explorer.providers.routing import SeedRef, route_seed_provider
from research_explorer.redaction import redact_secrets
from research_explorer.replay.trace import (
    HEARTBEAT_INTERVAL_SECONDS,
    RunTracer,
    RunTraceStore,
)

log = get_logger("orchestrator")


class Orchestrator:
    """Runs the full ACO exploration loop.

    Usage:
        orch = Orchestrator(config)
        narrative = await orch.run(seed_paper_id="10.1038/nrn3241", seed_query="...")
    """

    def __init__(self, config: Config, event_sink: EventSink | None = None):
        self.cfg = config
        self.event_sink = event_sink
        self.run_id: str = ""
        self.outcome: str = OUTCOME_COMPLETED
        self.terminal_reason: str = ""
        configure_logging(config.log_level)

        # Storage
        self.graph = GraphStore(config.storage.db_path)
        self.pheromone = PheromoneManager()

        # LLM
        api_key = get_api_key(config.llm.api_key_env)
        self.llm = LLMClient(
            base_url=config.llm.base_url,
            api_key=api_key,
            max_concurrent=config.llm.max_concurrent,
            rpm=config.llm.rpm,
        )

        # Embedding service (wraps LLM embed with SQLite cache)
        self.embedding = EmbeddingService(
            self.graph, embedder=self.llm.embed
        )

        # Provider registry: all active providers, for cross-provider fetching.
        # Never construct a fallback provider that is not enabled.
        self.providers: dict[str, ResilientProvider] = build_all_providers(config)
        if not self.providers:
            raise ValueError("No providers enabled; set providers.active in the config.")
        self.provider: ResilientProvider = self.providers.get(
            config.providers.default
        ) or next(iter(self.providers.values()))

        # Colony + scheduler
        self.colony = Colony(
            graph=self.graph,
            llm=self.llm,
            embedding=self.embedding,
            provider=self.provider,
            providers=self.providers,
            config=config,
        )
        self.structural = StructuralMetrics(self.graph)
        self.scheduler = Scheduler(self.colony, config, self.pheromone, self.structural)
        self.convergence = ConvergenceChecker(config)

        # Evaluation replay trace store (lazily opened)
        self.trace = RunTraceStore(config.storage.trace_db_path)
        self.tracer: RunTracer | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None

    def _provider_for_seed(self, seed_paper_id: str) -> tuple[ResilientProvider, SeedRef]:
        """Route the seed to a enabled, capable provider (see providers.routing)."""
        return route_seed_provider(
            seed_paper_id, self.providers, self.cfg.providers.seed_routing
        )

    async def run(self, seed_paper_id: str, seed_query: str) -> str:
        """Run the full exploration and return the winning narrative.

        Args:
            seed_paper_id: The seed paper ID (DOI, S2 ID, PMID, arXiv ID, etc.).
            seed_query: A text description of the research line to explore.

        Returns:
            The narrative of the agent with the highest Q score.
        """
        start_time = time.monotonic()
        self._elapsed = 0.0

        seed_provider, seed_ref = self._provider_for_seed(seed_paper_id)

        run_id = self.trace.create_run(
            seed_paper_id,
            seed_query,
            config_json=json.dumps(dataclasses.asdict(self.cfg), default=str),
        )
        self.run_id = run_id
        self.tracer = RunTracer(self.trace, run_id, sink=self.event_sink)
        self.llm.tracer = self.tracer
        self.scheduler.tracer = self.tracer
        self.trace.start_heartbeat(run_id)
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        log.info(
            "orchestrator_start",
            seed=seed_paper_id,
            query=seed_query,
            colony_size=self.cfg.aco.colony_size,
            K=self.cfg.aco.max_concurrent,
            max_fetches=self.cfg.budget.max_fetches,
        )
        try:
            return await self._run_impl(
                seed_paper_id,
                seed_query,
                start_time,
                run_id,
                self.tracer,
                seed_provider,
                seed_ref,
            )
        except Exception as e:
            if self.tracer is not None:
                self.tracer.emit("run_failed", run_id=run_id, error=redact_secrets(str(e)))
                self.tracer.record_artifact(
                    "run_error.txt", "error", redact_secrets(str(e))
                )
            self.trace.finish_run(run_id, "failed")
            raise
        finally:
            await self._stop_heartbeat()

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
            with contextlib.suppress(Exception):
                self.trace.heartbeat(self.run_id)

    async def _stop_heartbeat(self) -> None:
        task = getattr(self, "_heartbeat_task", None)
        if task is None:
            return
        self._heartbeat_task = None
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _run_impl(
        self,
        seed_paper_id: str,
        seed_query: str,
        start_time: float,
        run_id: str,
        tracer: RunTracer,
        seed_provider: ResilientProvider,
        seed_ref: SeedRef,
    ) -> str:
        tracer.emit(
            "orchestrator_start",
            run_id=run_id,
            seed=seed_paper_id,
            query=seed_query,
            pipeline=self.cfg.pipeline,
            colony_size=self.cfg.aco.colony_size,
            K=self.cfg.aco.max_concurrent,
            k_per_turn=self.cfg.aco.k_per_turn,
            max_fetches=self.cfg.budget.max_fetches,
            budget_type=self.cfg.budget.type,
            max_time_seconds=self.cfg.budget.max_time_seconds,
            explorer_model=self.cfg.llm.explorer_model,
            judge_model=self.cfg.llm.judge_model,
        )

        # 1. Fetch the seed paper and cache it
        tracer.emit("seed_routing_started", seed=seed_paper_id)
        tracer.emit(
            "seed_routed",
            seed=seed_paper_id,
            kind=seed_ref.kind.value,
            normalized=seed_ref.value,
            provider=seed_provider.name,
        )
        seed_paper = await seed_provider.get_paper(seed_ref.fetch_value)
        if seed_paper is None:
            raise RuntimeError(f"Could not fetch seed paper: {seed_paper_id}")
        self.graph.cache_paper(seed_paper)
        seed_nid = normalize_id(seed_paper.provider, seed_paper.id)

        # 2. Initialize the colony (each agent reads the seed). The tracer is
        # attached before seed discovery so its telemetry is durable/replayable.
        tracer.emit("colony_init_started", seed=seed_nid)
        await self.colony.initialize(seed_nid, seed_query, tracer=tracer)
        tracer.emit(
            "colony_initialized",
            size=len(self.colony.agents),
            seed=seed_nid,
            agents=[a.state.id for a in self.colony.agents],
        )

        # 3. Run oleadas until convergence
        while not self.convergence.should_stop(
            self.colony.best_quality,
            self.scheduler.total_fetches,
            self.colony.pheromone_concentration(),
        ):
            await self.scheduler.run_oleada()

            # Check if all agents are exhausted
            if not self.colony.active_candidates():
                log.info("all_agents_exhausted")
                break

        # 4. Final event/artifact before marking run complete
        winner = self.colony.best_agent
        self._elapsed = time.monotonic() - start_time

        if winner is None:
            reason_code = self.colony.init_reason
            reason = self.colony.init_reason_text or reason_text(REASON_NO_WINNER)
            self.outcome = OUTCOME_DEGRADED
            self.terminal_reason = reason
            log.warning("no_winner", reason_code=reason_code or REASON_NO_WINNER)
            # The warning precedes the terminal event so the timeline and the
            # warning-filtered event view explain the stop.
            tracer.emit(
                "warning",
                run_id=run_id,
                classification="warning",
                outcome=OUTCOME_DEGRADED,
                reason_code=reason_code or REASON_NO_WINNER,
                reason=reason,
                elapsed=round(self._elapsed, 1),
            )
            tracer.emit(
                "no_winner",
                run_id=run_id,
                status="completed",
                outcome=OUTCOME_DEGRADED,
                reason_code=reason_code or REASON_NO_WINNER,
                reason=reason,
                elapsed=round(self._elapsed, 1),
                total_fetches=self.scheduler.total_fetches,
                total_waves=self.scheduler.oleada_count,
            )
        else:
            self.outcome = OUTCOME_COMPLETED
            self.terminal_reason = ""
            log.info(
                "orchestrator_complete",
                winner=self.colony.best_snapshot_agent,
                best_Q=winner.state.quality,
                peak_Q=self.colony.best_quality,
                snapshot_oleada=self.colony.best_snapshot_oleada,
                total_fetches=self.scheduler.total_fetches,
                oleadas=self.scheduler.oleada_count,
                elapsed=self._elapsed,
            )
            tracer.emit(
                "orchestrator_complete",
                run_id=run_id,
                status="completed",
                outcome=OUTCOME_COMPLETED,
                winner=self.colony.best_snapshot_agent,
                best_Q=round(winner.state.quality, 4),
                peak_Q=round(self.colony.best_quality, 4),
                snapshot_oleada=self.colony.best_snapshot_oleada,
                total_fetches=self.scheduler.total_fetches,
                total_waves=self.scheduler.oleada_count,
                oleadas=self.scheduler.oleada_count,
                elapsed=round(self._elapsed, 1),
            )
            tracer.record_artifact(
                f"narrative_{self.colony.best_snapshot_agent}.md",
                "narrative",
                self.colony.best_narrative,
            )

        self.trace.finish_run(
            run_id,
            "completed",
            best_quality=round(self.colony.best_quality, 6),
        )
        return self.colony.best_narrative if winner is not None else ""

    def mark_cancelled(self) -> None:
        """Persist a distinct ``cancelled`` status for the active run.

        A run already terminal in the trace store is never relabelled by a stale
        UI cancellation (TUI-REL-3).
        """
        if not self.run_id:
            return
        run = self.trace.get_run(self.run_id)
        if run is not None and run.get("status") not in (None, STATUS_RUNNING):
            return
        if self.tracer is not None:
            self.tracer.emit("run_cancelled", run_id=self.run_id)
        self.trace.finish_run(self.run_id, "cancelled")

    def generate_report(self, seed_paper_id: str, seed_query: str) -> str:
        """Build a full markdown exploration report after run() has completed."""
        from research_explorer.orchestrator.report import build_report

        return build_report(
            config=self.cfg,
            colony=self.colony,
            scheduler=self.scheduler,
            convergence=self.convergence,
            graph=self.graph,
            seed_paper_id=seed_paper_id,
            seed_query=seed_query,
            elapsed=getattr(self, "_elapsed", 0.0),
            outcome=self.outcome,
            terminal_reason=self.terminal_reason,
        )

    def generate_obsidian(self, seed_query: str, output_dir: str = "obsidian") -> str | None:
        """Generate an Obsidian-compatible graph for the winner agent.

        Returns the path to the generated folder, or None if no winner.
        """
        from research_explorer.orchestrator.obsidian import generate_obsidian_graph

        winner = self.colony.best_agent
        if winner is None:
            return None
        return generate_obsidian_graph(
            agent_state=winner.state,
            graph=self.graph,
            seed_query=seed_query,
            output_dir=output_dir,
            shared_frontier=self.colony.shared_frontier,
        )

    async def aclose(self) -> None:
        """Clean up resources."""
        await self._stop_heartbeat()
        await self.llm.aclose()
        for p in self.providers.values():
            await p.aclose()
        self.graph.close()
        self.trace.close()
