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
    REASON_EMPTY_WINNER_NARRATIVE,
    REASON_NO_EVALUATED_EVIDENCE,
    REASON_NO_WINNER,
    REASON_WINNER_EVALUATION_MISSING,
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


def resolve_examiner_key(api_key_env: str) -> str:
    """Resolve the examiner credential without reusing another provider's key.

    The generic LLM client falls back to ``NAN_API_KEY`` when no explicit key
    is supplied. For the examiner that would silently transmit the explorer
    credential to the configured examiner provider, so an unset examiner key
    variable fails closed instead.
    """
    key = get_api_key(api_key_env)
    if not key:
        raise RuntimeError(
            f"examiner API key environment variable {api_key_env!r} is not set; "
            "refusing to reuse the explorer credential for the examiner provider"
        )
    return key


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
        self.stop_reason: str = ""
        self.effective_scope: str = ""
        self.scope_origin: str = "derived"
        self.benchmark_result: object | None = None
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
        task = self._heartbeat_task
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
        setup_started = time.monotonic()
        tracer.emit("wave_phase_started", oleada=0, phase="setup", selected=[])
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

        # SURV-1: the positional text is an optional research scope, not the
        # literal final-answer question. When blank, derive a bounded profile.
        from research_explorer.memory.scope import resolve_scope

        self.effective_scope, self.scope_origin = resolve_scope(seed_query, seed_paper)
        tracer.emit(
            "research_scope_resolved",
            seed=seed_nid,
            scope=self.effective_scope,
            origin=self.scope_origin,
        )

        # 2. Initialize the colony (each agent reads the seed). The tracer is
        # attached before seed discovery so its telemetry is durable/replayable.
        tracer.emit("colony_init_started", seed=seed_nid)
        await self.colony.initialize(
            seed_nid,
            self.effective_scope,
            tracer=tracer,
            scope_origin=self.scope_origin,
        )
        tracer.emit(
            "colony_initialized",
            size=len(self.colony.agents),
            seed=seed_nid,
            agents=[a.state.id for a in self.colony.agents],
        )
        tracer.emit(
            "wave_phase_completed",
            oleada=0,
            phase="setup",
            selected=[],
            completed=[],
            failed=[],
            skipped=[],
            elapsed=round(time.monotonic() - setup_started, 1),
        )

        # 3. Run oleadas until convergence
        stop_reason = ""
        while not self.convergence.should_stop(
            self.colony.best_quality,
            self.scheduler.total_fetches,
            self.colony.pheromone_concentration(),
        ):
            await self.scheduler.run_oleada()

            # Check if all agents are exhausted
            if not self.colony.active_candidates():
                log.info("all_agents_exhausted")
                stop_reason = "all_agents_exhausted"
                break
        if not stop_reason:
            stop_reason = getattr(self.convergence, "last_reason", "") or "converged"
        self.stop_reason = stop_reason

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
                stop_reason=stop_reason,
                total_fetches=self.scheduler.total_fetches,
                total_waves=self.scheduler.oleada_count,
            )
        else:
            winner_id = self.colony.best_snapshot_agent or winner.state.id
            narrative = (self.colony.best_narrative or "").strip()
            degraded_reason = self._outcome_gap(winner_id, narrative)
            if degraded_reason:
                self.outcome = OUTCOME_DEGRADED
                self.terminal_reason = reason_text(degraded_reason)
                log.warning(
                    "completed_degraded",
                    reason_code=degraded_reason,
                    winner=winner_id,
                )
                tracer.emit(
                    "warning",
                    run_id=run_id,
                    classification="warning",
                    outcome=OUTCOME_DEGRADED,
                    reason_code=degraded_reason,
                    reason=self.terminal_reason,
                    winner=winner_id,
                    elapsed=round(self._elapsed, 1),
                )
                tracer.emit(
                    "orchestrator_complete",
                    run_id=run_id,
                    status="completed",
                    outcome=OUTCOME_DEGRADED,
                    reason_code=degraded_reason,
                    reason=self.terminal_reason,
                    winner=winner_id,
                    best_Q=round(winner.state.quality, 4),
                    peak_Q=round(self.colony.best_quality, 4),
                    snapshot_oleada=self.colony.best_snapshot_oleada,
                    stop_reason=stop_reason,
                    total_fetches=self.scheduler.total_fetches,
                    total_waves=self.scheduler.oleada_count,
                    oleadas=self.scheduler.oleada_count,
                    elapsed=round(self._elapsed, 1),
                )
                if narrative:
                    tracer.record_artifact(f"narrative_{winner_id}.md", "narrative", narrative)
            else:
                self.outcome = OUTCOME_COMPLETED
                self.terminal_reason = ""
                log.info(
                    "orchestrator_complete",
                    winner=winner_id,
                    best_Q=winner.state.quality,
                    peak_Q=self.colony.best_quality,
                    snapshot_oleada=self.colony.best_snapshot_oleada,
                    total_fetches=self.scheduler.total_fetches,
                    oleadas=self.scheduler.oleada_count,
                    elapsed=self._elapsed,
                    stop_reason=stop_reason,
                )
                tracer.emit(
                    "orchestrator_complete",
                    run_id=run_id,
                    status="completed",
                    outcome=OUTCOME_COMPLETED,
                    winner=winner_id,
                    best_Q=round(winner.state.quality, 4),
                    peak_Q=round(self.colony.best_quality, 4),
                    snapshot_oleada=self.colony.best_snapshot_oleada,
                    stop_reason=stop_reason,
                    total_fetches=self.scheduler.total_fetches,
                    total_waves=self.scheduler.oleada_count,
                    oleadas=self.scheduler.oleada_count,
                    elapsed=round(self._elapsed, 1),
                )
                tracer.record_artifact(
                    f"narrative_{winner_id}.md",
                    "narrative",
                    narrative,
                )

        if self.cfg.examination.enabled and winner is not None:
            await self._run_terminal_benchmark(seed_nid, tracer)

        self.trace.finish_run(
            run_id,
            "completed",
            best_quality=round(self.colony.best_quality, 6),
        )
        return self.colony.best_narrative if winner is not None else ""

    async def _run_terminal_benchmark(self, seed_nid: str, tracer: RunTracer) -> None:
        """Run the hidden examination and matched naive baseline (EXAM-1..9)."""
        try:
            await self._benchmark_impl(seed_nid, tracer)
        except Exception as exc:
            from research_explorer.events.models import (
                OUTCOME_SURVIVOR_UNBENCHMARKED,
                REASON_SURVIVOR_UNAVAILABLE,
            )

            self.outcome = OUTCOME_SURVIVOR_UNBENCHMARKED
            self.terminal_reason = redact_secrets(str(exc))
            tracer.emit(
                "warning",
                classification="warning",
                outcome=self.outcome,
                reason_code=REASON_SURVIVOR_UNAVAILABLE,
                reason=self.terminal_reason,
            )
        log.info("terminal_benchmark_done", outcome=self.outcome)

    async def _benchmark_impl(self, seed_nid: str, tracer: RunTracer) -> None:
        from research_explorer.examination import (
            BenchmarkConfig,
            BenchmarkRunner,
            FakeAnswerClient,
            FakeExaminer,
            SelectionWeights,
            build_evidence_pack,
            config_fingerprint,
        )
        from research_explorer.examination.benchmark import (
            OUTCOME_BENCHMARKED,
            OUTCOME_DEGRADED,
            OUTCOME_FAILED,
            OUTCOME_SURVIVOR_UNBENCHMARKED,
        )
        from research_explorer.examination.events import (
            evidence_pack_frozen_payload,
            exam_payloads,
            survivor_payloads,
        )
        from research_explorer.examination.report import (
            benchmark_report_markdown,
            benchmark_result_json,
            private_key_artifact,
            public_exam_artifact,
        )
        from research_explorer.memory.extract import memory_from_state

        exam = self.cfg.examination
        union: dict = {}
        for agent in self.colony.agents:
            for paper_id, dossier in agent.state.dossiers.items():
                union.setdefault(paper_id, dossier)
        if not union:
            self.outcome = OUTCOME_DEGRADED
            tracer.emit(
                "warning",
                classification="warning",
                outcome=self.outcome,
                reason_code="no_evidence_bearing_dossier",
            )
            return

        distances = {
            paper_id: ("seed" if paper_id == seed_nid else "direct_reference")
            for paper_id in union
        }
        pack = build_evidence_pack(
            seed_nid, self.effective_scope, union, distances=distances
        ).freeze()
        tracer.emit("evidence_pack_frozen", **evidence_pack_frozen_payload(pack))

        if exam.examiner_provider == "fake":
            generator = FakeExaminer()
            answer_client = FakeAnswerClient()
        else:
            generator, answer_client = self._build_live_exam_clients()

        candidates = []
        acquired: dict = {}
        for agent in self.colony.agents:
            state = agent.state
            candidates.append((state.id, state.quality, memory_from_state(state)))
            acquired[state.id] = state.acquired_index()

        weights = SelectionWeights(
            selection=self.cfg.terminal_selection.w_selection,
            process=self.cfg.terminal_selection.w_process,
            grounding=self.cfg.terminal_selection.w_grounding,
        )
        config = BenchmarkConfig(
            selection_count=exam.selection_count,
            holdout_count=exam.holdout_count,
            partition_seed=exam.partition_seed,
            weights=weights,
            min_coverage=exam.min_examination_coverage,
            context_max_chars=self.cfg.baseline.context_max_chars,
            config_fingerprint=config_fingerprint(
                {
                    "examiner": exam.examiner_model,
                    "answer": exam.answer_model or exam.examiner_model,
                    "seed": exam.partition_seed,
                    "selection": exam.selection_count,
                    "holdout": exam.holdout_count,
                }
            ),
            model_ids={
                "examiner_model": exam.examiner_model,
                "answer_model": exam.answer_model or exam.examiner_model,
            },
            prompt_versions={"examiner": "v1", "answer": "v1"},
        )
        runner = BenchmarkRunner(pack, generator, answer_client, config)
        result = await runner.run(candidates, acquired)
        self.benchmark_result = result

        if runner.bank is not None:
            for event_type, payload in exam_payloads(
                runner.bank,
                runner.bank.accepted_count,
                runner.bank.rejected_count,
                runner.bank.rejection_reasons,
            ):
                tracer.emit(event_type, **payload)
        for event_type, payload in survivor_payloads(result):
            tracer.emit(event_type, **payload)

        if runner.bank is not None:
            tracer.record_artifact(
                "exam_public.json", "exam", public_exam_artifact(runner.bank)
            )
        if runner.answer_key is not None:
            tracer.record_private_artifact(
                "exam_key.private.json", "answer_key", private_key_artifact(runner.answer_key)
            )
        tracer.record_artifact(
            "benchmark_result.json", "benchmark", benchmark_result_json(result)
        )
        tracer.record_artifact(
            "benchmark_report.md",
            "report",
            benchmark_report_markdown(result, pack, scope=self.effective_scope, bank=runner.bank),
        )

        outcome_map = {
            OUTCOME_BENCHMARKED: OUTCOME_BENCHMARKED,
            OUTCOME_SURVIVOR_UNBENCHMARKED: OUTCOME_SURVIVOR_UNBENCHMARKED,
            OUTCOME_DEGRADED: OUTCOME_DEGRADED,
            OUTCOME_FAILED: OUTCOME_FAILED,
        }
        self.outcome = outcome_map.get(result.outcome, OUTCOME_DEGRADED)
        self.terminal_reason = result.reason or result.reason_code
        tracer.emit(
            "benchmark_completed",
            outcome=self.outcome,
            reason_code=result.reason_code,
            reason=self.terminal_reason,
            survivor=result.survivor_id,
            survivor_accuracy=result.survivor_accuracy,
            naive_accuracy=result.naive_accuracy,
            uplift=result.uplift,
        )

    def _build_live_exam_clients(self):
        """Build the examiner/answer clients (live path, not used by tests)."""
        from research_explorer.agents.llm_client import LLMClient
        from research_explorer.examination import LLMAnswerClient, LLMExaminer

        exam = self.cfg.examination
        examiner_llm = LLMClient(
            base_url=exam.examiner_base_url,
            api_key=resolve_examiner_key(exam.examiner_api_key_env),
            max_concurrent=self.cfg.llm.max_concurrent,
            rpm=self.cfg.llm.rpm,
        )
        answer_model = exam.answer_model or exam.examiner_model
        self._examiner_llm = examiner_llm
        return (
            LLMExaminer(examiner_llm, exam.examiner_model, exam.examiner_max_tokens),
            LLMAnswerClient(
                examiner_llm,
                answer_model,
                temperature=exam.answer_temperature,
                max_tokens=exam.answer_max_tokens,
                reasoning_effort=exam.answer_reasoning_effort or None,
            ),
        )

    def _outcome_gap(self, winner_id: str, narrative: str) -> str:
        """Return the stable degraded reason for an otherwise-completed run.

        A normal success requires a non-empty narrative, at least one
        evidence-bearing evaluated turn, and a terminal evaluation for the
        winner. Anything else is surfaced as ``completed``/``degraded`` rather
        than as a fabricated success.
        """
        if not narrative:
            return REASON_EMPTY_WINNER_NARRATIVE
        evaluations = getattr(self.scheduler, "evaluations", []) or []
        completed = [r for r in evaluations if getattr(r, "status", "complete") == "complete"]
        if not any(getattr(r, "new_papers", []) for r in completed):
            return REASON_NO_EVALUATED_EVIDENCE
        if not any(r.agent_id == winner_id for r in completed):
            return REASON_WINNER_EVALUATION_MISSING
        return ""

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
            stop_reason=self.stop_reason,
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
        examiner_llm = getattr(self, "_examiner_llm", None)
        if examiner_llm is not None:
            await examiner_llm.aclose()
        for p in self.providers.values():
            await p.aclose()
        self.graph.close()
        self.trace.close()
