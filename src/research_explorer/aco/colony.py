"""Colony — manages the population of ACO agents.

Initializes N agents at the seed, assigns castes, and tracks the best agent.
Bibliographic facts and topology live in the shared GraphStore; the seed's
neighbors are discovered concurrently during initialization.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from collections import Counter
from typing import Literal

from research_explorer.aco.frontier import SharedFrontier
from research_explorer.agents.explorer import DiscoveryOutcome, ExplorerAgent
from research_explorer.agents.llm_client import LLMClient
from research_explorer.agents.state import AgentState
from research_explorer.config import Config
from research_explorer.events.models import (
    REASON_NO_NEIGHBORS_DISCOVERED,
    REASON_NO_TRAVERSABLE_IDENTIFIERS,
    REASON_REFERENCE_EXTRACTION_FAILED,
    REASON_REFERENCE_MAPPING_INCOMPLETE,
    REASON_SEED_DISCOVERY_FAILED,
    reason_text,
)
from research_explorer.graph.embeddings import EmbeddingService
from research_explorer.graph.store import GraphStore
from research_explorer.logging_setup import get_logger
from research_explorer.providers.base import ResilientProvider
from research_explorer.redaction import redact_secrets
from research_explorer.references.builder import build_reference_builder
from research_explorer.replay.trace import RunTracer
from research_explorer.resolution.traversal import build_neighbor_expander

log = get_logger("colony")

Caste = Literal["fundaciones", "impacto", "mixto"]


class Colony:
    """Manages the colony of ACO agents."""

    def __init__(
        self,
        graph: GraphStore,
        llm: LLMClient,
        embedding: EmbeddingService,
        provider: ResilientProvider,
        config: Config,
        providers: dict[str, ResilientProvider] | None = None,
        colony_seed: int | None = None,
    ):
        self.graph = graph
        self.llm = llm
        self.embedding = embedding
        self.provider = provider
        self.providers = providers or {provider.name: provider}
        self.cfg = config
        self.colony_seed = colony_seed
        self.expander = build_neighbor_expander(config, self.providers, graph)
        self.agents: list[ExplorerAgent] = []
        self.seed_id: str = ""
        self.seed_query: str = ""
        self.seed_embedding: list[float] | None = None
        self.shared_visited: set[str] = set()
        self.shared_frontier = SharedFrontier()
        self.reference_builder = build_reference_builder(
            config,
            self.providers,
            graph,
            llm,
            frontier=self.shared_frontier,
            visited=self.shared_visited,
        )
        self._best_agent: ExplorerAgent | None = None
        self._best_quality: float = 0.0
        self._best_narrative: str = ""
        self._best_snapshot_agent: str = ""
        self._best_snapshot_oleada: int = 0
        self._current_oleada: int = 0
        self.init_reason: str = ""
        self.init_failures: list[str] = []

    async def initialize(
        self, seed_id: str, seed_query: str, tracer: RunTracer | None = None
    ) -> None:
        """Initialize the colony: N agents at the seed with assigned castes.

        The run tracer is attached to every agent *before* seed neighbor
        discovery so the seed-discovery telemetry is durable and replayable.
        """
        self.seed_id = seed_id
        self.seed_query = seed_query

        # Compute seed embedding for eta heuristic
        seed_paper = self.graph.get_paper(seed_id)
        if seed_paper:
            text = f"{seed_paper.title} {seed_paper.abstract or ''}"
            try:
                self.seed_embedding = await self.embedding.embed(text)
                self.graph.set_paper_embedding(seed_id, self.seed_embedding)
            except Exception as e:
                log.warning("seed_embedding_failed", error=str(e))

        # Create N agents
        n = self.cfg.aco.colony_size
        budget_per_agent = self._compute_budget_per_agent()
        castes = self._assign_castes(n)

        for i in range(n):
            state = AgentState(
                id=f"agent-{i:03d}-{uuid.uuid4().hex[:6]}",
                pos=seed_id,
                visited=[seed_id],
                frontier=[],
                narrative="",
                budget=budget_per_agent,
                caste=castes[i],
            )

            agent = ExplorerAgent(
                state=state,
                graph=self.graph,
                llm=self.llm,
                embedding=self.embedding,
                provider=self.provider,
                providers=self.providers,
                config=self.cfg,
                seed_query=seed_query,
                seed_embedding=self.seed_embedding,
                shared_visited=self.shared_visited,
                shared_frontier=self.shared_frontier,
                expander=self.expander,
                reference_builder=self.reference_builder,
                rng=self._agent_rng(i),
            )
            agent.tracer = tracer
            self.agents.append(agent)

        # Each agent reads the seed and discovers its neighbors into the shared
        # graph. Done concurrently.
        self.shared_visited.add(seed_id)
        results = await asyncio.gather(
            *(a._discover_neighbors(seed_id, wave=0) for a in self.agents),
            return_exceptions=True,
        )
        self._classify_seed_discovery(results)

        log.info(
            "colony_initialized",
            size=n,
            budget_per_agent=budget_per_agent,
            castes=dict(Counter(castes)),
            shared_frontier=len(self.shared_frontier),
            init_reason=self.init_reason or "ok",
        )

    def _classify_seed_discovery(self, results: list) -> None:
        """Contain initialization failures and classify an empty frontier.

        Exactly one primary reason is recorded so the terminal state stays
        actionable; per-agent failures are emitted as diagnostics instead of
        being silently discarded by ``gather(return_exceptions=True)``.
        """
        outcomes = [r for r in results if isinstance(r, DiscoveryOutcome)]
        self.init_failures = []
        for index, result in enumerate(results):
            if not isinstance(result, BaseException):
                continue
            agent = self.agents[index] if index < len(self.agents) else None
            agent_id = agent.state.id if agent is not None else ""
            error = redact_secrets(str(result))
            self.init_failures.append(f"{agent_id or 'agent'}: {error}")
        for failure in self.init_failures:
            log.warning("seed_discovery_failure", detail=failure)

        total_traversable = sum(o.traversable for o in outcomes)
        total_found = sum(o.found for o in outcomes)
        extraction_failed = any(o.extraction_failed for o in outcomes)
        mapping_incomplete = any(o.mapping_incomplete for o in outcomes)

        if total_traversable > 0:
            self.init_reason = ""
        elif self.init_failures or not outcomes:
            self.init_reason = REASON_SEED_DISCOVERY_FAILED
        elif total_found > 0:
            self.init_reason = REASON_NO_TRAVERSABLE_IDENTIFIERS
        elif extraction_failed:
            self.init_reason = REASON_REFERENCE_EXTRACTION_FAILED
        elif mapping_incomplete:
            self.init_reason = REASON_REFERENCE_MAPPING_INCOMPLETE
        else:
            self.init_reason = REASON_NO_NEIGHBORS_DISCOVERED

    @property
    def init_reason_text(self) -> str:
        return reason_text(self.init_reason) if self.init_reason else ""

    def _compute_budget_per_agent(self) -> int:
        """Distribute the global budget across the colony."""
        total = self.cfg.budget.max_fetches
        return max(1, total // self.cfg.aco.colony_size)

    def _agent_rng(self, i: int) -> random.Random:
        """Return a deterministic per-agent RNG when a colony seed is set."""
        if self.colony_seed is not None:
            return random.Random(self.colony_seed + i)
        return random.Random()

    def _assign_castes(self, n: int) -> list[str]:
        """Assign castes to agents for diversity.

        Roughly: 40% fundaciones, 30% impacto, 30% mixto.
        """
        n_fund = max(1, int(n * 0.4))
        n_imp = max(1, int(n * 0.3))
        castes = (
            ["fundaciones"] * n_fund
            + ["impacto"] * n_imp
            + ["mixto"] * (n - n_fund - n_imp)
        )
        return castes[:n]

    def update_best(self, oleada: int = 0) -> ExplorerAgent | None:
        """Track the agent with the highest Q across the colony.

        When a new best Q is found, snapshot the narrative at that point
        so later degradation doesn't overwrite the peak narrative.
        """
        self._current_oleada = oleada
        for agent in self.agents:
            if agent.state.quality > self._best_quality:
                self._best_quality = agent.state.quality
                self._best_agent = agent
                self._best_narrative = agent.state.narrative
                self._best_snapshot_agent = agent.state.id
                self._best_snapshot_oleada = oleada
                log.info(
                    "new_best",
                    agent=agent.state.id,
                    Q=agent.state.quality,
                    oleada=oleada,
                )
        return self._best_agent

    @property
    def best_narrative(self) -> str:
        """The narrative snapshot at the moment of peak Q."""
        return self._best_narrative

    @property
    def best_snapshot_agent(self) -> str:
        return self._best_snapshot_agent

    @property
    def best_snapshot_oleada(self) -> int:
        return self._best_snapshot_oleada

    @property
    def best_agent(self) -> ExplorerAgent | None:
        return self._best_agent

    @property
    def best_quality(self) -> float:
        return self._best_quality

    def total_fetches_used(self) -> int:
        """Count total fetches used across all agents."""
        return sum(
            self.cfg.budget.max_fetches // self.cfg.aco.colony_size - a.state.budget
            for a in self.agents
        )

    def active_candidates(self) -> list[ExplorerAgent]:
        """Agents that still have budget and there are unclaimed frontier papers."""
        has_candidates = self.shared_frontier.best(exclude=self.shared_visited) is not None
        if not has_candidates:
            return []
        return [a for a in self.agents if not a.state.is_exhausted()]

    @property
    def agent_states(self) -> list[AgentState]:
        return [a.state for a in self.agents]

    def pheromone_concentration(self) -> float:
        """Colony-level pheromone concentration: max over agents' private trails.

        With per-agent private pheromone there is no single shared trail, so we
        take the most-concentrated agent as the convergence signal.
        """
        if not self.agents:
            return 0.0
        return max(a.state.pheromone_concentration() for a in self.agents)
