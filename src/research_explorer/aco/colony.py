"""Colony — manages the population of ACO agents.

Initializes N agents at the seed, assigns castes, and tracks the best agent.
Each agent keeps its own private graph; the seed's neighbors are discovered
per-agent (reading the seed) during initialization.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from collections import Counter
from typing import TYPE_CHECKING, Literal

from research_explorer.aco.diagnostics import SeedDiscovery, classify_empty_frontier
from research_explorer.aco.frontier import SharedFrontier
from research_explorer.agents.explorer import ExplorerAgent
from research_explorer.agents.llm_client import LLMClient
from research_explorer.agents.state import AgentState
from research_explorer.config import Config
from research_explorer.graph.embeddings import EmbeddingService
from research_explorer.graph.models import parse_normalized_id
from research_explorer.graph.store import GraphStore
from research_explorer.logging_setup import get_logger
from research_explorer.providers.base import ResilientProvider
from research_explorer.redaction import redact_secrets
from research_explorer.resolution.traversal import build_neighbor_expander

if TYPE_CHECKING:
    from research_explorer.replay.trace import RunTracer

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
        self._best_agent: ExplorerAgent | None = None
        self._best_quality: float = 0.0
        self._best_narrative: str = ""
        self._best_snapshot_agent: str = ""
        self._best_snapshot_oleada: int = 0
        self._current_oleada: int = 0
        self._seed_discovery_records: list[SeedDiscovery] = []

    async def initialize(
        self,
        seed_id: str,
        seed_query: str,
        tracer: RunTracer | None = None,
    ) -> None:
        """Initialize the colony: N agents at the seed with assigned castes.

        ``tracer`` is attached to every explorer *before* seed neighbor
        discovery starts, so the seed read emits structured telemetry.
        """
        self.seed_id = seed_id
        self.seed_query = seed_query
        self._seed_discovery_records = []

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
                rng=self._agent_rng(i),
            )
            if tracer is not None:
                agent.tracer = tracer
            self.agents.append(agent)

        # Each agent reads the seed and discovers its neighbors into its own
        # private graph (per-agent incomplete graph). Done concurrently, with
        # structured telemetry and contained failures.
        self.shared_visited.add(seed_id)
        await asyncio.gather(*(self._seed_discover(a, tracer) for a in self.agents))

        log.info(
            "colony_initialized",
            size=n,
            budget_per_agent=budget_per_agent,
            castes=dict(Counter(castes)),
            shared_frontier=len(self.shared_frontier),
            empty_frontier_reason=self.empty_frontier_reason(),
        )

    async def _seed_discover(
        self, agent: ExplorerAgent, tracer: RunTracer | None
    ) -> None:
        """Discover the seed's neighbors for one agent with telemetry.

        Concurrent initialization failures are contained per agent: they are
        logged, emitted as ``neighbor_discovery_failed``, and recorded for the
        empty-frontier classification instead of being silently discarded.
        """
        agent_id = agent.state.id
        seed_id = self.seed_id
        if tracer is not None:
            tracer.emit(
                "neighbor_discovery_started",
                agent_id=agent_id,
                paper_id=seed_id,
                seed=seed_id,
                oleada=0,
                turn=0,
            )
        failed = False
        try:
            await agent._discover_neighbors(seed_id)
        except Exception as exc:
            failed = True
            log.warning(
                "seed_discovery_failed",
                agent=agent_id,
                error=redact_secrets(str(exc)),
            )
            if tracer is not None:
                tracer.emit(
                    "neighbor_discovery_failed",
                    agent_id=agent_id,
                    paper_id=seed_id,
                    seed=seed_id,
                    oleada=0,
                    turn=0,
                    error=redact_secrets(str(exc)),
                )
        refs = agent.state.local_references(seed_id)
        cits = agent.state.local_citants(seed_id)
        traversable = self._traversable_count(refs) + self._traversable_count(cits)
        extraction_attempted = bool(getattr(agent, "extraction_attempted", False))
        extraction_failed = bool(getattr(agent, "extraction_failed", False))
        record = SeedDiscovery(
            agent_id=agent_id,
            refs=len(refs),
            cits=len(cits),
            traversable=traversable,
            failed=failed,
            extraction_attempted=extraction_attempted,
            extraction_failed=extraction_failed,
        )
        self._seed_discovery_records.append(record)
        if tracer is not None and not failed:
            tracer.emit(
                "neighbor_discovery_completed",
                agent_id=agent_id,
                paper_id=seed_id,
                seed=seed_id,
                oleada=0,
                turn=0,
                refs=len(refs),
                cits=len(cits),
                traversable=traversable,
                extraction_attempted=extraction_attempted,
                extraction_failed=extraction_failed,
            )

    @staticmethod
    def _traversable_count(paper_ids: list[str]) -> int:
        return sum(1 for pid in paper_ids if parse_normalized_id(pid)[0] != "unknown")

    @property
    def seed_discovery_records(self) -> list[SeedDiscovery]:
        return list(self._seed_discovery_records)

    def empty_frontier_reason(self) -> str | None:
        """The single primary reason the initial frontier is empty, if any."""
        return classify_empty_frontier(self._seed_discovery_records)

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
