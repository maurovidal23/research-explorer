"""Agent state — the persistible state of an ACO agent between turns.

Kept intentionally light: IDs + narrative text + scalars.
Heavy metadata lives in the shared GraphStore (not duplicated per agent).

Each agent keeps its OWN incomplete, private view of the citation graph
(local_refs / local_cits) and its OWN private pheromone trail. Edges and
pheromone are NOT shared across agents — agents explore independently and
compete on their separately-built graphs.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from research_explorer.memory.models import (
    Concept,
    ContentKind,
    KnowledgeGap,
    LedgerClaim,
    PaperDossier,
    PaperRelation,
)
from research_explorer.memory.synthesis import synthesize_memory


def normalize_narrative(value: Any, fallback: str = "") -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts: list[str] = []
        for k in sorted(value, key=str):
            v = value[k]
            if isinstance(v, str):
                parts.append(f"**{k}**: {v}")
            elif v is not None:
                parts.append(
                    f"**{k}**: {json.dumps(v, ensure_ascii=False, sort_keys=True, default=str)}"
                )
        return "\n\n".join(parts) if parts else fallback
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, str):
                lines.append(f"- {item}")
            elif isinstance(item, dict):
                title = item.get("title") or item.get("name")
                if title is None:
                    title = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
                lines.append(f"- {title}")
            else:
                lines.append(f"- {item!s}")
        return "\n".join(lines) if lines else fallback
    return fallback


def _pkey(src: str, dst: str, mode: str) -> str:
    return f"{src}|{dst}|{mode}"


class AgentState(BaseModel):
    """Persistible state of an ACO agent."""

    id: str = Field(description="Unique agent ID")
    pos: str = Field(description="Normalized paper ID of current position")
    visited: list[str] = Field(default_factory=list, description="Normalized paper IDs visited")
    frontier: list[str] = Field(default_factory=list, description="Revealed but unvisited candidates")
    frontier_scores: dict[str, float] = Field(
        default_factory=dict,
        description="LLM evaluation scores for frontier papers (paper_id -> 0.0-1.0)",
    )
    frontier_sources: dict[str, tuple[str, str]] = Field(
        default_factory=dict,
        description="For each frontier paper: (source_paper_id, mode) of discovery",
    )
    narrative: str = Field(default="", description="Accumulated narrative (living document)")
    budget: int = Field(default=0, description="Remaining fetch budget")
    quality: float = Field(default=0.0, description="Current Q score")
    caste: str = Field(default="mixto", description="Caste: fundaciones | impacto | mixto")
    path: list[tuple[str, str]] = Field(
        default_factory=list,
        description="History of (paper_id, mode) moves this turn",
    )
    full_path: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Full history of (paper_id, mode) moves across all turns",
    )
    delta_q: float = Field(default=0.0, description="Quality change this turn")
    turn_count: int = Field(default=0, description="Number of turns completed")
    oleada: int = Field(default=0, description="Wave (oleada) of the current turn")

    local_refs: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Private outgoing edges: paper_id -> reference IDs discovered by this agent",
    )
    local_cits: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Private incoming edges: paper_id -> citation IDs discovered by this agent",
    )
    discovered: list[str] = Field(
        default_factory=list,
        description="Paper IDs whose neighbors have already been discovered (avoid re-reading)",
    )
    local_pheromone: dict[str, float] = Field(
        default_factory=dict,
        description="Private pheromone: 'src|dst|mode' -> tau",
    )
    paper_analyses: dict[str, dict] = Field(
        default_factory=dict,
        description="Per-paper structured analysis: paper_id -> {summary, key_concepts, methods, findings, relevance, limitations, key_references}",
    )
    metadata_transits: list[str] = Field(
        default_factory=list,
        description="Papers traversed as metadata-only (no readable content integrated, no evidence credit)",
    )
    fetch_work: int = Field(
        default=0,
        description="Cumulative budget-consuming provider work units (fetches + metadata transits)",
    )
    delta_work: int = Field(
        default=0,
        description="Provider work units consumed by the most recent turn",
    )

    # ---- Structured research memory (SURV-2/3/4) --------------------------
    research_scope: str = Field(
        default="",
        description="Effective research scope used to rank exploration",
    )
    scope_origin: str = Field(
        default="derived",
        description="Whether the scope was 'user' supplied or 'derived' from the seed",
    )
    dossiers: dict[str, PaperDossier] = Field(
        default_factory=dict,
        description="Typed, evidence-backed per-paper research memory",
    )
    claims: dict[str, LedgerClaim] = Field(
        default_factory=dict,
        description="Atomic typed claims, separate from rendered prose",
    )
    concepts: dict[str, Concept] = Field(default_factory=dict)
    relations: list[PaperRelation] = Field(default_factory=list)
    knowledge_gaps: list[KnowledgeGap] = Field(default_factory=list)
    extraction_failures: list[str] = Field(
        default_factory=list,
        description="Dossier extraction failures retained without deleting prior memory",
    )
    synthesis: str = Field(
        default="",
        description="Derived readable synthesis over structured memory (not authoritative)",
    )
    synthesis_words: int = Field(
        default=2000,
        description="Word budget used to derive the synthesis view",
    )

    def record_dossier(self, dossier: PaperDossier) -> bool:
        """Persist a dossier, preserving prior valid memory on failed re-extraction."""
        existing = self.dossiers.get(dossier.paper_id)
        if dossier.extraction_status == "failed":
            if dossier.extraction_error:
                self.extraction_failures.append(
                    f"{dossier.paper_id}: {dossier.extraction_error}"
                )
            if existing is not None and existing.has_evidence_credit:
                return False
            if existing is not None:
                self.dossiers[dossier.paper_id] = dossier
                return True
            self.dossiers[dossier.paper_id] = dossier
            return True
        self.dossiers[dossier.paper_id] = dossier
        return True

    def record_claims(self, claims: list[LedgerClaim]) -> None:
        for claim in claims:
            existing = self.claims.get(claim.id)
            if existing is None:
                self.claims[claim.id] = claim
            else:
                from research_explorer.memory.extract import merge_claims

                self.claims[claim.id] = merge_claims(existing, claim)

    def record_relation(self, relation: PaperRelation) -> None:
        if all(existing.id != relation.id for existing in self.relations):
            self.relations.append(relation)

    def record_gap(self, gap: KnowledgeGap) -> None:
        if all(existing.id != gap.id for existing in self.knowledge_gaps):
            self.knowledge_gaps.append(gap)

    def regenerate_synthesis(self, max_words: int | None = None) -> str:
        """Regenerate the derived synthesis view without mutating memory."""
        from research_explorer.memory.models import ResearchMemory

        if max_words is not None:
            self.synthesis_words = max_words
        memory = ResearchMemory(
            agent_id=self.id,
            scope=self.research_scope,
            scope_origin=self.scope_origin,
            dossiers=self.dossiers,
            claims=self.claims,
            concepts=self.concepts,
            relations=self.relations,
            gaps=self.knowledge_gaps,
        )
        self.synthesis = synthesize_memory(memory, self.synthesis_words)
        return self.synthesis

    def acquired_index(self) -> dict[str, ContentKind]:
        """paper_id -> acquired ContentKind for evidence resolution."""
        return {pid: dossier.content_kind for pid, dossier in self.dossiers.items()}

    def start_turn(self) -> None:
        """Reset per-turn accumulators."""
        self.path = []
        self.delta_q = 0.0
        self.delta_work = 0

    def visit(self, paper_id: str, mode: str) -> None:
        """Record a visit."""
        self.visited.append(paper_id)
        self.pos = paper_id
        self.path.append((paper_id, mode))
        self.full_path.append((paper_id, mode))
        self.budget -= 1
        self.fetch_work += 1
        self.delta_work += 1

    def record_transit(self, paper_id: str) -> None:
        """Record a metadata-only transit: budget consumed, no visit/evidence credit."""
        self.metadata_transits.append(paper_id)
        self.budget -= 1
        self.fetch_work += 1
        self.delta_work += 1

    def add_to_frontier(
        self, paper_ids: list[str], source: str = "", mode: str = "ref"
    ) -> None:
        """Add newly revealed candidates to the frontier (deduplicated, unvisited)."""
        for pid in paper_ids:
            if pid not in self.visited and pid not in self.frontier:
                self.frontier.append(pid)
                self.frontier_sources[pid] = (source, mode)

    def remove_from_frontier(self, paper_id: str) -> None:
        """Remove a visited candidate from the frontier."""
        if paper_id in self.frontier:
            self.frontier.remove(paper_id)
        self.frontier_scores.pop(paper_id, None)
        self.frontier_sources.pop(paper_id, None)

    def set_frontier_score(self, paper_id: str, score: float) -> None:
        """Set the LLM evaluation score for a frontier paper."""
        if paper_id in self.frontier:
            self.frontier_scores[paper_id] = score

    def unevaluated_frontier(self) -> list[str]:
        """Return frontier papers that haven't been LLM-evaluated yet."""
        return [pid for pid in self.frontier if pid not in self.frontier_scores]

    def best_frontier(self, exclude: set[str] | None = None) -> str | None:
        """Return the highest-scored paper in the frontier, excluding the given set."""
        if not self.frontier:
            return None
        exclude = exclude or set()
        best_id: str | None = None
        best_score = -1.0
        for pid in self.frontier:
            if pid in exclude:
                continue
            score = self.frontier_scores.get(pid, 0.0)
            if score > best_score:
                best_score = score
                best_id = pid
        return best_id

    def is_exhausted(self) -> bool:
        """True if the agent has no budget left."""
        return self.budget <= 0

    # ---- Private graph ---------------------------------------------------

    def mark_discovered(self, paper_id: str) -> None:
        if paper_id not in self.discovered:
            self.discovered.append(paper_id)

    def is_discovered(self, paper_id: str) -> bool:
        return paper_id in self.discovered

    def set_local_neighbors(
        self, paper_id: str, refs: list[str], cits: list[str]
    ) -> None:
        """Record the private outgoing/incoming edges of paper_id."""
        self.local_refs[paper_id] = list(refs)
        self.local_cits[paper_id] = list(cits)
        self.mark_discovered(paper_id)

    def local_references(self, paper_id: str) -> list[str]:
        return self.local_refs.get(paper_id, [])

    def local_citants(self, paper_id: str) -> list[str]:
        return self.local_cits.get(paper_id, [])

    def local_neighbors(self, paper_id: str) -> tuple[list[str], list[str]]:
        return self.local_references(paper_id), self.local_citants(paper_id)

    # ---- Private pheromone -----------------------------------------------

    def get_pheromone(self, src: str, dst: str, mode: str) -> float:
        return self.local_pheromone.get(_pkey(src, dst, mode), 1.0)

    def set_pheromone(self, src: str, dst: str, mode: str, tau: float) -> None:
        self.local_pheromone[_pkey(src, dst, mode)] = tau

    def evaporate_pheromone(self, rho: float, tau_min: float = 0.1) -> None:
        """tau <- max(tau_min, (1-rho)*tau) over the agent's private pheromone."""
        for k in list(self.local_pheromone.keys()):
            self.local_pheromone[k] = max(tau_min, (1.0 - rho) * self.local_pheromone[k])

    def clip_pheromone(self, tau_min: float, tau_max: float) -> None:
        for k in list(self.local_pheromone.keys()):
            self.local_pheromone[k] = min(tau_max, max(tau_min, self.local_pheromone[k]))

    def pheromone_concentration(self) -> float:
        """max/mean ratio of this agent's private pheromone (0.0 if empty)."""
        vals = list(self.local_pheromone.values())
        if not vals:
            return 0.0
        mx = max(vals)
        avg = sum(vals) / len(vals)
        return mx / avg if avg > 0 else 0.0
