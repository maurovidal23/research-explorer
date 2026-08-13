"""Structural metrics (R) of the quality function Q.

Deterministic, 0 LLM calls. Computes:
  - Coverage: |V_a| / L (target context size)
  - Diversity: 1 - avg pairwise embedding similarity (penalizes redundancy)
  - Depth/seminality: avg normalized citation count
  - Coherence: fraction of V_a connected to the seed lineage
"""

from __future__ import annotations

import math

import numpy as np

from research_explorer.agents.state import AgentState
from research_explorer.graph.store import GraphStore


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


def _cosine_sim(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm == 0:
        return 0.0
    return float(np.dot(va, vb) / norm)


class StructuralMetrics:
    """Structural metrics (R) — deterministic, no LLM calls."""

    def __init__(self, graph: GraphStore, target_size: int = 30):
        self.graph = graph
        self.target_size = target_size

    def compute(self, state: AgentState) -> float:
        """Compute R ∈ [0, 1] from the agent's structural properties."""
        if not state.visited:
            return 0.0

        coverage = self._coverage(state)
        diversity = self._diversity(state)
        depth = self._depth(state)
        coherence = self._coherence(state)

        # Equal weights by default (configurable via the quality weights if needed)
        return 0.25 * coverage + 0.25 * diversity + 0.25 * depth + 0.25 * coherence

    def _coverage(self, state: AgentState) -> float:
        """Coverage = min(1, |V_a| / L)."""
        return min(1.0, len(state.visited) / self.target_size)

    def _diversity(self, state: AgentState) -> float:
        """Diversity = 1 - avg pairwise embedding similarity."""
        embeddings: list[list[float]] = []
        for nid in state.visited:
            emb = self.graph.get_paper_embedding(nid)
            if emb is not None:
                embeddings.append(emb)
        if len(embeddings) < 2:
            return 1.0  # can't be redundant with < 2 papers

        sims: list[float] = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sims.append(_cosine_sim(embeddings[i], embeddings[j]))
        avg_sim = sum(sims) / len(sims) if sims else 0.0
        return max(0.0, 1.0 - avg_sim)

    def _depth(self, state: AgentState) -> float:
        """Depth = avg normalized citation count (seminality signal)."""
        counts: list[float] = []
        for nid in state.visited:
            summary = self.graph.get_paper_summary(nid)
            if summary and summary.citation_count is not None:
                counts.append(summary.citation_count)
        if not counts:
            return 0.0
        avg = sum(counts) / len(counts)
        return _sigmoid(math.log1p(avg) / 5.0)

    def _coherence(self, state: AgentState) -> float:
        """Coherence = fraction of V_a reachable from seed via the agent's private edges."""
        if not state.visited:
            return 0.0
        if len(state.visited) <= 1:
            return 1.0
        seed = state.visited[0]
        visited_set = set(state.visited)
        reachable: set[str] = {seed}
        queue: list[str] = [seed]
        while queue:
            node = queue.pop(0)
            refs = state.local_references(node)
            cits = state.local_citants(node)
            for n in refs + cits:
                if n in visited_set and n not in reachable:
                    reachable.add(n)
                    queue.append(n)
        return len(reachable) / len(state.visited)
