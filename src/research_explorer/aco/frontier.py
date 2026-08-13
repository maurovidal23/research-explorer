"""Shared frontier — the colony-wide pool of candidate papers.

All agents pick from the same shared frontier (best-first search by LLM score).
This prevents agents from blocking each other when they discover the same seed
neighbors. The shared_visited set ensures no two agents explore the same paper.

Private per agent: narrative, pheromone, local graph edges (for structural metrics).
Shared across colony: frontier (candidates), visited set (claimed papers).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SharedFrontier:
    """Colony-wide pool of candidate papers with LLM evaluation scores."""

    papers: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    sources: dict[str, tuple[str, str]] = field(default_factory=dict)

    def add(
        self,
        paper_ids: list[str],
        source: str,
        mode: str,
        exclude: set[str] | None = None,
    ) -> None:
        """Add newly revealed candidates (deduplicated, excluding visited)."""
        for pid in paper_ids:
            if pid not in self.papers and (exclude is None or pid not in exclude):
                self.papers.append(pid)
                self.sources[pid] = (source, mode)

    def remove(self, paper_id: str) -> None:
        """Remove a visited/claimed candidate."""
        if paper_id in self.papers:
            self.papers.remove(paper_id)
        self.scores.pop(paper_id, None)
        self.sources.pop(paper_id, None)

    def set_score(self, paper_id: str, score: float) -> None:
        """Set the LLM evaluation score for a frontier paper."""
        if paper_id in self.papers:
            self.scores[paper_id] = score

    def unevaluated(self) -> list[str]:
        """Return frontier papers that haven't been LLM-evaluated yet."""
        return [pid for pid in self.papers if pid not in self.scores]

    def best(self, exclude: set[str] | None = None) -> str | None:
        """Return the highest-scored paper, excluding the given set."""
        if not self.papers:
            return None
        exclude = exclude or set()
        best_id: str | None = None
        best_score = -1.0
        for pid in self.papers:
            if pid in exclude:
                continue
            score = self.scores.get(pid, 0.0)
            if score > best_score:
                best_score = score
                best_id = pid
        return best_id

    def __len__(self) -> int:
        return len(self.papers)
