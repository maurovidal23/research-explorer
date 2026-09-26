"""Shared frontier — the colony-wide pool of candidate papers.

All agents pick from the same shared frontier. To avoid two agents fetching
the same candidate concurrently, a frontier node carries a claim/lease: an
agent claims a node for the duration of its fetch and releases it afterwards.
Claiming a node only blocks it during the concurrent window — it does not
remove the node from the frontier and it does not erase the already-discovered
neighbor paths that hang off it, so reusable paths survive.

Transient provider failures do not evict a candidate (that would confuse
temporary failure with absence): the claim is released, the attempt count is
increased, and the node becomes eligible again after a bounded retry delay.
Only definitive absence removes a candidate.

Private per agent: narrative, pheromone, local graph edges (for structural metrics).
Shared across colony: frontier (candidates), visited set (claimed papers).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SharedFrontier:
    """Colony-wide pool of candidate papers with transition-relevant factors."""

    papers: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    sources: dict[str, tuple[str, str]] = field(default_factory=dict)
    claims: dict[str, str | None] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    next_eligible_turn: dict[str, int] = field(default_factory=dict)
    max_attempts: int = 3
    retry_backoff: int = 1

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
        """Remove a visited/claimed candidate (e.g. definitive absence)."""
        if paper_id in self.papers:
            self.papers.remove(paper_id)
        self.scores.pop(paper_id, None)
        self.sources.pop(paper_id, None)
        self.claims.pop(paper_id, None)
        self.attempts.pop(paper_id, None)
        self.next_eligible_turn.pop(paper_id, None)

    def set_score(self, paper_id: str, score: float) -> None:
        """Set the eta value for a frontier paper."""
        if paper_id in self.papers:
            self.scores[paper_id] = score

    def unevaluated(self) -> list[str]:
        """Return frontier papers that haven't been eta-evaluated yet."""
        return [pid for pid in self.papers if pid not in self.scores]

    # ---- Claims -----------------------------------------------------------

    def claim_for(self, paper_id: str, agent: str) -> bool:
        """Claim ``paper_id`` for ``agent``. Returns True if acquired."""
        if paper_id not in self.papers:
            return False
        owner = self.claims.get(paper_id)
        if owner is None or owner == agent:
            self.claims[paper_id] = agent
            return True
        return False

    def release(self, paper_id: str, agent: str) -> None:
        """Release a claim held by ``agent`` (no-op otherwise)."""
        if self.claims.get(paper_id) == agent:
            del self.claims[paper_id]

    def release_all(self, agent: str) -> None:
        """Release every claim held by ``agent`` (exception-safe cleanup)."""
        for paper_id in [pid for pid, owner in self.claims.items() if owner == agent]:
            del self.claims[paper_id]

    def is_claimed(self, paper_id: str) -> bool:
        return paper_id in self.claims

    def claimed_by(self, paper_id: str) -> str | None:
        return self.claims.get(paper_id)

    # ---- Bounded retry ----------------------------------------------------

    def record_transient_failure(
        self, paper_id: str, agent: str, turn: int = 0
    ) -> None:
        """Release the claim, retain the candidate, and delay re-eligibility."""
        self.release(paper_id, agent)
        self.attempts[paper_id] = self.attempts.get(paper_id, 0) + 1
        self.next_eligible_turn[paper_id] = turn + self.retry_backoff

    def record_absence(self, paper_id: str) -> None:
        """Definitive absence: the candidate may be removed."""
        self.remove(paper_id)

    def attempt_count(self, paper_id: str) -> int:
        return self.attempts.get(paper_id, 0)

    def is_exhausted(self, paper_id: str) -> bool:
        return self.attempts.get(paper_id, 0) >= max(self.max_attempts, 0)

    def is_eligible(self, paper_id: str, turn: int | None = None) -> bool:
        if paper_id not in self.papers:
            return False
        if self.is_exhausted(paper_id):
            return False
        if turn is not None and self.next_eligible_turn.get(paper_id, -1) > turn:
            return False
        return not self.is_claimed(paper_id)

    def eligible(
        self, exclude: set[str] | None = None, turn: int | None = None
    ) -> list[str]:
        """Frontier papers that are neither visited, claimed, nor retry-delayed."""
        exclude = exclude or set()
        return [
            pid
            for pid in self.papers
            if pid not in exclude and self.is_eligible(pid, turn)
        ]

    def best(self, exclude: set[str] | None = None, turn: int | None = None) -> str | None:
        """Return the highest-eta paper among eligible candidates."""
        eligible = self.eligible(exclude, turn)
        if not eligible:
            return None
        best_id: str | None = None
        best_score = -1.0
        for pid in eligible:
            score = self.scores.get(pid, 0.0)
            if score > best_score:
                best_score = score
                best_id = pid
        return best_id

    def __len__(self) -> int:
        return len(self.papers)
