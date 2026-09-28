"""Convergence checker — determines when the ACO search should stop.

Criteria (any one triggers stop, configurable):
  - Budget exhausted (fetches or time)
  - Quality plateau: Q_best hasn't improved by epsilon over T oleadas
  - Pheromone concentration: max/mean ratio exceeds theta (routes consolidated)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from research_explorer.config import Config


@dataclass
class ConvergenceState:
    """Tracks convergence-related state across oleadas."""

    quality_history: list[float] = field(default_factory=list)
    start_time: float = field(default_factory=time.monotonic)


class ConvergenceChecker:
    """Checks whether the ACO search has converged or exhausted its budget."""

    def __init__(self, config: Config):
        self.cfg = config
        self.state = ConvergenceState()
        self.last_reason: str = ""

    def should_stop(
        self,
        best_quality: float,
        total_fetches: int,
        pheromone_concentration: float,
    ) -> bool:
        """Return True if any convergence criterion is met."""
        self.state.quality_history.append(best_quality)

        # 1. Budget exhausted
        if self.cfg.budget.type == "fetches" and total_fetches >= self.cfg.budget.max_fetches:
            self.last_reason = "budget_exhausted"
            return True
        if self.cfg.budget.type == "time":
            elapsed = time.monotonic() - self.state.start_time
            if elapsed >= self.cfg.budget.max_time_seconds:
                self.last_reason = "time_budget_exhausted"
                return True

        # 2. Quality plateau
        T = self.cfg.convergence.plateau_T
        eps = self.cfg.convergence.epsilon
        if len(self.state.quality_history) > T:
            recent = self.state.quality_history[-1]
            past = self.state.quality_history[-T]
            if recent - past < eps:
                self.last_reason = "quality_plateau"
                return True

        # 3. Pheromone concentration
        if pheromone_concentration > self.cfg.convergence.theta:
            self.last_reason = "pheromone_converged"
            return True
        self.last_reason = ""
        return False

    def reset(self) -> None:
        self.state = ConvergenceState()
        self.last_reason = ""
