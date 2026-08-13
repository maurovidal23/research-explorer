"""Pheromone update logic for the ACO engine.

Per-agent (private) pheromone: each agent keeps its own pheromone trail in its
AgentState. Edges are not shared, so pheromone is not shared either — agents
self-reinforce their own discovered trails. Evaporation, deposit, elitism,
and MMAS clipping all operate on each agent's private pheromone map.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from research_explorer.agents.state import AgentState


@dataclass
class AgentPath:
    """An agent's path for a turn: list of (src, dst, mode) edges traversed."""

    edges: list[tuple[str, str, str]]  # (src, dst, mode)
    delta_q: float  # quality improvement this turn
    state: AgentState = field(default=None)  # type: ignore[assignment]


class PheromoneManager:
    """Manages per-agent pheromone updates on each agent's private trail."""

    def update(
        self,
        agent_paths: list[AgentPath],
        best_path: AgentPath | None,
        all_states: list[AgentState],
        rho: float,
        lambda_elite: float,
        tau_min: float,
        tau_max: float,
    ) -> None:
        """Full ACO pheromone update cycle on per-agent private pheromone.

        Args:
            agent_paths: Active agents' paths and delta_q for this oleada.
            best_path: The best agent's path (for elitism deposit).
            all_states: All agents' states (for evaporation + clipping).
            rho: Evaporation rate in [0, 1).
            lambda_elite: Elitism deposit weight.
            tau_min, tau_max: MMAS clipping bounds.
        """
        # 1. Evaporation over every agent's private pheromone
        for s in all_states:
            s.evaporate_pheromone(rho, tau_min)

        # 2. Per-agent deposit: ΔQ+ / |path| on the agent's own edges
        for ap in agent_paths:
            delta = max(0.0, ap.delta_q)
            path_len = max(1, len(ap.edges))
            deposit = delta / path_len
            if deposit <= 0 or ap.state is None:
                continue
            for (src, dst, mode) in ap.edges:
                old = ap.state.get_pheromone(src, dst, mode)
                ap.state.set_pheromone(src, dst, mode, old + deposit)

        # 3. Elitism: extra deposit on the best agent's own path
        if best_path is not None and best_path.state is not None and best_path.delta_q > 0:
            for (src, dst, mode) in best_path.edges:
                old = best_path.state.get_pheromone(src, dst, mode)
                best_path.state.set_pheromone(
                    src, dst, mode, old + lambda_elite * best_path.delta_q
                )

        # 4. MMAS clipping on every agent's private pheromone
        for s in all_states:
            s.clip_pheromone(tau_min, tau_max)
