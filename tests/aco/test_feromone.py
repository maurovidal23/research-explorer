"""Tests for the per-agent pheromone manager."""

from __future__ import annotations

from research_explorer.agents.state import AgentState
from research_explorer.graph.feromone import AgentPath, PheromoneManager


def _state(pid: str = "a1", pos: str = "s2:seed") -> AgentState:
    return AgentState(id=pid, pos=pos)


def test_evaporate() -> None:
    pm = PheromoneManager()
    s = _state()
    s.set_pheromone("a", "b", "ref", 10.0)
    pm.update([], None, [s], rho=0.1, lambda_elite=0.5, tau_min=0.1, tau_max=10.0)
    assert abs(s.get_pheromone("a", "b", "ref") - 9.0) < 0.01


def test_deposit() -> None:
    pm = PheromoneManager()
    s = _state()
    path = AgentPath(
        edges=[("a", "b", "ref"), ("b", "c", "ref")], delta_q=0.5, state=s
    )
    pm.update([path], None, [s], rho=0.0, lambda_elite=0.5, tau_min=0.1, tau_max=10.0)
    # deposit = 0.5 / 2 = 0.25 per edge
    assert abs(s.get_pheromone("a", "b", "ref") - 1.25) < 0.01
    assert abs(s.get_pheromone("b", "c", "ref") - 1.25) < 0.01


def test_elitism() -> None:
    pm = PheromoneManager()
    s = _state()
    path = AgentPath(edges=[("a", "b", "ref")], delta_q=0.5, state=s)
    pm.update([path], path, [s], rho=0.0, lambda_elite=0.5, tau_min=0.1, tau_max=10.0)
    # deposit = 0.5/1 = 0.5 (agent) + 0.5*0.5 = 0.25 (elitism) = 0.75 total
    assert abs(s.get_pheromone("a", "b", "ref") - 1.75) < 0.01


def test_clip() -> None:
    pm = PheromoneManager()
    s = _state()
    s.set_pheromone("a", "b", "ref", 100.0)
    pm.update([], None, [s], rho=0.0, lambda_elite=0.5, tau_min=0.1, tau_max=10.0)
    assert s.get_pheromone("a", "b", "ref") == 10.0


def test_no_deposit_on_negative_delta() -> None:
    pm = PheromoneManager()
    s = _state()
    path = AgentPath(edges=[("a", "b", "ref")], delta_q=-0.5, state=s)
    pm.update([path], None, [s], rho=0.0, lambda_elite=0.5, tau_min=0.1, tau_max=10.0)
    # delta_q < 0 -> no deposit, default 1.0
    assert s.get_pheromone("a", "b", "ref") == 1.0


def test_pheromone_is_per_agent_private() -> None:
    pm = PheromoneManager()
    s1, s2 = _state("a1"), _state("a2")
    path = AgentPath(edges=[("a", "b", "ref")], delta_q=1.0, state=s1)
    pm.update([path], path, [s1, s2], rho=0.0, lambda_elite=0.0, tau_min=0.1, tau_max=10.0)
    # s1 deposited on its own edge; s2 is untouched (private)
    assert s1.get_pheromone("a", "b", "ref") > 1.0
    assert s2.get_pheromone("a", "b", "ref") == 1.0  # default, not shared


def test_evaporation_isolated_per_agent() -> None:
    pm = PheromoneManager()
    s1, s2 = _state("a1"), _state("a2")
    s1.set_pheromone("a", "b", "ref", 10.0)
    s2.set_pheromone("a", "b", "ref", 10.0)
    # evaporate only s1 by passing [s1]
    pm.update([], None, [s1], rho=0.5, lambda_elite=0.5, tau_min=0.1, tau_max=10.0)
    assert abs(s1.get_pheromone("a", "b", "ref") - 5.0) < 0.01
    assert abs(s2.get_pheromone("a", "b", "ref") - 10.0) < 0.01
