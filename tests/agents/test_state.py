"""Tests for AgentState."""

from research_explorer.agents.state import AgentState


def test_state_defaults() -> None:
    s = AgentState(id="a1", pos="s2:seed")
    assert s.visited == []
    assert s.frontier == []
    assert s.narrative == ""
    assert s.budget == 0
    assert s.quality == 0.0
    assert s.caste == "mixto"
    assert s.path == []


def test_start_turn_resets() -> None:
    s = AgentState(id="a1", pos="s2:seed", path=[("x", "ref")], delta_q=0.5)
    s.start_turn()
    assert s.path == []
    assert s.delta_q == 0.0


def test_visit_updates_state() -> None:
    s = AgentState(id="a1", pos="s2:seed", budget=10)
    s.visit("s2:paper1", "ref")
    assert s.pos == "s2:paper1"
    assert "s2:paper1" in s.visited
    assert s.budget == 9
    assert s.path == [("s2:paper1", "ref")]


def test_frontier_management() -> None:
    s = AgentState(id="a1", pos="s2:seed", visited=["s2:seed"])
    s.add_to_frontier(["s2:a", "s2:b", "s2:seed"])
    assert s.frontier == ["s2:a", "s2:b"]  # seed already visited
    s.add_to_frontier(["s2:a"])  # dedup
    assert s.frontier == ["s2:a", "s2:b"]
    s.remove_from_frontier("s2:a")
    assert s.frontier == ["s2:b"]


def test_is_exhausted() -> None:
    s = AgentState(id="a1", pos="s2:seed", budget=0)
    assert s.is_exhausted()
    s.budget = 5
    assert not s.is_exhausted()


# ---- Private graph ----------------------------------------------------------


def test_local_neighbors_set_and_get() -> None:
    s = AgentState(id="a1", pos="s2:seed")
    assert s.local_references("s2:p") == []
    assert s.local_citants("s2:p") == []
    s.set_local_neighbors("s2:p", ["s2:r1", "s2:r2"], ["s2:c1"])
    assert s.local_references("s2:p") == ["s2:r1", "s2:r2"]
    assert s.local_citants("s2:p") == ["s2:c1"]
    assert s.local_neighbors("s2:p") == (["s2:r1", "s2:r2"], ["s2:c1"])
    assert s.is_discovered("s2:p")
    assert not s.is_discovered("s2:other")


def test_mark_discovered_idempotent() -> None:
    s = AgentState(id="a1", pos="s2:seed")
    s.mark_discovered("s2:p")
    s.mark_discovered("s2:p")
    assert s.discovered == ["s2:p"]


# ---- Private pheromone ------------------------------------------------------


def test_pheromone_default_is_one() -> None:
    s = AgentState(id="a1", pos="s2:seed")
    assert s.get_pheromone("a", "b", "ref") == 1.0


def test_pheromone_set_and_get() -> None:
    s = AgentState(id="a1", pos="s2:seed")
    s.set_pheromone("a", "b", "ref", 5.0)
    assert s.get_pheromone("a", "b", "ref") == 5.0
    # different mode is independent
    assert s.get_pheromone("a", "b", "cites") == 1.0


def test_pheromone_evaporate() -> None:
    s = AgentState(id="a1", pos="s2:seed")
    s.set_pheromone("a", "b", "ref", 10.0)
    s.evaporate_pheromone(0.1, tau_min=0.1)
    assert abs(s.get_pheromone("a", "b", "ref") - 9.0) < 0.01


def test_pheromone_evaporate_clipped_to_min() -> None:
    s = AgentState(id="a1", pos="s2:seed")
    s.set_pheromone("a", "b", "ref", 0.5)
    s.evaporate_pheromone(0.9, tau_min=0.1)
    assert s.get_pheromone("a", "b", "ref") == 0.1


def test_pheromone_clip() -> None:
    s = AgentState(id="a1", pos="s2:seed")
    s.set_pheromone("a", "b", "ref", 100.0)
    s.set_pheromone("c", "d", "ref", 0.01)
    s.clip_pheromone(0.1, 10.0)
    assert s.get_pheromone("a", "b", "ref") == 10.0
    assert s.get_pheromone("c", "d", "ref") == 0.1


def test_pheromone_concentration() -> None:
    s = AgentState(id="a1", pos="s2:seed")
    assert s.pheromone_concentration() == 0.0
    s.set_pheromone("a", "b", "ref", 10.0)
    s.set_pheromone("c", "d", "ref", 2.0)
    assert abs(s.pheromone_concentration() - (10.0 / 6.0)) < 0.01


def test_pheromone_is_private_per_agent() -> None:
    s1 = AgentState(id="a1", pos="s2:seed")
    s2 = AgentState(id="a2", pos="s2:seed")
    s1.set_pheromone("a", "b", "ref", 5.0)
    assert s2.get_pheromone("a", "b", "ref") == 1.0  # not shared
