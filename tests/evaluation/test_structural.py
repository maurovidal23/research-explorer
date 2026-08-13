"""Tests for structural metrics (R)."""

from __future__ import annotations

import tempfile

import pytest

from research_explorer.agents.state import AgentState
from research_explorer.evaluation.structural import StructuralMetrics
from research_explorer.graph.models import Paper
from research_explorer.graph.store import GraphStore


@pytest.fixture
def store() -> GraphStore:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    s = GraphStore(path)
    yield s
    s.close()


def test_coverage_empty(store: GraphStore) -> None:
    state = AgentState(id="a1", pos="s2:seed")
    m = StructuralMetrics(store, target_size=10)
    assert m._coverage(state) == 0.0


def test_coverage_partial(store: GraphStore) -> None:
    state = AgentState(id="a1", pos="s2:seed", visited=["s2:a"] * 5)
    m = StructuralMetrics(store, target_size=10)
    assert m._coverage(state) == 0.5


def test_coverage_full(store: GraphStore) -> None:
    state = AgentState(id="a1", pos="s2:seed", visited=["s2:a"] * 20)
    m = StructuralMetrics(store, target_size=10)
    assert m._coverage(state) == 1.0


def test_diversity_single_paper(store: GraphStore) -> None:
    state = AgentState(id="a1", pos="s2:seed", visited=["s2:a"])
    m = StructuralMetrics(store)
    assert m._diversity(state) == 1.0


def test_depth_no_citations(store: GraphStore) -> None:
    state = AgentState(id="a1", pos="s2:seed", visited=["s2:a"])
    m = StructuralMetrics(store)
    assert m._depth(state) == 0.0


def test_depth_with_citations(store: GraphStore) -> None:
    store.cache_paper(
        Paper(
            id="a",
            title="Test",
            provider="semantic_scholar",
            citation_count=100,
        )
    )
    state = AgentState(id="a1", pos="s2:seed", visited=["s2:a"])
    m = StructuralMetrics(store)
    d = m._depth(state)
    assert 0.0 < d < 1.0


def test_coherence_single(store: GraphStore) -> None:
    state = AgentState(id="a1", pos="s2:seed", visited=["s2:seed"])
    m = StructuralMetrics(store)
    assert m._coherence(state) == 1.0


def test_compute_empty(store: GraphStore) -> None:
    state = AgentState(id="a1", pos="s2:seed")
    m = StructuralMetrics(store)
    assert m.compute(state) == 0.0
