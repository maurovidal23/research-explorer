"""Tests for the GraphStore."""

from __future__ import annotations

import tempfile

import pytest

from research_explorer.graph.models import Paper, PaperSummary
from research_explorer.graph.store import GraphStore


@pytest.fixture
def store() -> GraphStore:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    s = GraphStore(path)
    yield s
    s.close()


@pytest.fixture
def sample_paper() -> Paper:
    return Paper(
        id="da82f8e6",
        doi="10.1038/nrn3241",
        title="Test paper",
        year=2020,
        authors=["Author A"],
        citation_count=42,
        abstract="An abstract",
        provider="semantic_scholar",
        references=[
            PaperSummary(id="ref1", title="Ref 1", provider="semantic_scholar"),
        ],
        citations=[
            PaperSummary(id="cit1", title="Cit 1", provider="semantic_scholar"),
        ],
        external_ids={"DOI": "10.1038/nrn3241"},
        fields_of_study=["Neuroscience"],
    )


def test_cache_and_get_paper(store: GraphStore, sample_paper: Paper) -> None:
    store.cache_paper(sample_paper)
    nid = "s2:da82f8e6"
    paper = store.get_paper(nid)
    assert paper is not None
    assert paper.title == "Test paper"
    assert paper.year == 2020
    assert paper.doi == "10.1038/nrn3241"


def test_get_references(store: GraphStore, sample_paper: Paper) -> None:
    store.cache_paper(sample_paper)
    refs = store.get_references("s2:da82f8e6")
    assert refs == ["s2:ref1"]


def test_get_citants(store: GraphStore, sample_paper: Paper) -> None:
    store.cache_paper(sample_paper)
    cits = store.get_citants("s2:da82f8e6")
    assert cits == ["s2:cit1"]


def test_get_neighbors(store: GraphStore, sample_paper: Paper) -> None:
    store.cache_paper(sample_paper)
    refs, cits = store.get_neighbors("s2:da82f8e6")
    assert refs == ["s2:ref1"]
    assert cits == ["s2:cit1"]


def test_pheromone_default(store: GraphStore) -> None:
    assert store.get_pheromone("a", "b", "ref") == 1.0


def test_pheromone_set_and_get(store: GraphStore) -> None:
    store.set_pheromone("a", "b", "ref", 5.0)
    assert store.get_pheromone("a", "b", "ref") == 5.0


def test_pheromone_evaporate(store: GraphStore) -> None:
    store.set_pheromone("a", "b", "ref", 10.0)
    store.evaporate_all(0.1, tau_min=0.1)
    assert abs(store.get_pheromone("a", "b", "ref") - 9.0) < 0.01


def test_pheromone_clip(store: GraphStore) -> None:
    store.set_pheromone("a", "b", "ref", 100.0)
    store.set_pheromone("c", "d", "ref", 0.01)
    store.clip_pheromone(0.1, 10.0)
    assert store.get_pheromone("a", "b", "ref") == 10.0
    assert store.get_pheromone("c", "d", "ref") == 0.1


def test_embedding_cache(store: GraphStore) -> None:
    assert store.get_cached_embedding("hello") is None
    store.cache_embedding("hello", [0.1, 0.2, 0.3])
    result = store.get_cached_embedding("hello")
    assert result is not None
    assert abs(result[0] - 0.1) < 0.001


def test_pheromone_concentration(store: GraphStore) -> None:
    store.set_pheromone("a", "b", "ref", 10.0)
    store.set_pheromone("c", "d", "ref", 2.0)
    conc = store.pheromone_concentration()
    assert abs(conc - 10.0 / 6.0) < 0.01  # max=10, avg=(10+2)/2=6
