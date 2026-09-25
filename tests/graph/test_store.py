"""Tests for the GraphStore."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from research_explorer.graph.models import Paper, PaperSummary
from research_explorer.graph.store import GraphStore

LEGACY_SCHEMA = """
CREATE TABLE papers (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    doi TEXT,
    title TEXT,
    year INTEGER,
    authors TEXT,
    citation_count INTEGER,
    abstract TEXT,
    embedding BLOB,
    fetched_at TEXT,
    metadata_json TEXT
);
CREATE TABLE edges (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    PRIMARY KEY (src, dst)
);
CREATE TABLE pheromone (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    mode TEXT NOT NULL,
    tau REAL DEFAULT 1.0,
    PRIMARY KEY (src, dst, mode)
);
CREATE TABLE embeddings (
    text_hash TEXT PRIMARY KEY,
    embedding BLOB,
    created_at TEXT
);
"""


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


def test_arxiv_id_roundtrip(store: GraphStore) -> None:
    paper = Paper(
        id="2301.00001",
        title="arXiv paper",
        provider="arxiv",
        arxiv_id="2301.00001",
        doi=None,
    )
    store.cache_paper(paper)
    fetched = store.get_paper("arxiv:2301.00001")
    assert fetched is not None
    assert fetched.arxiv_id == "2301.00001"
    assert fetched.doi is None
    summary = store.get_paper_summary("arxiv:2301.00001")
    assert summary is not None
    assert summary.arxiv_id == "2301.00001"


def test_legacy_db_migration_adds_columns_and_preserves_data(tmp_path: Path) -> None:
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(LEGACY_SCHEMA)
    conn.execute(
        "INSERT INTO papers (id, provider, doi, title, year, authors, citation_count,"
        " abstract, embedding, fetched_at, metadata_json)"
        " VALUES ('s2:legacy', 'semantic_scholar', '10.1/legacy', 'Legacy paper', 1999,"
        " '[]', 5, NULL, NULL, NULL, '{}')"
    )
    conn.commit()
    conn.close()

    store = GraphStore(str(db))
    try:
        paper = store.get_paper("s2:legacy")
        assert paper is not None
        assert paper.title == "Legacy paper"
        assert paper.year == 1999
        assert paper.arxiv_id is None
        assert not store.is_integrated("s2:legacy")
        store.cache_summary(
            PaperSummary(id="new", title="New", provider="s2", arxiv_id="2301.00001")
        )
        assert store.get_paper_summary("s2:new") is not None
    finally:
        store.close()

    conn = sqlite3.connect(str(db))
    cols = {row[1] for row in conn.execute("PRAGMA table_info(papers)")}
    conn.close()
    assert {"arxiv_id", "integrated"} <= cols

    reopened = GraphStore(str(db))
    try:
        assert reopened.get_paper("s2:legacy") is not None
        assert reopened.get_paper_summary("s2:new") is not None
    finally:
        reopened.close()


def test_mark_integrated_flags_existing_paper(store: GraphStore, sample_paper: Paper) -> None:
    store.cache_paper(sample_paper)
    assert not store.is_integrated("s2:da82f8e6")
    store.mark_integrated("s2:da82f8e6")
    assert store.is_integrated("s2:da82f8e6")
    assert not store.is_integrated("s2:ref1")


def test_recache_paper_preserves_integrated_flag(store: GraphStore, sample_paper: Paper) -> None:
    store.cache_paper(sample_paper)
    store.mark_integrated("s2:da82f8e6")
    store.cache_paper(sample_paper)
    assert store.is_integrated("s2:da82f8e6")


def test_mark_integrated_creates_stub_row(store: GraphStore) -> None:
    store.mark_integrated("openalex:W1")
    assert store.is_integrated("openalex:W1")
    summary = store.get_paper_summary("openalex:W1")
    assert summary is not None
    assert summary.title == ""
    assert summary.provider == "unknown"


def test_add_alias_first_mapping_wins(store: GraphStore) -> None:
    store.add_alias("doi:10.1/x", "s2:a")
    store.add_alias("doi:10.1/x", "s2:b")
    assert store.get_canonical_id("doi:10.1/x") == "s2:a"
    assert store.get_aliases("s2:a") == ["doi:10.1/x"]
    assert store.get_aliases("s2:b") == []


def test_add_alias_ignores_self_and_empty(store: GraphStore) -> None:
    store.add_alias("s2:a", "s2:a")
    store.add_alias("", "s2:a")
    store.add_alias("doi:10.1/x", "")
    assert store.get_canonical_id("s2:a") is None
    assert store.get_canonical_id("") is None


def test_canonical_id_follows_alias_chain(store: GraphStore) -> None:
    store.add_alias("alias_a", "alias_b")
    store.add_alias("alias_b", "s2:root")
    assert store.canonical_id("alias_a") == "s2:root"
    assert store.canonical_id("alias_b") == "s2:root"
    assert store.canonical_id("s2:root") == "s2:root"
    assert store.canonical_id("unmapped") == "unmapped"


def test_alias_persistence_across_reopen(tmp_path: Path) -> None:
    db = str(tmp_path / "aliases.db")
    first = GraphStore(db)
    try:
        first.add_alias("doi:10.1/x", "s2:abc")
        first.add_alias("arxiv:2301.00001", "s2:abc")
    finally:
        first.close()

    second = GraphStore(db)
    try:
        assert second.get_canonical_id("doi:10.1/x") == "s2:abc"
        assert second.get_canonical_id("arxiv:2301.00001") == "s2:abc"
        assert second.get_aliases("s2:abc") == ["arxiv:2301.00001", "doi:10.1/x"]
    finally:
        second.close()


def test_cache_paper_records_edge_provenance(store: GraphStore, sample_paper: Paper) -> None:
    store.cache_paper(sample_paper)

    outgoing = store.get_edge_provenance(src="s2:da82f8e6")
    assert len(outgoing) == 1
    row = outgoing[0]
    assert row["src"] == "s2:da82f8e6"
    assert row["dst"] == "s2:ref1"
    assert row["direction"] == "references"
    assert row["provider"] == "semantic_scholar"
    assert row["created_at"]

    incoming = store.get_edge_provenance(dst="s2:da82f8e6")
    assert len(incoming) == 1
    assert incoming[0]["src"] == "s2:cit1"
    assert incoming[0]["direction"] == "cited_by"


def test_edge_provenance_supports_both_directions(store: GraphStore) -> None:
    store.record_edge("a", "b", "openalex", "references")
    store.record_edge("a", "b", "openalex", "cited_by")
    store.commit()

    rows = store.get_edge_provenance()
    assert [(r["direction"], r["provider"]) for r in rows] == [
        ("cited_by", "openalex"),
        ("references", "openalex"),
    ]
    assert store.get_references("a") == ["b"]


def test_record_edge_is_idempotent(store: GraphStore) -> None:
    store.record_edge("a", "b", "openalex", "references")
    store.record_edge("a", "b", "openalex", "references")
    store.commit()
    assert len(store.get_edge_provenance()) == 1
    assert store.get_references("a") == ["b"]


def test_edge_provenance_filtering(store: GraphStore) -> None:
    store.record_edge("a", "b", "openalex", "references")
    store.record_edge("c", "b", "s2", "cited_by")
    store.commit()

    assert [r["src"] for r in store.get_edge_provenance(dst="b")] == ["a", "c"]
    assert [r["dst"] for r in store.get_edge_provenance(src="c")] == ["b"]
    assert [r["provider"] for r in store.get_edge_provenance(src="a", dst="b")] == ["openalex"]
    assert store.get_edge_provenance(src="missing") == []
