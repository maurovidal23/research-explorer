"""STAB-5: re-caching metadata must preserve a stored embedding."""

from __future__ import annotations

import tempfile

from research_explorer.graph.models import Paper
from research_explorer.graph.store import GraphStore


def _paper(**overrides: object) -> Paper:
    values: dict[str, object] = {
        "id": "p1",
        "provider": "semantic_scholar",
        "title": "A paper",
    }
    values.update(overrides)
    return Paper(**values)  # type: ignore[arg-type]


def test_recache_without_embedding_preserves_stored() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = GraphStore(f"{tmp}/g.db")
        try:
            store.cache_paper(_paper())
            store.set_paper_embedding("s2:p1", [1.0, 2.0, 3.0])
            store.cache_paper(_paper(citation_count=42))
            assert store.get_paper_embedding("s2:p1") == [1.0, 2.0, 3.0]
        finally:
            store.close()


def test_explicit_embedding_replaces_stored() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = GraphStore(f"{tmp}/g.db")
        try:
            store.cache_paper(_paper())
            store.set_paper_embedding("s2:p1", [1.0, 2.0, 3.0])
            store.cache_paper(_paper(embedding=[9.0]))
            assert store.get_paper_embedding("s2:p1") == [9.0]
        finally:
            store.close()


def test_summary_recache_does_not_wipe_embedding() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = GraphStore(f"{tmp}/g.db")
        try:
            store.cache_paper(_paper())
            store.set_paper_embedding("s2:p1", [4.0, 5.0])
            store.cache_summary(_paper().model_copy(update={"citation_count": 7}))
            assert store.get_paper_embedding("s2:p1") == [4.0, 5.0]
        finally:
            store.close()
