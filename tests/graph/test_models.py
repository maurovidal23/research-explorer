"""Tests for the data models."""

from research_explorer.graph.models import (
    Paper,
    PaperSummary,
    normalize_id,
    parse_normalized_id,
)


def test_paper_summary_defaults() -> None:
    p = PaperSummary(id="x", title="Test", provider="semantic_scholar")
    assert p.doi is None
    assert p.authors == []
    assert p.abstract is None


def test_paper_inherits_summary() -> None:
    p = Paper(id="x", title="Test", provider="openalex")
    assert p.references == []
    assert p.citations == []
    assert p.external_ids == {}


def test_normalize_id() -> None:
    assert normalize_id("semantic_scholar", "abc") == "s2:abc"
    assert normalize_id("openalex", "W123") == "openalex:W123"
    assert normalize_id("pubmed", "22595786") == "pmid:22595786"


def test_parse_normalized_id() -> None:
    provider, native = parse_normalized_id("s2:abc")
    assert provider == "semantic_scholar"
    assert native == "abc"

    provider, native = parse_normalized_id("pmid:22595786")
    assert provider == "pubmed"
    assert native == "22595786"
