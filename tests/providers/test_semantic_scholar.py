"""Tests for the Semantic Scholar provider (parsing logic, no network)."""

from __future__ import annotations

from research_explorer.providers.semantic_scholar import SemanticScholarProvider


def test_format_id_auto_doi() -> None:
    p = SemanticScholarProvider.__new__(SemanticScholarProvider)
    assert p._format_id("10.1038/nrn3241", "auto") == "DOI:10.1038/nrn3241"


def test_format_id_auto_pmid() -> None:
    p = SemanticScholarProvider.__new__(SemanticScholarProvider)
    assert p._format_id("22595786", "auto") == "PMID:22595786"


def test_format_id_explicit() -> None:
    p = SemanticScholarProvider.__new__(SemanticScholarProvider)
    assert p._format_id("abc123", "arXiv") == "arXiv:abc123"


def test_parse_summary() -> None:
    p = SemanticScholarProvider.__new__(SemanticScholarProvider)
    raw = {
        "paperId": "da82f8e6",
        "title": "Test paper",
        "year": 2020,
        "authors": [{"name": "Author A"}, {"name": "Author B"}],
        "citationCount": 42,
        "abstract": "An abstract",
        "externalIds": {"DOI": "10.1/test", "PubMed": "12345"},
    }
    s = p._parse_summary(raw)
    assert s.id == "da82f8e6"
    assert s.doi == "10.1/test"
    assert s.title == "Test paper"
    assert s.year == 2020
    assert s.authors == ["Author A", "Author B"]
    assert s.citation_count == 42
    assert s.provider == "semantic_scholar"


def test_parse_paper_with_refs_and_citations() -> None:
    p = SemanticScholarProvider.__new__(SemanticScholarProvider)
    raw = {
        "paperId": "p1",
        "title": "Main",
        "year": 2020,
        "authors": [],
        "citationCount": 5,
        "externalIds": {"DOI": "10.1/main"},
        "references": [
            {"paperId": "r1", "title": "Ref 1", "authors": []},
            {"paperId": "r2", "title": "Ref 2", "authors": []},
        ],
        "citations": [
            {"paperId": "c1", "title": "Cit 1", "authors": []},
        ],
        "tldr": {"text": "A short summary."},
        "fieldsOfStudy": ["Neuroscience"],
    }
    paper = p._parse_paper(raw)
    assert paper.id == "p1"
    assert len(paper.references) == 2
    assert paper.references[0].id == "r1"
    assert len(paper.citations) == 1
    assert paper.citations[0].id == "c1"
    assert paper.tldr == "A short summary."
    assert paper.fields_of_study == ["Neuroscience"]


def test_parse_paper_with_null_refs_and_citations() -> None:
    p = SemanticScholarProvider.__new__(SemanticScholarProvider)
    raw = {
        "paperId": "p1",
        "title": "Main",
        "year": 2020,
        "authors": None,
        "citationCount": 5,
        "externalIds": {"DOI": "10.1/main"},
        "references": None,
        "citations": None,
        "tldr": None,
        "fieldsOfStudy": None,
    }
    paper = p._parse_paper(raw)
    assert paper.id == "p1"
    assert paper.references == []
    assert paper.citations == []
    assert paper.authors == []
    assert paper.fields_of_study == []
    assert paper.tldr is None
