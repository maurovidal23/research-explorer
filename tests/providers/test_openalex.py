"""Tests for the OpenAlex provider (parsing logic, no network)."""

from __future__ import annotations

from research_explorer.providers.openalex import OpenAlexProvider


def test_format_id_auto_doi() -> None:
    p = OpenAlexProvider.__new__(OpenAlexProvider)
    assert p._format_id("10.1038/nrn3241", "auto") == "doi:10.1038/nrn3241"


def test_format_id_auto_openalex() -> None:
    p = OpenAlexProvider.__new__(OpenAlexProvider)
    assert p._format_id("W1973275441", "auto") == "W1973275441"


def test_reconstruct_abstract() -> None:
    p = OpenAlexProvider.__new__(OpenAlexProvider)
    inverted = {"The": [0], "quick": [1], "brown": [2], "fox": [3]}
    assert p._reconstruct_abstract(inverted) == "The quick brown fox"


def test_reconstruct_abstract_none() -> None:
    p = OpenAlexProvider.__new__(OpenAlexProvider)
    assert p._reconstruct_abstract(None) is None


def test_parse_summary() -> None:
    p = OpenAlexProvider.__new__(OpenAlexProvider)
    raw = {
        "id": "https://openalex.org/W123",
        "title": "Test",
        "publication_year": 2020,
        "authorships": [
            {"author": {"display_name": "Author A"}},
            {"author": {"display_name": "Author B"}},
        ],
        "cited_by_count": 10,
        "ids": {"doi": "https://doi.org/10.1/test"},
        "abstract_inverted_index": {"Hello": [0], "world": [1]},
    }
    s = p._parse_summary(raw)
    assert s.id == "W123"
    assert s.doi == "10.1/test"
    assert s.title == "Test"
    assert s.year == 2020
    assert s.authors == ["Author A", "Author B"]
    assert s.citation_count == 10
    assert s.abstract == "Hello world"
    assert s.provider == "openalex"


def test_parse_paper_with_references() -> None:
    p = OpenAlexProvider.__new__(OpenAlexProvider)
    raw = {
        "id": "https://openalex.org/W123",
        "title": "Main",
        "publication_year": 2020,
        "authorships": [],
        "cited_by_count": 5,
        "ids": {"doi": "https://doi.org/10.1/main"},
        "referenced_works": [
            "https://openalex.org/W111",
            "https://openalex.org/W222",
        ],
        "topics": [{"display_name": "Neuroscience"}],
    }
    paper = p._parse_paper(raw)
    assert paper.id == "W123"
    assert paper.doi == "10.1/main"
    assert len(paper.references) == 2
    assert paper.references[0].id == "W111"
    assert paper.references[1].id == "W222"
    assert paper.references[0].provider == "openalex"
    assert paper.fields_of_study == ["Neuroscience"]


def test_parse_paper_null_fields() -> None:
    p = OpenAlexProvider.__new__(OpenAlexProvider)
    raw = {
        "id": "https://openalex.org/W123",
        "title": "Main",
        "publication_year": 2020,
        "authorships": None,
        "cited_by_count": 5,
        "ids": None,
        "referenced_works": None,
        "topics": None,
    }
    paper = p._parse_paper(raw)
    assert paper.id == "W123"
    assert paper.references == []
    assert paper.authors == []
    assert paper.fields_of_study == []
