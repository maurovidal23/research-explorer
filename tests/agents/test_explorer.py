"""Tests for the explorer agent's reference extraction (no network, no LLM)."""

from __future__ import annotations

from research_explorer.agents.explorer import ExplorerAgent, _parse_json_response
from research_explorer.graph.models import PaperSummary


def _explorer(providers: dict[str, object] | None = None) -> ExplorerAgent:
    e = ExplorerAgent.__new__(ExplorerAgent)
    e.provider = type("P", (), {"name": "semantic_scholar"})()
    e.providers = providers or {"semantic_scholar": object(), "arxiv": object(), "openalex": object()}
    return e


def test_parse_json_response_plain() -> None:
    assert _parse_json_response('{"narrative": "x", "references": []}') == {
        "narrative": "x",
        "references": [],
    }


def test_parse_json_response_fenced() -> None:
    out = _parse_json_response('```json\n{"narrative": "y", "references": []}\n```')
    assert out == {"narrative": "y", "references": []}


def test_parse_json_response_with_surrounding_text() -> None:
    out = _parse_json_response('Here you go: {"narrative": "z", "references": []} done')
    assert out == {"narrative": "z", "references": []}


def test_parse_json_response_invalid() -> None:
    assert _parse_json_response("not json") is None


def test_parse_extracted_refs_arxiv_id() -> None:
    e = _explorer()
    refs = e._parse_extracted_refs(
        [{"title": "Attention is all you need", "authors": ["Vaswani", "Shazeer"],
          "year": 2017, "arxiv_id": "1706.03762", "doi": "10.1/x"}]
    )
    assert len(refs) == 1
    r = refs[0]
    assert r.id == "1706.03762"
    assert r.provider == "arxiv"
    assert r.doi == "10.1/x"
    assert r.year == 2017
    assert r.authors == ["Vaswani", "Shazeer"]


def test_parse_extracted_refs_doi_only_uses_s2() -> None:
    e = _explorer()
    refs = e._parse_extracted_refs(
        [{"title": "A journal paper", "authors": "Smith, Jones", "year": "2020", "doi": "10.1/abc"}]
    )
    assert len(refs) == 1
    assert refs[0].provider == "semantic_scholar"
    assert refs[0].id == "10.1/abc"
    assert refs[0].authors == ["Smith", "Jones"]
    assert refs[0].year == 2020


def test_parse_extracted_refs_doi_only_falls_back_to_openalex() -> None:
    e = _explorer(providers={"openalex": object(), "arxiv": object()})
    refs = e._parse_extracted_refs([{"title": "X", "doi": "10.1/y"}])
    assert refs[0].provider == "openalex"


def test_parse_extracted_refs_no_id_is_unknown() -> None:
    e = _explorer()
    refs = e._parse_extracted_refs([{"title": "Some book", "authors": ["Author"]}, {"title": ""}])
    assert len(refs) == 1
    assert refs[0].provider == "unknown"
    assert refs[0].id == "Some book"


def test_parse_extracted_refs_skips_non_dict_and_empty_title() -> None:
    e = _explorer()
    refs = e._parse_extracted_refs(["not a dict", {"title": ""}, {"title": "ok", "arxiv_id": "1.2"}])
    assert len(refs) == 1
    assert refs[0].id == "1.2"


def test_neighbor_summaries_prefers_native_refs() -> None:
    from research_explorer.graph.models import Paper
    e = _explorer()
    paper = Paper(
        id="p1", title="t", provider="semantic_scholar",
        references=[PaperSummary(id="r1", title="ref", provider="semantic_scholar")],
        citations=[PaperSummary(id="c1", title="cit", provider="semantic_scholar")],
    )
    refs, cits = e._neighbor_summaries(paper, [PaperSummary(id="x", title="extracted", provider="arxiv")])
    assert len(refs) == 1 and refs[0].id == "r1"
    assert len(cits) == 1 and cits[0].id == "c1"


def test_neighbor_summaries_falls_back_to_extracted() -> None:
    from research_explorer.graph.models import Paper
    e = _explorer()
    paper = Paper(id="p1", title="t", provider="arxiv")
    extracted = [PaperSummary(id="r1", title="ref", provider="arxiv")]
    refs, cits = e._neighbor_summaries(paper, extracted)
    assert refs == extracted
    assert cits == []
