"""Tests for the explorer agent's reference extraction (no network, no LLM)."""

from __future__ import annotations

from types import SimpleNamespace

from research_explorer.agents.explorer import ExplorerAgent, _parse_json_response, _titles_match
from research_explorer.agents.state import AgentState
from research_explorer.graph.models import Paper, PaperSummary


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


def test_titles_match_identical() -> None:
    assert _titles_match("Attention Is All You Need", "Attention Is All You Need")


def test_titles_match_close_punctuation() -> None:
    assert _titles_match(
        "A Cosmic Battery in accretion flows around astrophysical black holes",
        "A Cosmic Battery in Accretion Flows Around Astrophysical Black Holes",
    )


def test_titles_match_minor_rewording() -> None:
    assert _titles_match(
        "Deep Residual Learning for Image Recognition",
        "Deep Residual Learning for Image Recognition",
    )


def test_titles_match_partial_overlap() -> None:
    assert _titles_match(
        "On the Origin of Species by Means of Natural Selection",
        "On the Origin of Species",
    )


def test_titles_reject_unrelated() -> None:
    assert not _titles_match(
        "Counterfactual Explanations for Machine Learning",
        "A Cosmic Battery in Accretion Flows Around Black Holes",
    )


def test_titles_reject_completely_different() -> None:
    assert not _titles_match(
        "Diverse Feasible Counterfactual Explanations",
        "Quantum Entanglement in Topological Insulators",
    )


def test_titles_match_empty_expected_passes() -> None:
    assert _titles_match("", "Any title at all")


def test_titles_match_empty_actual_passes() -> None:
    assert _titles_match("Some Title", "")


def test_titles_match_both_empty() -> None:
    assert _titles_match("", "")


async def test_take_turn_rejects_mismatched_title() -> None:
    e = ExplorerAgent.__new__(ExplorerAgent)
    fetched_paper = Paper(
        id="1901.03228", title="Completely Unrelated Paper",
        year=2019, provider="arxiv",
    )

    class Provider:
        name = "arxiv"
        supports_fulltext = False

        async def get_paper(self, paper_id):
            return fetched_paper

    e.provider = Provider()
    e.providers = {"arxiv": e.provider}
    cached = []
    graph_mock = SimpleNamespace(
        get_paper_summary=lambda nid: PaperSummary(
            id="1901.03228", title="Expected Paper Title",
            provider="arxiv",
        ) if nid == "arxiv:1901.03228" else None,
        get_paper=lambda nid: None,
        cache_paper=cached.append,
        cache_summary=lambda s: None,
    )

    e.graph = graph_mock
    e.state = AgentState(id="test-agent", pos="arxiv:seed", budget=5, turn_count=1)
    e.seed_query = "test query"
    e.seed_embedding = None
    e._shared_visited = set()
    removed = []
    emitted = []
    e._frontier = SimpleNamespace(
        best=lambda exclude=None: "arxiv:1901.03228" if not exclude or "arxiv:1901.03228" not in exclude else None,
        sources={"arxiv:1901.03228": ("arxiv:seed", "ref")},
        remove=removed.append,
        unevaluated=lambda: [],
        set_score=lambda nid, s: None,
    )
    e.llm = None
    e.embedding = None
    e.cfg = SimpleNamespace(
        aco=SimpleNamespace(k_per_turn=1),
        llm=SimpleNamespace(explorer_model="test", temperature=0.3, max_tokens=2000),
        heuristica=SimpleNamespace(w_sim=0.5, w_citas=0.3, w_recencia=0.2),
    )
    e.tracer = SimpleNamespace(emit=lambda event_type, **payload: emitted.append((event_type, payload)))
    e._frontier.add = lambda *a, **kw: None
    e._frontier.best_frontier = lambda exclude=None: None

    async def _must_not_discover(*a, **kw):
        raise RuntimeError("_discover must not be called")

    async def _must_not_eval():
        raise RuntimeError("_eval must not be called")

    e._discover_neighbors = _must_not_discover
    e._evaluate_new_refs = _must_not_eval

    async def _noop_integrate(p):
        return p.title, []

    e._integrate = _noop_integrate
    e.state.mark_discovered("arxiv:seed")

    edges = await e.take_turn(1)
    assert edges == []
    assert e.state.narrative == ""
    assert e.state.budget == 5
    assert e._shared_visited == set()
    assert cached == []
    assert removed == ["arxiv:1901.03228"]
    assert emitted == [
        (
            "id_title_mismatch",
            {
                "paper_id": "arxiv:1901.03228",
                "expected_title": "Expected Paper Title",
                "actual_title": "Completely Unrelated Paper",
                "reason": "id_title_mismatch",
            },
        )
    ]
