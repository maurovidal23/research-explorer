"""Tests for the explorer agent's reference extraction (no network, no LLM)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from research_explorer.agents.explorer import ExplorerAgent, _parse_json_response, _titles_match
from research_explorer.agents.state import AgentState
from research_explorer.graph.models import Paper, PaperSummary
from research_explorer.providers.base import TransientProviderError


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
        eligible=lambda exclude=None, turn=None: ["arxiv:1901.03228"],
        sources={"arxiv:1901.03228": ("arxiv:seed", "ref")},
        remove=removed.append,
        record_absence=removed.append,
        unevaluated=lambda: [],
        set_score=lambda nid, s: None,
        claim_for=lambda nid, agent: True,
        release=lambda nid, agent: None,
        release_all=lambda agent: None,
        is_claimed=lambda nid: False,
        claims={},
    )
    e.llm = None
    e.embedding = None
    e.cfg = SimpleNamespace(
        aco=SimpleNamespace(k_per_turn=1, alpha=1.0, beta=3.0, epsilon=0.1),
        llm=SimpleNamespace(explorer_model="test", temperature=0.3, max_tokens=2000),
        heuristica=SimpleNamespace(
            w_sim=0.5, w_citas=0.3, w_recencia=0.2, w_confidence=0.1, w_llm=0.0, eta_llm=False
        ),
        direction=SimpleNamespace(ref_weight=0.7, cites_weight=0.3),
    )
    import random as _random
    e.rng = _random.Random(0)
    e._eta_cache = {}
    e._llm_priority = {}
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


def _transit_explorer(expander) -> ExplorerAgent:
    from research_explorer.aco.frontier import SharedFrontier

    e = ExplorerAgent.__new__(ExplorerAgent)
    e.expander = expander
    e.state = AgentState(id="test-agent", pos="openalex:seed", budget=3, turn_count=1)
    e._shared_visited = {"openalex:Wmeta"}
    e._frontier = SharedFrontier()
    e._frontier.add(["openalex:Wmeta"], source="openalex:seed", mode="ref")
    e.tracer = SimpleNamespace(emit=lambda event_type, **payload: None)
    return e


class _StubExpander:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def expand(self, node_id, paper=None, extracted=None, tracer=None):
        from types import SimpleNamespace as Ns

        self.calls.append((node_id, paper))
        empty = Ns(node_ids=[], provider=None, fallback_used=False,
                   discovered=0, rejected=0)
        return Ns(node_id=node_id, incoming=Ns(**vars(empty), direction="incoming"),
                  outgoing=Ns(**vars(empty), direction="outgoing"))


async def test_metadata_transit_on_fetch_failure_uses_cached_summary() -> None:
    e = _transit_explorer(_StubExpander())
    e.graph = SimpleNamespace(
        get_paper_summary=lambda nid: PaperSummary(
            id="Wmeta", title="Metadata only", provider="openalex", doi="10.1/meta"
        ) if nid == "openalex:Wmeta" else None,
    )
    emitted: list[tuple] = []
    e.tracer = SimpleNamespace(emit=lambda event_type, **payload: emitted.append((event_type, payload)))

    ok = await e._metadata_transit("openalex:Wmeta", "openalex:seed", "ref")

    assert ok is True
    assert e.state.budget == 2
    assert "openalex:Wmeta" in e.state.metadata_transits
    assert e.state.full_path == []
    assert e.expander.calls[0][0] == "openalex:Wmeta"
    assert e.expander.calls[0][1] is None
    assert emitted[0][0] == "metadata_transit"
    payload = emitted[0][1]
    assert payload["paper_id"] == "openalex:Wmeta"
    assert payload["reason"] == "full_text_unavailable"
    assert payload["provider"] == "openalex"


async def test_metadata_transit_fetched_paper_without_content() -> None:
    e = _transit_explorer(_StubExpander())
    e.graph = SimpleNamespace(
        get_paper_summary=lambda nid: PaperSummary(
            id="Wmeta", title="Metadata only", provider="openalex", doi="10.1/meta"
        ) if nid == "openalex:Wmeta" else None,
    )
    emitted: list[tuple] = []
    e.tracer = SimpleNamespace(emit=lambda event_type, **payload: emitted.append((event_type, payload)))

    paper = Paper(id="Wmeta", title="Metadata only", provider="openalex",
                  doi="10.1/meta", fulltext=None, abstract=None)
    ok = await e._metadata_transit("openalex:Wmeta", "openalex:seed", "ref", paper=paper)

    assert ok is True
    assert e.state.budget == 2
    assert e.state.full_path == []
    assert e.expander.calls[0][1] is paper
    assert emitted[0][1]["reason"] == "metadata_only"
    assert emitted[0][1]["mode"] == "ref"
    assert emitted[0][1]["src"] == "openalex:seed"


async def test_metadata_transit_returns_false_without_expander_or_summary() -> None:
    e = _transit_explorer(None)
    e.graph = SimpleNamespace(get_paper_summary=lambda nid: None)

    ok = await e._metadata_transit("openalex:Wmissing", "openalex:seed", "ref")

    assert ok is False
    assert e.state.budget == 3


async def test_metadata_transit_adds_expanded_neighbors_to_frontier() -> None:
    from types import SimpleNamespace as Ns

    class RecordingExpander(_StubExpander):
        async def expand(self, node_id, paper=None, extracted=None, tracer=None):
            await super().expand(node_id, paper, extracted, tracer)
            return Ns(
                node_id=node_id,
                incoming=Ns(direction="incoming", provider="openalex",
                            fallback_used=False, discovered=1, rejected=0,
                            node_ids=["openalex:Wcit1"]),
                outgoing=Ns(direction="outgoing", provider="openalex",
                            fallback_used=False, discovered=1, rejected=0,
                            node_ids=["openalex:Wref1"]),
            )

    e = _transit_explorer(RecordingExpander())
    e.graph = SimpleNamespace(
        get_paper_summary=lambda nid: PaperSummary(
            id="Wmeta", title="Metadata only", provider="openalex", doi="10.1/meta"
        ) if nid == "openalex:Wmeta" else None,
    )

    ok = await e._metadata_transit("openalex:Wmeta", "openalex:seed", "ref")

    assert ok is True
    assert {"openalex:Wcit1", "openalex:Wref1"} <= set(e._frontier.papers)
    assert e._frontier.sources["openalex:Wcit1"] == ("openalex:Wmeta", "cites")
    assert e._frontier.sources["openalex:Wref1"] == ("openalex:Wmeta", "ref")
    refs, cits = e.state.local_neighbors("openalex:Wmeta")
    assert set(refs) == {"openalex:Wref1"}
    assert set(cits) == {"openalex:Wcit1"}
    assert "openalex:Wmeta" not in e._frontier.papers


async def test_fetch_paper_translates_unexpected_error_to_transient() -> None:
    """STAB-2/3: an unexpected provider error must not look like absence."""

    class Provider:
        name = "openalex"
        supports_fulltext = False

        async def get_paper(self, paper_id):
            raise RuntimeError("socket reset token=SENTINEL-SECRET-FETCH")

    e = ExplorerAgent.__new__(ExplorerAgent)
    e.provider = Provider()
    e.providers = {"openalex": e.provider}

    with pytest.raises(TransientProviderError) as exc_info:
        await e._fetch_paper("openalex:W1")

    assert exc_info.value.reason == "unexpected_error"
    assert exc_info.value.provider == "openalex"
    assert "SENTINEL-SECRET-FETCH" not in str(exc_info.value)


async def test_fetch_paper_returns_none_for_definitive_absence() -> None:
    class Provider:
        name = "openalex"
        supports_fulltext = False

        async def get_paper(self, paper_id):
            return None

    e = ExplorerAgent.__new__(ExplorerAgent)
    e.provider = Provider()
    e.providers = {"openalex": e.provider}

    assert await e._fetch_paper("openalex:W1") is None


async def test_take_turn_records_transient_on_unexpected_fetch_error() -> None:
    """STAB-2/3: an unexpected fetch error is retained, never treated as absence."""
    import random as _random

    class Provider:
        name = "openalex"
        supports_fulltext = False

        async def get_paper(self, paper_id):
            raise RuntimeError("socket reset")

    calls: dict[str, list] = {
        "transient": [], "absence": [], "remove": [], "release_all": [],
    }

    class Frontier:
        def __init__(self) -> None:
            self.claims: dict[str, str] = {}
            self.sources: dict[str, tuple[str, str]] = {
                "openalex:W1": ("openalex:seed", "ref")
            }

        def eligible(self, exclude=None, turn=None):
            return ["openalex:W1"]

        def claim_for(self, nid, agent):
            self.claims[nid] = agent
            return True

        def record_transient_failure(self, nid, agent, turn):
            calls["transient"].append((nid, agent, turn))
            self.claims.pop(nid, None)

        def record_absence(self, nid):
            calls["absence"].append(nid)

        def remove(self, nid):
            calls["remove"].append(nid)

        def release(self, nid, agent):
            self.claims.pop(nid, None)

        def release_all(self, agent):
            calls["release_all"].append(agent)

        def attempt_count(self, nid):
            return 1

        def is_claimed(self, nid):
            return nid in self.claims

    e = ExplorerAgent.__new__(ExplorerAgent)
    e.provider = Provider()
    e.providers = {"openalex": e.provider}
    e._frontier = Frontier()
    e.graph = SimpleNamespace(
        get_paper_summary=lambda nid: None,
        cache_paper=lambda p: None,
    )
    e.state = AgentState(id="test-agent", pos="openalex:seed", budget=5, turn_count=1)
    e.state.mark_discovered("openalex:seed")
    e.seed_embedding = None
    e.seed_query = "q"
    e._shared_visited = set()
    e._eta_cache = {}
    e._llm_priority = {}
    e.rng = _random.Random(0)
    e.embedding = None
    e.expander = None
    e.cfg = SimpleNamespace(
        aco=SimpleNamespace(k_per_turn=1, alpha=1.0, beta=3.0, epsilon=0.1),
        heuristica=SimpleNamespace(
            w_sim=0.5, w_citas=0.3, w_recencia=0.2, w_confidence=0.1,
            w_llm=0.0, eta_llm=False,
        ),
        llm=SimpleNamespace(fulltext_max_chars=1000),
        direction=SimpleNamespace(ref_weight=0.7, cites_weight=0.3),
    )
    emitted: list[tuple] = []
    e.tracer = SimpleNamespace(
        emit=lambda event_type, **payload: emitted.append((event_type, payload))
    )

    edges = await e.take_turn(1)

    assert edges == []
    assert "openalex:W1" not in e._shared_visited
    assert calls["transient"] and calls["transient"][0][:2] == ("openalex:W1", "test-agent")
    assert calls["absence"] == []
    assert calls["remove"] == []
    assert not e._frontier.is_claimed("openalex:W1")
    assert calls["release_all"] == ["test-agent"]

    failures = [payload for kind, payload in emitted if kind == "provider_failure"]
    assert failures
    assert failures[0]["paper_id"] == "openalex:W1"
    assert failures[0]["provider"] == "openalex"
    assert failures[0]["reason"] == "unexpected_error"
    assert failures[0]["classification"] == "transient"
