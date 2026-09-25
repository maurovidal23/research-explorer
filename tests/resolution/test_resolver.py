"""Tests for the deterministic identity resolver, using mocked providers."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_explorer.graph.models import PaperSummary
from research_explorer.graph.store import GraphStore
from research_explorer.resolution.models import (
    BibliographicEntry,
    RejectReason,
    ResolutionMethod,
    ResolutionStatus,
)
from research_explorer.resolution.resolver import IdentityResolver

VASWANI_DOI = "10.5555/3295222.3295349"
VASWANI_TITLE = "Attention is all you need"


class MockProvider:
    def __init__(
        self,
        name: str,
        doi_result: PaperSummary | None | Exception = None,
        arxiv_result: PaperSummary | None | Exception = None,
        title_results: list[PaperSummary] | Exception | None = None,
    ) -> None:
        self.name = name
        self.doi_result = doi_result
        self.arxiv_result = arxiv_result
        self.title_results = title_results if title_results is not None else []
        self.doi_calls: list[str] = []
        self.arxiv_calls: list[str] = []
        self.title_calls: list[str] = []

    async def lookup_doi(self, doi: str) -> PaperSummary | None:
        self.doi_calls.append(doi)
        if isinstance(self.doi_result, Exception):
            raise self.doi_result
        return self.doi_result

    async def lookup_arxiv(self, arxiv_id: str) -> PaperSummary | None:
        self.arxiv_calls.append(arxiv_id)
        if isinstance(self.arxiv_result, Exception):
            raise self.arxiv_result
        return self.arxiv_result

    async def search_title(self, title: str) -> list[PaperSummary]:
        self.title_calls.append(title)
        if isinstance(self.title_results, Exception):
            raise self.title_results
        return list(self.title_results)


def vaswani_summary(**overrides: object) -> PaperSummary:
    values: dict[str, object] = {
        "id": "abc123",
        "doi": VASWANI_DOI,
        "title": VASWANI_TITLE,
        "year": 2017,
        "authors": ["A. Vaswani", "N. Shazeer"],
        "provider": "semantic_scholar",
    }
    values.update(overrides)
    return PaperSummary(**values)


def vaswani_entry(**overrides: object) -> BibliographicEntry:
    values: dict[str, object] = {
        "title": "Attention Is All You Need!",
        "authors": ["Vaswani"],
        "year": 2017,
        "doi": VASWANI_DOI,
    }
    values.update(overrides)
    return BibliographicEntry(**values)


@pytest.fixture
def store(tmp_path: Path) -> GraphStore:
    s = GraphStore(str(tmp_path / "graph.db"))
    yield s
    s.close()


async def test_doi_lookup_resolves_and_registers_aliases(store: GraphStore) -> None:
    provider = MockProvider("p1", doi_result=vaswani_summary())
    resolver = IdentityResolver(providers=[provider], store=store)

    result = await resolver.resolve(vaswani_entry())

    assert result.accepted
    assert result.status is ResolutionStatus.RESOLVED
    assert result.method is ResolutionMethod.DOI_LOOKUP
    assert result.provider == "p1"
    assert result.canonical_id == "s2:abc123"
    assert result.summary is not None
    assert result.actual is not None and result.actual.doi == VASWANI_DOI
    assert provider.doi_calls == [VASWANI_DOI]
    assert store.get_canonical_id(f"doi:{VASWANI_DOI}") == "s2:abc123"
    assert "doi:" + VASWANI_DOI in result.aliases


async def test_doi_miss_falls_back_to_title_search() -> None:
    provider = MockProvider(
        "p1",
        doi_result=None,
        title_results=[vaswani_summary()],
    )
    resolver = IdentityResolver(providers=[provider])

    result = await resolver.resolve(vaswani_entry())

    assert result.accepted
    assert result.method is ResolutionMethod.TITLE_SEARCH
    assert provider.doi_calls == [VASWANI_DOI]
    assert provider.title_calls == ["Attention Is All You Need!"]


async def test_arxiv_lookup_resolves() -> None:
    summary = vaswani_summary(id="2301.00001", doi=None, arxiv_id="2301.00001", provider="arxiv")
    provider = MockProvider("p1", doi_result=None, arxiv_result=summary)
    resolver = IdentityResolver(providers=[provider])

    result = await resolver.resolve(
        vaswani_entry(doi=None, arxiv_id="arXiv:2301.00001v2")
    )

    assert result.accepted
    assert result.method is ResolutionMethod.ARXIV_LOOKUP
    assert result.canonical_id == "arxiv:2301.00001"
    assert provider.arxiv_calls == ["2301.00001"]


async def test_alias_cache_hit_skips_providers(store: GraphStore) -> None:
    summary = vaswani_summary()
    store.cache_summary(summary)
    store.add_alias(f"doi:{VASWANI_DOI}", "s2:abc123")
    provider = MockProvider("p1", doi_result=vaswani_summary())
    resolver = IdentityResolver(providers=[provider], store=store)

    result = await resolver.resolve(vaswani_entry())

    assert result.accepted
    assert result.method is ResolutionMethod.ALIAS_CACHE
    assert result.canonical_id == "s2:abc123"
    assert result.confidence == 1.0
    assert provider.doi_calls == []


async def test_title_only_match_accepts_on_title_evidence() -> None:
    provider = MockProvider(
        "p1",
        title_results=[vaswani_summary(authors=[], year=None, doi=None)],
    )
    resolver = IdentityResolver(providers=[provider])

    result = await resolver.resolve(
        vaswani_entry(doi=None, authors=[], year=None)
    )

    assert result.accepted
    assert result.method is ResolutionMethod.TITLE_SEARCH
    best = result.evidence[0]
    assert best.score.title_checked
    assert not best.score.authors_checked
    assert not best.score.year_checked
    assert best.score.confidence == pytest.approx(best.score.title_similarity)


async def test_author_mismatch_rejects() -> None:
    provider = MockProvider("p1", doi_result=vaswani_summary(authors=["N. Shazeer"]))
    resolver = IdentityResolver(providers=[provider])

    result = await resolver.resolve(vaswani_entry())

    assert not result.accepted
    assert result.status is ResolutionStatus.REJECTED
    assert result.reject_reason is RejectReason.AUTHOR_MISMATCH
    assert result.evidence[0].reject_reason is RejectReason.AUTHOR_MISMATCH


async def test_year_mismatch_rejects() -> None:
    provider = MockProvider("p1", doi_result=vaswani_summary(year=2020))
    resolver = IdentityResolver(providers=[provider])

    result = await resolver.resolve(vaswani_entry())

    assert not result.accepted
    assert result.reject_reason is RejectReason.YEAR_MISMATCH


async def test_year_within_tolerance_accepts() -> None:
    provider = MockProvider("p1", doi_result=vaswani_summary(year=2019))
    resolver = IdentityResolver(providers=[provider])

    result = await resolver.resolve(vaswani_entry())

    assert result.accepted
    assert result.evidence[0].score.year_ok


async def test_title_mismatch_rejects() -> None:
    provider = MockProvider(
        "p1",
        doi_result=vaswani_summary(title="Completely unrelated research topic"),
    )
    resolver = IdentityResolver(providers=[provider])

    result = await resolver.resolve(vaswani_entry())

    assert not result.accepted
    assert result.reject_reason is RejectReason.TITLE_MISMATCH


async def test_no_candidates_rejected_with_no_provider_match() -> None:
    provider = MockProvider("p1", doi_result=None, title_results=[])
    resolver = IdentityResolver(providers=[provider])

    result = await resolver.resolve(vaswani_entry())

    assert result.status is ResolutionStatus.REJECTED
    assert result.reject_reason is RejectReason.NO_PROVIDER_MATCH
    assert result.evidence == []


async def test_provider_error_recorded_and_next_provider_used() -> None:
    failing = MockProvider("p_fail", doi_result=RuntimeError("boom"))
    working = MockProvider("p_ok", doi_result=vaswani_summary())
    resolver = IdentityResolver(providers=[failing, working])

    result = await resolver.resolve(vaswani_entry())

    assert result.accepted
    assert result.provider == "p_ok"
    unavailable = [a for a in result.attempts if a.provider == "p_fail"]
    assert unavailable[0].reject_reason is RejectReason.PROVIDER_UNAVAILABLE
    assert unavailable[0].error == "boom"


async def test_invalid_identifier_falls_back_to_title_search() -> None:
    provider = MockProvider("p1", title_results=[vaswani_summary()])
    resolver = IdentityResolver(providers=[provider])

    result = await resolver.resolve(vaswani_entry(doi="not-a-doi"))

    assert result.accepted
    assert provider.doi_calls == []
    invalid = [a for a in result.attempts if a.reject_reason is RejectReason.UNVERIFIABLE_IDENTIFIER]
    assert invalid
    assert result.method is ResolutionMethod.TITLE_SEARCH


async def test_doi_stage_stops_after_first_accept() -> None:
    first = MockProvider("p1", doi_result=vaswani_summary())
    second = MockProvider("p2", doi_result=vaswani_summary(id="other"))
    resolver = IdentityResolver(providers=[first, second])

    result = await resolver.resolve(vaswani_entry())

    assert result.accepted
    assert second.doi_calls == []


async def test_deterministic_tie_prefers_smaller_canonical_id() -> None:
    candidate_a = vaswani_summary(id="aaa", doi=None)
    candidate_z = vaswani_summary(id="zzz", doi=None)
    forward = MockProvider("p1", title_results=[candidate_z, candidate_a])
    backward = MockProvider("p1", title_results=[candidate_a, candidate_z])
    result_forward = await IdentityResolver(providers=[forward]).resolve(
        vaswani_entry(doi=None)
    )
    result_backward = await IdentityResolver(providers=[backward]).resolve(
        vaswani_entry(doi=None)
    )

    assert result_forward.accepted and result_backward.accepted
    assert result_forward.canonical_id == "s2:aaa"
    assert result_backward.canonical_id == "s2:aaa"
    assert "s2:zzz" in result_forward.aliases
    assert "s2:aaa" not in result_forward.aliases


async def test_same_work_candidates_dedup_into_one_cluster() -> None:
    candidate_a = vaswani_summary(id="aaa", doi=VASWANI_DOI, provider="openalex")
    candidate_b = vaswani_summary(id="bbb", doi=VASWANI_DOI, provider="semantic_scholar")
    provider = MockProvider("p1", title_results=[candidate_a, candidate_b])
    resolver = IdentityResolver(providers=[provider])

    result = await resolver.resolve(vaswani_entry(doi=None))

    assert result.status is ResolutionStatus.RESOLVED
    assert result.canonical_id == "openalex:aaa"
    assert "s2:bbb" in result.aliases
    assert f"doi:{VASWANI_DOI}" in result.aliases


async def test_distinct_accepted_works_are_ambiguous() -> None:
    candidate_a = vaswani_summary(id="aaa", doi="10.1/aaaa")
    candidate_b = vaswani_summary(id="bbb", doi="10.1/bbbb")
    provider = MockProvider("p1", title_results=[candidate_a, candidate_b])
    resolver = IdentityResolver(providers=[provider])

    result = await resolver.resolve(vaswani_entry(doi=None))

    assert result.status is ResolutionStatus.AMBIGUOUS
    assert result.reject_reason is RejectReason.MULTIPLE_CANDIDATES
    assert result.canonical_id is None


async def test_resolution_registers_cross_provider_aliases(store: GraphStore) -> None:
    provider = MockProvider(
        "p1",
        doi_result=None,
        title_results=[vaswani_summary(arxiv_id="2301.00001")],
    )
    resolver = IdentityResolver(providers=[provider], store=store)

    result = await resolver.resolve(vaswani_entry())

    assert result.accepted
    assert store.get_canonical_id(f"doi:{VASWANI_DOI}") == "s2:abc123"
    assert store.get_canonical_id("arxiv:2301.00001") == "s2:abc123"
    assert store.get_aliases("s2:abc123") == sorted(
        [f"doi:{VASWANI_DOI}", "arxiv:2301.00001"]
    )


async def test_second_resolve_hits_alias_cache(store: GraphStore) -> None:
    provider = MockProvider("p1", doi_result=vaswani_summary())
    resolver = IdentityResolver(providers=[provider], store=store)

    first = await resolver.resolve(vaswani_entry())
    assert first.method is ResolutionMethod.DOI_LOOKUP
    provider.doi_result = None

    second = await resolver.resolve(vaswani_entry())

    assert second.method is ResolutionMethod.ALIAS_CACHE
    assert second.canonical_id == first.canonical_id
    assert provider.doi_calls == [VASWANI_DOI]


async def test_alias_cache_miss_without_store_falls_through() -> None:
    provider = MockProvider("p1", doi_result=vaswani_summary())
    resolver = IdentityResolver(providers=[provider])

    result = await resolver.resolve(vaswani_entry())

    assert result.accepted
    assert result.method is ResolutionMethod.DOI_LOOKUP


async def test_no_providers_rejected() -> None:
    resolver = IdentityResolver(providers=[])

    result = await resolver.resolve(vaswani_entry())

    assert result.status is ResolutionStatus.REJECTED
    assert result.reject_reason is RejectReason.NO_PROVIDER_MATCH


async def test_attempts_recorded_for_all_stages() -> None:
    provider = MockProvider("p1", doi_result=None, arxiv_result=None, title_results=[])
    resolver = IdentityResolver(providers=[provider])

    result = await resolver.resolve(
        vaswani_entry(doi="10.1/x", arxiv_id="2301.00001", title=VASWANI_TITLE)
    )

    methods = [a.method for a in result.attempts]
    assert ResolutionMethod.DOI_LOOKUP in methods
    assert ResolutionMethod.ARXIV_LOOKUP in methods
    assert ResolutionMethod.TITLE_SEARCH in methods
    assert all(a.status is ResolutionStatus.REJECTED for a in result.attempts)


async def test_arxiv_1905_07697v2_resolves_to_versionless_canonical() -> None:
    summary = vaswani_summary(
        id="1905.07697", doi=None, arxiv_id="1905.07697", provider="arxiv",
        title="Graph Attention Networks", year=2018, authors=["P. Velickovic"],
    )
    provider = MockProvider("p1", doi_result=None, arxiv_result=summary)
    resolver = IdentityResolver(providers=[provider])

    result = await resolver.resolve(
        BibliographicEntry(
            title="Graph Attention Networks",
            authors=["Velickovic"],
            year=2018,
            arxiv_id="arXiv:1905.07697v2",
        )
    )

    assert result.accepted
    assert result.method is ResolutionMethod.ARXIV_LOOKUP
    assert result.canonical_id == "arxiv:1905.07697"
    assert provider.arxiv_calls == ["1905.07697"]
    assert result.actual is not None and result.actual.arxiv_id == "1905.07697"


async def test_arxiv_1905_07697v2_title_mismatch_rejects() -> None:
    summary = vaswani_summary(
        id="1905.07697", doi=None, arxiv_id="1905.07697", provider="arxiv",
        title="Unrelated work", year=2018, authors=["P. Velickovic"],
    )
    provider = MockProvider("p1", doi_result=None, arxiv_result=summary)
    resolver = IdentityResolver(providers=[provider])

    result = await resolver.resolve(
        BibliographicEntry(
            title="Graph Attention Networks",
            authors=["Velickovic"],
            year=2018,
            arxiv_id="arXiv:1905.07697v2",
        )
    )

    assert not result.accepted
    assert result.status is ResolutionStatus.REJECTED
    assert result.reject_reason is RejectReason.TITLE_MISMATCH


async def test_title_only_resolution_with_arxiv_candidate() -> None:
    summary = vaswani_summary(
        id="1905.07697", doi=None, arxiv_id="1905.07697", provider="arxiv",
        title="Graph Attention Networks", year=2018, authors=["P. Velickovic"],
    )
    provider = MockProvider("p1", doi_result=None, arxiv_result=None,
                           title_results=[summary])
    resolver = IdentityResolver(providers=[provider])

    result = await resolver.resolve(
        BibliographicEntry(title="Graph Attention Networks", authors=["Velickovic"])
    )

    assert result.accepted
    assert result.method is ResolutionMethod.TITLE_SEARCH
    assert result.canonical_id == "arxiv:1905.07697"
    assert provider.title_calls == ["Graph Attention Networks"]
