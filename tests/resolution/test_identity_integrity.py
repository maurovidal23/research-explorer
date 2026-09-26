"""STAB-4: identity integrity for metadata-less candidates and identifier evidence."""

from __future__ import annotations

from research_explorer.graph.models import PaperSummary
from research_explorer.resolution.models import (
    BibliographicEntry,
    RejectReason,
    ResolutionStatus,
)
from research_explorer.resolution.resolver import IdentityResolver, candidate_dedup_key

DOI = "10.1038/nrn3241"


class _Provider:
    name = "openalex"

    def __init__(self, doi_result: PaperSummary | None) -> None:
        self.doi_result = doi_result
        self.arxiv_result: PaperSummary | None = None

    async def lookup_doi(self, doi: str) -> PaperSummary | None:
        return self.doi_result

    async def lookup_arxiv(self, arxiv_id: str) -> PaperSummary | None:
        return self.arxiv_result

    async def search_title(self, title: str) -> list[PaperSummary]:
        return []


def _metadata_less(native_id: str, provider: str = "openalex") -> PaperSummary:
    return PaperSummary(id=native_id, title="", provider=provider, doi=DOI)


def test_candidate_dedup_key_metadata_less_uses_provider_native_id() -> None:
    key = candidate_dedup_key(
        PaperSummary(id="W1", title="", provider="openalex", doi=None)
    )
    assert key == "openalex:W1"


def test_metadata_less_candidates_do_not_collapse() -> None:
    a = candidate_dedup_key(
        PaperSummary(id="W1", title="", provider="openalex", doi=None)
    )
    b = candidate_dedup_key(
        PaperSummary(id="W2", title="", provider="openalex", doi=None)
    )
    assert a != b


def test_same_metadata_less_candidate_stays_equal() -> None:
    key1 = candidate_dedup_key(
        PaperSummary(id="W1", title="", provider="openalex", doi=None)
    )
    key2 = candidate_dedup_key(
        PaperSummary(id="W1", title="", provider="openalex", doi=None)
    )
    assert key1 == key2


async def test_exact_doi_lookup_passes_floor_without_metadata() -> None:
    resolver = IdentityResolver(providers=[_Provider(_metadata_less("W1"))])
    result = await resolver.resolve(BibliographicEntry(doi=DOI))
    assert result.status is ResolutionStatus.RESOLVED
    assert result.confidence == 1.0
    assert result.canonical_id == "openalex:W1"


async def test_contradictory_title_still_triggers_hard_gate() -> None:
    candidate = PaperSummary(
        id="W1",
        title="A completely different paper about pastry baking",
        provider="openalex",
        doi=DOI,
    )
    resolver = IdentityResolver(providers=[_Provider(candidate)])
    result = await resolver.resolve(
        BibliographicEntry(doi=DOI, title="The origin of extracellular fields")
    )
    assert result.status is ResolutionStatus.REJECTED
    assert result.reject_reason is RejectReason.TITLE_MISMATCH
