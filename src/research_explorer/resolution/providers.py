"""Adapters that expose academic providers as resolver VerificationProviders.

Each adapter translates the resolver's three lookup operations (DOI, arXiv id,
title search) onto an existing ResilientProvider instance, so the deterministic
IdentityResolver can verify bibliographic entries without any new network code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from research_explorer.graph.models import Paper, PaperSummary
from research_explorer.providers.base import ResilientProvider
from research_explorer.providers.openalex import OpenAlexProvider
from research_explorer.providers.semantic_scholar import SemanticScholarProvider
from research_explorer.resolution.resolver import normalize_doi

if TYPE_CHECKING:
    from research_explorer.providers.arxiv import ArxivProvider

ARXIV_DOI_PREFIX = "10.48550/arxiv."


def as_summary(paper: Paper | None) -> PaperSummary | None:
    if paper is None:
        return None
    return PaperSummary(
        id=paper.id,
        doi=paper.doi,
        arxiv_id=paper.arxiv_id,
        title=paper.title,
        year=paper.year,
        authors=paper.authors,
        citation_count=paper.citation_count,
        abstract=paper.abstract,
        provider=paper.provider,
    )


class TitleSearchCapable(Protocol):
    name: str

    async def lookup_doi(self, doi: str) -> PaperSummary | None: ...

    async def lookup_arxiv(self, arxiv_id: str) -> PaperSummary | None: ...

    async def search_title(self, title: str) -> list[PaperSummary]: ...


class OpenAlexVerification:
    """Verification adapter over OpenAlexProvider.

    arXiv identifiers are looked up through their DataCite DOI form
    (10.48550/arxiv.<id>), which OpenAlex resolves for the whole arXiv corpus.
    """

    name = "openalex"

    def __init__(self, provider: OpenAlexProvider, title_limit: int = 5):
        self.provider = provider
        self.title_limit = title_limit

    async def lookup_doi(self, doi: str) -> PaperSummary | None:
        paper = await self.provider.get_paper(doi, id_type="DOI")
        return as_summary(paper)

    async def lookup_arxiv(self, arxiv_id: str) -> PaperSummary | None:
        paper = await self.provider.get_paper(f"{ARXIV_DOI_PREFIX}{arxiv_id}", id_type="DOI")
        return as_summary(paper)

    async def search_title(self, title: str) -> list[PaperSummary]:
        return await self.provider.search(title, limit=self.title_limit)


class SemanticScholarVerification:
    """Verification adapter over SemanticScholarProvider."""

    name = "semantic_scholar"

    def __init__(self, provider: SemanticScholarProvider, title_limit: int = 5):
        self.provider = provider
        self.title_limit = title_limit

    async def lookup_doi(self, doi: str) -> PaperSummary | None:
        paper = await self.provider.get_paper(f"DOI:{doi}")
        return as_summary(paper)

    async def lookup_arxiv(self, arxiv_id: str) -> PaperSummary | None:
        paper = await self.provider.get_paper(f"arXiv:{arxiv_id}")
        return as_summary(paper)

    async def search_title(self, title: str) -> list[PaperSummary]:
        return await self.provider.search(title, limit=self.title_limit)


class ArxivVerification:
    """Verification adapter over ArxivProvider."""

    name = "arxiv"

    def __init__(self, provider: ArxivProvider, title_limit: int = 5):
        self.provider = provider
        self.title_limit = title_limit

    async def lookup_doi(self, doi: str) -> PaperSummary | None:
        candidates = await self.provider.search(f"doi:{doi}", limit=self.title_limit)
        normalized = normalize_doi(doi)
        return next(
            (
                candidate
                for candidate in candidates
                if candidate.doi and normalize_doi(candidate.doi) == normalized
            ),
            None,
        )

    async def lookup_arxiv(self, arxiv_id: str) -> PaperSummary | None:
        paper = await self.provider.get_paper(arxiv_id, id_type="arxiv")
        return as_summary(paper)

    async def search_title(self, title: str) -> list[PaperSummary]:
        return await self.provider.search(f'ti:"{title}"', limit=self.title_limit)


VerificationAdapter = (
    type[ArxivVerification]
    | type[OpenAlexVerification]
    | type[SemanticScholarVerification]
)

_ADAPTERS: dict[str, VerificationAdapter] = {
    "arxiv": ArxivVerification,
    "openalex": OpenAlexVerification,
    "semantic_scholar": SemanticScholarVerification,
}


def build_verification_providers(
    order: list[str],
    providers: dict[str, ResilientProvider],
    title_limit: int = 5,
) -> list[TitleSearchCapable]:
    """Build verification adapters in the given provider order.

    Providers without an adapter or not present in the registry are skipped.
    """
    adapters: list[TitleSearchCapable] = []
    for name in order:
        provider = providers.get(name)
        adapter_cls = _ADAPTERS.get(name)
        if provider is None or adapter_cls is None:
            continue
        if adapter_cls is ArxivVerification:
            adapters.append(
                ArxivVerification(
                    cast("ArxivProvider", provider), title_limit=title_limit
                )
            )
        elif adapter_cls is OpenAlexVerification:
            adapters.append(OpenAlexVerification(cast(OpenAlexProvider, provider), title_limit=title_limit))
        else:
            adapters.append(SemanticScholarVerification(cast(SemanticScholarProvider, provider), title_limit=title_limit))
    return adapters
