"""Data models for papers and citation graph nodes.

Shared by providers, graph store, MCP servers, and the ACO engine.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PaperSummary(BaseModel):
    """Lightweight paper reference — used in neighbor lists and search results."""

    id: str = Field(description="Native provider ID (e.g. S2 paperId, OpenAlex W-id, PMID)")
    doi: str | None = None
    arxiv_id: str | None = None
    pmid: str | None = None
    title: str
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    citation_count: int | None = None
    abstract: str | None = None
    provider: str = Field(description="Provider name: semantic_scholar | openalex | pubmed | arxiv")


class Paper(PaperSummary):
    """Full paper record with references, citations, and extra metadata."""

    references: list[PaperSummary] = Field(default_factory=list)
    citations: list[PaperSummary] = Field(default_factory=list)
    external_ids: dict[str, str] = Field(default_factory=dict)
    fields_of_study: list[str] = Field(default_factory=list)
    tldr: str | None = None
    embedding: list[float] | None = None
    fulltext: str | None = Field(default=None, description="Full-text body (truncated), when available")
    ref_entries: list[str] = Field(
        default_factory=list,
        description="Raw bibliography entry strings (for LLM reference extraction)",
    )
    bibliography_error: str | None = Field(
        default=None,
        description="Distinct failure code when bibliography segmentation failed (§9)",
    )


def normalize_id(provider: str, native_id: str) -> str:
    """Build a normalized cross-provider ID: 'provider:native_id'.

    Examples:
        normalize_id("semantic_scholar", "da82f8e6...") -> "s2:da82f8e6..."
        normalize_id("openalex", "W1973275441") -> "openalex:W1973275441"
        normalize_id("pubmed", "22595786") -> "pmid:22595786"
        normalize_id("arxiv", "2301.00001") -> "arxiv:2301.00001"
    """
    prefix = {
        "semantic_scholar": "s2",
        "openalex": "openalex",
        "pubmed": "pmid",
        "arxiv": "arxiv",
    }.get(provider, provider)
    existing_prefix, separator, existing_native = native_id.partition(":")
    if separator and existing_prefix.lower() == prefix.lower():
        return f"{prefix}:{existing_native}"
    return f"{prefix}:{native_id}"


def parse_normalized_id(nid: str) -> tuple[str, str]:
    """Split a normalized ID into (provider, native_id).

    Returns:
        Tuple of (provider_name, native_id). provider_name is one of
        semantic_scholar, openalex, pubmed, arxiv, or the raw prefix if unknown.
    """
    prefix, _, native = nid.partition(":")
    provider = {
        "s2": "semantic_scholar",
        "openalex": "openalex",
        "pmid": "pubmed",
        "arxiv": "arxiv",
    }.get(prefix, prefix)
    return provider, native
