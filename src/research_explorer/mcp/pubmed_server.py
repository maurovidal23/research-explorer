"""MCP server exposing PubMed E-utilities tools.

Run standalone:
    python -m research_explorer.mcp.pubmed_server
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from research_explorer.config import load_config
from research_explorer.graph.models import Paper, PaperSummary
from research_explorer.providers.pubmed import PubMedProvider

mcp = FastMCP("pubmed")

_provider: PubMedProvider | None = None


def _get_provider() -> PubMedProvider:
    global _provider
    if _provider is None:
        cfg = load_config("config/default.toml")
        _provider = PubMedProvider(
            api_key=os.environ.get("PUBMED_API_KEY"),
            email=os.environ.get("PUBMED_EMAIL", "research@example.com"),
            cache_dir=cfg.storage.cache_dir,
        )
    return _provider


@mcp.tool()
async def search_papers(query: str, limit: int = 10) -> list[PaperSummary]:
    """Search PubMed by term (supports [field] tags and boolean operators)."""
    return await _get_provider().search(query, limit=limit)


@mcp.tool()
async def get_paper(paper_id: str, id_type: str = "auto") -> Paper | None:
    """Get full metadata for a PubMed article by PMID.

    Args:
        paper_id: PubMed ID (PMID).
        id_type: Identifier type (default "auto" — assumes PMID).
    """
    return await _get_provider().get_paper(paper_id, id_type=id_type)


@mcp.tool()
async def get_references(paper_id: str, limit: int = 50) -> list[PaperSummary]:
    """Get papers referenced by this PubMed article (elink: pubmed_pubmed_refs)."""
    return await _get_provider().get_references(paper_id, limit=limit)


@mcp.tool()
async def get_citations(paper_id: str, limit: int = 50) -> list[PaperSummary]:
    """Get papers that cite this PubMed article (elink: pubmed_pubmed_citedin)."""
    return await _get_provider().get_citations(paper_id, limit=limit)


@mcp.tool()
async def get_abstract(paper_id: str) -> str:
    """Get the abstract of a PubMed article."""
    result = await _get_provider().get_abstract(paper_id)
    return result or "Abstract not available."


if __name__ == "__main__":
    mcp.run(transport="stdio")
