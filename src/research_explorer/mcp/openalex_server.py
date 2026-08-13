"""MCP server exposing OpenAlex tools.

Run standalone:
    python -m research_explorer.mcp.openalex_server
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from research_explorer.config import load_config
from research_explorer.graph.models import Paper, PaperSummary
from research_explorer.providers.openalex import OpenAlexProvider

mcp = FastMCP("openalex")

_provider: OpenAlexProvider | None = None


def _get_provider() -> OpenAlexProvider:
    global _provider
    if _provider is None:
        cfg = load_config("config/default.toml")
        _provider = OpenAlexProvider(
            api_key=os.environ.get("OPENALEX_API_KEY"),
            cache_dir=cfg.storage.cache_dir,
        )
    return _provider


@mcp.tool()
async def search_papers(query: str, limit: int = 10) -> list[PaperSummary]:
    """Search academic papers by text query via OpenAlex (270M+ works)."""
    return await _get_provider().search(query, limit=limit)


@mcp.tool()
async def get_paper(paper_id: str, id_type: str = "auto") -> Paper | None:
    """Get full metadata for a paper via OpenAlex.

    Args:
        paper_id: OpenAlex ID (W...), DOI, or PMID.
        id_type: "auto" (default), "DOI", or "PMID".
    """
    return await _get_provider().get_paper(paper_id, id_type=id_type)


@mcp.tool()
async def get_references(paper_id: str, limit: int = 50) -> list[PaperSummary]:
    """Get papers referenced by this paper — outgoing citations (from referenced_works)."""
    return await _get_provider().get_references(paper_id, limit=limit)


@mcp.tool()
async def get_citations(paper_id: str, limit: int = 50) -> list[PaperSummary]:
    """Get papers that cite this paper — incoming citations (filter=cited_by)."""
    return await _get_provider().get_citations(paper_id, limit=limit)


@mcp.tool()
async def get_abstract(paper_id: str) -> str:
    """Get the abstract of a paper (reconstructed from inverted index)."""
    result = await _get_provider().get_abstract(paper_id)
    return result or "Abstract not available."


if __name__ == "__main__":
    mcp.run(transport="stdio")
