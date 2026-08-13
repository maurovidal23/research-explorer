"""MCP server exposing arXiv tools.

Run standalone:
    python -m research_explorer.mcp.arxiv_server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from research_explorer.config import load_config
from research_explorer.graph.models import Paper, PaperSummary
from research_explorer.providers.arxiv import ArxivProvider

mcp = FastMCP("arxiv")

_provider: ArxivProvider | None = None


def _get_provider() -> ArxivProvider:
    global _provider
    if _provider is None:
        cfg = load_config("config/default.toml")
        _provider = ArxivProvider(cache_dir=cfg.storage.cache_dir)
    return _provider


@mcp.tool()
async def search_papers(query: str, limit: int = 10) -> list[PaperSummary]:
    """Search arXiv preprints by text query (supports field prefixes like ti:, au:, cat:)."""
    return await _get_provider().search(query, limit=limit)


@mcp.tool()
async def get_paper(paper_id: str, id_type: str = "auto") -> Paper | None:
    """Get full metadata for an arXiv preprint by arXiv ID.

    Args:
        paper_id: arXiv ID (e.g. 2301.00001); 'arXiv:' prefix is accepted.
        id_type: Identifier type (default "auto" — assumes arXiv ID).
    """
    return await _get_provider().get_paper(paper_id, id_type=id_type)


@mcp.tool()
async def get_references(paper_id: str, limit: int = 50) -> list[PaperSummary]:
    """arXiv does not expose a reference graph — always returns an empty list."""
    return await _get_provider().get_references(paper_id, limit=limit)


@mcp.tool()
async def get_citations(paper_id: str, limit: int = 50) -> list[PaperSummary]:
    """arXiv does not expose a citation graph — always returns an empty list."""
    return await _get_provider().get_citations(paper_id, limit=limit)


@mcp.tool()
async def get_abstract(paper_id: str) -> str:
    """Get the abstract of an arXiv preprint."""
    result = await _get_provider().get_abstract(paper_id)
    return result or "Abstract not available."


if __name__ == "__main__":
    mcp.run(transport="stdio")
