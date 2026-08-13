"""MCP server exposing Semantic Scholar tools.

Run standalone:
    python -m research_explorer.mcp.semantic_scholar_server

Or via Claude Desktop config (see README).
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from research_explorer.config import Config, load_config
from research_explorer.graph.models import Paper, PaperSummary
from research_explorer.providers.semantic_scholar import SemanticScholarProvider

mcp = FastMCP("semantic-scholar")

_provider: SemanticScholarProvider | None = None
_config: Config | None = None


def _get_provider() -> SemanticScholarProvider:
    global _provider, _config
    if _provider is None:
        _config = load_config("config/default.toml")
        _provider = SemanticScholarProvider(
            api_key=__import__("os").environ.get("S2_API_KEY"),
            cache_dir=_config.storage.cache_dir,
        )
    return _provider


@mcp.tool()
async def search_papers(query: str, limit: int = 10) -> list[PaperSummary]:
    """Search academic papers by text query via Semantic Scholar.

    Args:
        query: Text query (title, keywords, author).
        limit: Max results (1-100).
    """
    return await _get_provider().search(query, limit=limit)


@mcp.tool()
async def get_paper(paper_id: str, id_type: str = "auto") -> Paper | None:
    """Get full metadata for a paper, including references and citations.

    Args:
        paper_id: Paper identifier (DOI, S2 paperId, arXiv ID, or PMID).
        id_type: Identifier type — "auto" (default), "DOI", "PMID", "arXiv".
    """
    return await _get_provider().get_paper(paper_id, id_type=id_type)


@mcp.tool()
async def get_references(paper_id: str, limit: int = 50) -> list[PaperSummary]:
    """Get papers referenced by (cited by) this paper — outgoing citations.

    Args:
        paper_id: Paper identifier (DOI, S2 paperId, arXiv ID, or PMID).
        limit: Max references to return.
    """
    return await _get_provider().get_references(paper_id, limit=limit)


@mcp.tool()
async def get_citations(paper_id: str, limit: int = 50) -> list[PaperSummary]:
    """Get papers that cite this paper — incoming citations.

    Args:
        paper_id: Paper identifier (DOI, S2 paperId, arXiv ID, or PMID).
        limit: Max citations to return.
    """
    return await _get_provider().get_citations(paper_id, limit=limit)


@mcp.tool()
async def get_abstract(paper_id: str) -> str:
    """Get the abstract of a paper.

    Args:
        paper_id: Paper identifier (DOI, S2 paperId, arXiv ID, or PMID).
    """
    result = await _get_provider().get_abstract(paper_id)
    return result or "Abstract not available."


if __name__ == "__main__":
    mcp.run(transport="stdio")
