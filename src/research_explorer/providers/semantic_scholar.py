"""Semantic Scholar Academic Graph API adapter.

Endpoints:
  - GET /paper/search          — search by query
  - GET /paper/{id}            — full metadata (accepts DOI, S2 ID, arXiv ID, PMID)
  - GET /paper/{id}/references — outgoing citations (references)
  - GET /paper/{id}/citations  — incoming citations (who cites this)

Free tier: ~100 req/5min unauthenticated, 1 req/s with free API key.
"""

from __future__ import annotations

from research_explorer.graph.models import Paper, PaperSummary
from research_explorer.providers.base import ResilientProvider

# Fields requested from the API for summaries vs full papers
SUMMARY_FIELDS = "title,year,authors,citationCount,abstract,externalIds"
PAPER_FIELDS = (
    "title,year,authors,citationCount,abstract,externalIds,"
    "references,citations,tldr,fieldsOfStudy"
)
NEIGHBOR_FIELDS = "title,year,authors,citationCount,externalIds"


class SemanticScholarProvider(ResilientProvider):
    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str = "data/.cache",
        rate: int = 1,
        period: int = 3,
        concurrency: int = 3,
        timeout: float = 30.0,
    ):
        super().__init__(
            name="semantic_scholar",
            base_url="https://api.semanticscholar.org/graph/v1",
            rate=rate,
            period=period,
            concurrency=concurrency,
            timeout=timeout,
            cache_ttl=86400,
            cache_dir=cache_dir,
            api_key=api_key,
        )

    async def search(self, query: str, limit: int = 10) -> list[PaperSummary]:
        data = await self.get_json("/paper/search", query=query, limit=limit, fields=SUMMARY_FIELDS)
        if not data or not isinstance(data, dict):
            return []
        return [self._parse_summary(p) for p in data.get("data", [])]

    async def get_paper(self, paper_id: str, id_type: str = "auto") -> Paper | None:
        pid = self._format_id(paper_id, id_type)
        data = await self.get_json(f"/paper/{pid}", fields=PAPER_FIELDS)
        if not data or not isinstance(data, dict):
            return None
        return self._parse_paper(data)

    async def get_references(self, paper_id: str, limit: int = 50) -> list[PaperSummary]:
        pid = self._format_id(paper_id, "auto")
        data = await self.get_json(
            f"/paper/{pid}/references", fields=NEIGHBOR_FIELDS, limit=limit
        )
        if not data or not isinstance(data, dict):
            return []
        return [
            self._parse_summary(r["citedPaper"])
            for r in data.get("data", [])
            if r.get("citedPaper")
        ]

    async def get_citations(self, paper_id: str, limit: int = 50) -> list[PaperSummary]:
        pid = self._format_id(paper_id, "auto")
        data = await self.get_json(
            f"/paper/{pid}/citations", fields=NEIGHBOR_FIELDS, limit=limit
        )
        if not data or not isinstance(data, dict):
            return []
        return [
            self._parse_summary(c["citingPaper"])
            for c in data.get("data", [])
            if c.get("citingPaper")
        ]

    async def get_abstract(self, paper_id: str) -> str | None:
        pid = self._format_id(paper_id, "auto")
        data = await self.get_json(f"/paper/{pid}", fields="abstract")
        if not data or not isinstance(data, dict):
            return None
        return data.get("abstract")

    def _format_id(self, paper_id: str, id_type: str) -> str:
        """Format an ID for the API path. Detects type automatically unless overridden."""
        if id_type == "auto":
            if paper_id.startswith("10."):
                return f"DOI:{paper_id}"
            if paper_id.isdigit():
                return f"PMID:{paper_id}"
            if paper_id.startswith("arXiv:"):
                return paper_id
            return paper_id  # assume S2 paperId
        return f"{id_type}:{paper_id}"

    def _parse_summary(self, d: dict) -> PaperSummary:
        ext = d.get("externalIds") or {}
        return PaperSummary(
            id=d.get("paperId") or "",
            doi=ext.get("DOI"),
            title=d.get("title") or "",
            year=d.get("year"),
            authors=[a.get("name", "") for a in (d.get("authors") or []) if isinstance(a, dict)],
            citation_count=d.get("citationCount"),
            abstract=d.get("abstract"),
            provider="semantic_scholar",
        )

    def _parse_paper(self, d: dict) -> Paper:
        ext = d.get("externalIds") or {}
        return Paper(
            id=d.get("paperId") or "",
            doi=ext.get("DOI"),
            title=d.get("title") or "",
            year=d.get("year"),
            authors=[a.get("name", "") for a in (d.get("authors") or []) if isinstance(a, dict)],
            citation_count=d.get("citationCount"),
            abstract=d.get("abstract"),
            provider="semantic_scholar",
            external_ids={k: str(v) for k, v in ext.items() if v is not None},
            fields_of_study=d.get("fieldsOfStudy") or [],
            tldr=(d.get("tldr") or {}).get("text"),
            references=[
                self._parse_summary(r)
                for r in (d.get("references") or [])
                if r and isinstance(r, dict) and r.get("paperId")
            ],
            citations=[
                self._parse_summary(c)
                for c in (d.get("citations") or [])
                if c and isinstance(c, dict) and c.get("paperId")
            ],
        )
