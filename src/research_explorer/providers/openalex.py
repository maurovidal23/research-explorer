"""OpenAlex API adapter.

Endpoints:
  - GET /works?search={q}                 -- search by query
  - GET /works/{id}                       -- full metadata (OpenAlex ID, DOI, PMID)
  - GET /works?filter=cited_by:{id}       -- incoming citations (who cites this)

References (outgoing) come from the `referenced_works` field on the work object.
Abstracts are an inverted index (often null for closed-access).

Free tier: ~10 req/s polite pool (add mailto). No API key required.
"""

from __future__ import annotations

from research_explorer.graph.models import Paper, PaperSummary
from research_explorer.providers.base import ResilientProvider

SELECT_FIELDS = (
    "id,title,publication_year,authorships,cited_by_count,"
    "referenced_works,abstract_inverted_index,ids,topics,open_access"
)
SUMMARY_SELECT = "id,title,publication_year,authorships,cited_by_count,ids"
CITATION_LIMIT = 50


class OpenAlexProvider(ResilientProvider):
    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str = "data/.cache",
        rate: int = 10,
        period: int = 1,
        concurrency: int = 10,
        timeout: float = 30.0,
        email: str = "research@example.com",
    ):
        headers = {}
        if api_key:
            headers["api_key"] = api_key
        super().__init__(
            name="openalex",
            base_url="https://api.openalex.org",
            rate=rate,
            period=period,
            concurrency=concurrency,
            timeout=timeout,
            cache_ttl=86400,
            cache_dir=cache_dir,
            api_key=None,
            extra_headers={
                "User-Agent": f"research-explorer/0.1 (mailto:{email})",
                **headers,
            },
        )

    async def search(self, query: str, limit: int = 10) -> list[PaperSummary]:
        data = await self.get_json(
            "/works", search=query, per_page=limit, select=SUMMARY_SELECT,
            mailto="research@example.com",
        )
        if not data or not isinstance(data, dict):
            return []
        return [self._parse_summary(w) for w in data.get("results", [])]

    async def get_paper(self, paper_id: str, id_type: str = "auto") -> Paper | None:
        oid = self._format_id(paper_id, id_type)
        data = await self.get_json(
            f"/works/{oid}", select=SELECT_FIELDS, mailto="research@example.com",
        )
        if not data or not isinstance(data, dict):
            return None
        paper = self._parse_paper(data)

        citations = await self._fetch_citations(paper.id)
        paper.citations = citations
        return paper

    async def _fetch_citations(self, openalex_id: str) -> list[PaperSummary]:
        short_id = openalex_id.replace("https://openalex.org/", "")
        data = await self.get_json(
            "/works",
            filter=f"cited_by:{short_id}",
            per_page=CITATION_LIMIT,
            select=SUMMARY_SELECT,
            mailto="research@example.com",
        )
        if not data or not isinstance(data, dict):
            return []
        return [self._parse_summary(w) for w in data.get("results", [])]

    async def get_references(self, paper_id: str, limit: int = 50) -> list[PaperSummary]:
        paper = await self.get_paper(paper_id)
        if not paper:
            return []
        return paper.references[:limit]

    async def get_citations(self, paper_id: str, limit: int = 50) -> list[PaperSummary]:
        oid = self._format_id(paper_id, "auto")
        short_id = oid.replace("https://openalex.org/", "")
        data = await self.get_json(
            "/works",
            filter=f"cited_by:{short_id}",
            per_page=limit,
            select=SUMMARY_SELECT,
            mailto="research@example.com",
        )
        if not data or not isinstance(data, dict):
            return []
        return [self._parse_summary(w) for w in data.get("results", [])]

    async def get_works_batch(self, openalex_ids: list[str], limit: int = 50) -> list[PaperSummary]:
        """Fetch metadata for a batch of OpenAlex IDs in a single request."""
        ids: list[str] = []
        for raw in openalex_ids[:limit]:
            wid = raw.replace("https://openalex.org/", "")
            if wid and wid not in ids:
                ids.append(wid)
        if not ids:
            return []
        data = await self.get_json(
            "/works",
            filter=f"ids.openalex:{'|'.join(ids)}",
            per_page=len(ids),
            select=SUMMARY_SELECT,
            mailto="research@example.com",
        )
        if not data or not isinstance(data, dict):
            return []
        return [self._parse_summary(w) for w in data.get("results", [])]

    async def get_abstract(self, paper_id: str) -> str | None:
        oid = self._format_id(paper_id, "auto")
        data = await self.get_json(
            f"/works/{oid}", select="abstract_inverted_index",
            mailto="research@example.com",
        )
        if not data or not isinstance(data, dict):
            return None
        return self._reconstruct_abstract(data.get("abstract_inverted_index"))

    def _format_id(self, paper_id: str, id_type: str) -> str:
        if id_type == "auto":
            if paper_id.startswith("10."):
                return f"doi:{paper_id}"
            if paper_id.startswith("doi:"):
                return paper_id
            if paper_id.isdigit():
                return f"pmid:{paper_id}"
            if paper_id.startswith("W"):
                return paper_id
            return paper_id
        if id_type == "DOI":
            return f"doi:{paper_id}"
        if id_type == "PMID":
            return f"pmid:{paper_id}"
        return paper_id

    def _reconstruct_abstract(self, inverted: dict | None) -> str | None:
        if not inverted:
            return None
        positions: dict[int, str] = {}
        for word, idxs in inverted.items():
            for i in idxs:
                positions[i] = word
        return " ".join(positions[i] for i in sorted(positions))

    def _parse_ref_id(self, url: str) -> str:
        return url.replace("https://openalex.org/", "")

    def _parse_summary(self, w: dict) -> PaperSummary:
        ids = w.get("ids") or {}
        return PaperSummary(
            id=w.get("id", "").replace("https://openalex.org/", "") or self._parse_ref_id(w.get("id", "")),
            doi=(ids.get("doi") or "").replace("https://doi.org/", "") or None,
            title=w.get("title") or "",
            year=w.get("publication_year"),
            authors=[
                (a.get("author") or {}).get("display_name", "")
                for a in (w.get("authorships") or [])
                if isinstance(a, dict)
            ],
            citation_count=w.get("cited_by_count"),
            abstract=self._reconstruct_abstract(w.get("abstract_inverted_index")),
            provider="openalex",
        )

    def _parse_paper(self, w: dict) -> Paper:
        ids = w.get("ids") or {}
        ref_urls = w.get("referenced_works") or []
        references = [
            PaperSummary(
                id=self._parse_ref_id(url),
                title="",
                provider="openalex",
            )
            for url in ref_urls
            if isinstance(url, str) and url.startswith("https://openalex.org/")
        ]
        external_ids = {k: str(v) for k, v in ids.items() if v is not None}
        return Paper(
            id=w.get("id", "").replace("https://openalex.org/", ""),
            doi=(ids.get("doi") or "").replace("https://doi.org/", "") or None,
            title=w.get("title") or "",
            year=w.get("publication_year"),
            authors=[
                (a.get("author") or {}).get("display_name", "")
                for a in (w.get("authorships") or [])
                if isinstance(a, dict)
            ],
            citation_count=w.get("cited_by_count"),
            abstract=self._reconstruct_abstract(w.get("abstract_inverted_index")),
            provider="openalex",
            external_ids=external_ids,
            fields_of_study=[
                t.get("display_name", "")
                for t in (w.get("topics") or [])
                if isinstance(t, dict)
            ],
            references=references,
            citations=[],
        )
