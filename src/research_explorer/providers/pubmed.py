"""PubMed / NCBI E-utilities adapter.

Multi-step workflow:
  - esearch: search by term -> PMIDs
  - efetch: get full record by PMID (XML -> parsed)
  - elink: get references (pubmed_pubmed_refs) or citations (pubmed_pubmed_citedin) by PMID

Free tier: 3 req/s unauthenticated, 10 req/s with free API key.
Requires `tool` and `email` params for polite usage.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from research_explorer.graph.models import Paper, PaperSummary
from research_explorer.providers.base import ResilientProvider


class PubMedProvider(ResilientProvider):
    def __init__(
        self,
        api_key: str | None = None,
        email: str = "research@example.com",
        tool: str = "research-explorer",
        cache_dir: str = "data/.cache",
        rate: int = 3,
        period: int = 1,
        concurrency: int = 5,
        timeout: float = 30.0,
    ):
        self.email = email
        self.tool = tool
        self.api_key = api_key
        super().__init__(
            name="pubmed",
            base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            rate=rate,
            period=period,
            concurrency=concurrency,
            timeout=timeout,
            cache_ttl=86400,
            cache_dir=cache_dir,
            api_key=None,  # PubMed uses query params, not headers
        )

    def _common_params(self) -> dict[str, Any]:
        params = {"tool": self.tool, "email": self.email}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    async def search(self, query: str, limit: int = 10) -> list[PaperSummary]:
        params = {**self._common_params(), "db": "pubmed", "term": query, "retmax": limit}
        data = await self.get_json("/esearch.fcgi", **params, retmode="json")
        if not data or not isinstance(data, dict):
            return []
        id_list = (data.get("esearchresult") or {}).get("idlist", [])
        if not id_list:
            return []
        # Fetch summaries for the PMIDs
        return await self._fetch_summaries(id_list)

    async def get_paper(self, paper_id: str, id_type: str = "auto") -> Paper | None:
        record = await self._fetch_record(paper_id)
        if not record:
            return None
        return record

    async def get_references(self, paper_id: str, limit: int = 50) -> list[PaperSummary]:
        params = {
            **self._common_params(),
            "dbfrom": "pubmed",
            "db": "pubmed",
            "id": paper_id,
            "linkname": "pubmed_pubmed_refs",
        }
        data = await self.get_json("/elink.fcgi", **params, retmode="json")
        if not data or not isinstance(data, dict):
            return []
        link_sets = (data.get("linksets") or [])
        pmids: list[str] = []
        for ls in link_sets:
            for ldb in ls.get("linksetdbs", []):
                if ldb.get("linkname") == "pubmed_pubmed_refs":
                    pmids.extend(ldb.get("links", []))
        pmids = pmids[:limit]
        return await self._fetch_summaries(pmids)

    async def get_citations(self, paper_id: str, limit: int = 50) -> list[PaperSummary]:
        params = {
            **self._common_params(),
            "dbfrom": "pubmed",
            "db": "pubmed",
            "id": paper_id,
            "linkname": "pubmed_pubmed_citedin",
        }
        data = await self.get_json("/elink.fcgi", **params, retmode="json")
        if not data or not isinstance(data, dict):
            return []
        link_sets = data.get("linksets") or []
        pmids: list[str] = []
        for ls in link_sets:
            for ldb in ls.get("linksetdbs", []):
                if ldb.get("linkname") == "pubmed_pubmed_citedin":
                    pmids.extend(ldb.get("links", []))
        pmids = pmids[:limit]
        return await self._fetch_summaries(pmids)

    async def get_abstract(self, paper_id: str) -> str | None:
        record = await self._fetch_record(paper_id)
        return record.abstract if record else None

    async def _fetch_summaries(self, pmids: list[str]) -> list[PaperSummary]:
        if not pmids:
            return []
        params = {**self._common_params(), "db": "pubmed", "id": ",".join(pmids)}
        data = await self.get_json("/esummary.fcgi", **params, retmode="json")
        if not data or not isinstance(data, dict):
            return []
        result = data.get("result") or {}
        summaries: list[PaperSummary] = []
        for pmid in pmids:
            rec = result.get(pmid)
            if not rec:
                continue
            summaries.append(
                PaperSummary(
                    id=pmid,
                    doi=rec.get("elocationid", "").replace("doi:", "") or None,
                    title=rec.get("title", ""),
                    year=int(rec.get("pubdate", "0")[:4]) if rec.get("pubdate") else None,
                    authors=[a.get("name", "") for a in rec.get("authors", [])],
                    citation_count=None,  # PubMed esummary doesn't provide this
                    abstract=None,  # requires efetch
                    provider="pubmed",
                )
            )
        return summaries

    async def _fetch_record(self, pmid: str) -> Paper | None:
        params = {**self._common_params(), "db": "pubmed", "id": pmid, "retmode": "xml"}
        # efetch returns XML, not JSON — use a raw GET via the client
        key = self._cache_key(f"/efetch/{pmid}", params)
        hit = self.cache.get(key)
        xml_text = hit
        if xml_text is None:
            try:
                resp = await self.breaker.call(self._attempt_get, "/efetch.fcgi", **params)
                xml_text = resp.text
                self.cache.set(key, xml_text, expire=self.cache_ttl)
            except Exception:
                return None
        if not xml_text:
            return None
        return self._parse_pubmed_xml(pmid, xml_text)

    def _parse_pubmed_xml(self, pmid: str, xml_text: str) -> Paper | None:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return None
        article = root.find(".//PubmedArticle/MedlineCitation/Article")
        if article is None:
            return None
        title_el = article.find("ArticleTitle")
        title = title_el.text if title_el is not None and title_el.text else ""
        year = None
        pubdate = root.find(".//PubmedData/History/PubMedPubDate[@PubStatus='pubmed']/Year")
        if pubdate is not None and pubdate.text:
            year = int(pubdate.text)
        authors: list[str] = []
        for au in article.findall(".//Author"):
            last = au.find("LastName")
            fore = au.find("ForeName")
            name = " ".join(filter(None, [fore.text if fore is not None else "", last.text if last is not None else ""]))
            if name:
                authors.append(name)
        abstract_parts = article.findall(".//Abstract/AbstractText")
        abstract = " ".join((p.text or "") for p in abstract_parts) if abstract_parts else None
        doi = None
        for aid in root.findall(".//PubmedData/ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi":
                doi = aid.text
                break
        return Paper(
            id=pmid,
            doi=doi,
            title=title,
            year=year,
            authors=authors,
            citation_count=None,
            abstract=abstract,
            provider="pubmed",
        )
