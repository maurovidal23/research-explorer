"""arXiv API adapter.

Endpoints:
  - GET /api/query?search_query=...  -- search by query (all fields)
  - GET /api/query?id_list=...       -- fetch records by arXiv ID (Atom XML)
  - GET https://arxiv.org/html/<id>   -- full-text HTML (LaTeXML), used to
                                          read the paper and extract its
                                          bibliography (the citation graph is
                                          NOT exposed via any API).

arXiv has no citation/reference API, so get_references/get_citations return
empty lists at the provider level. Instead, the agent reads the full text
(get_fulltext_and_refs) and the LLM extracts the references from the
bibliography to build a per-agent graph.

Free tier: 1 req per 3s recommended (be polite). No API key required.
"""

from __future__ import annotations

import io
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any

import aiobreaker
import httpx
from pypdf import PdfReader
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
)

from research_explorer.graph.models import Paper, PaperSummary
from research_explorer.logging_setup import get_logger
from research_explorer.providers.base import (
    RETRYABLE,
    ResilientProvider,
    TransientProviderError,
    _raise_for_retryable,
    _wait_retry_after,
)
from research_explorer.redaction import redact_secrets

log = get_logger("providers")

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

_FIELD_PREFIX = re.compile(r"^[a-z_]+:")
_VERSION = re.compile(r"v\d+$")

_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")
_HEAD = re.compile(r"<head>.*?</head>", re.DOTALL | re.IGNORECASE)
_SCRIPT = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_BIB_SECTION = re.compile(
    r'<section[^>]*class="ltx_bibliography"[^>]*>(.*?)</section>', re.DOTALL
)
_BIB_ITEM = re.compile(
    r'<li[^>]*class="ltx_bibitem"[^>]*>(.*?)</li>', re.DOTALL
)
_REF_HEADING = re.compile(r'\n[ \t]*References[ \t]*\n', re.IGNORECASE)
_BRACKET_REF = re.compile(r'\n\s*\[1\]\s*')
_BRACKET_SPLIT = re.compile(r'\n\s*\[\d+\]\s*')
_DOT_SPLIT = re.compile(r'\n\s*\d+\.\s+')


class ArxivProvider(ResilientProvider):
    supports_fulltext = True

    def __init__(
        self,
        cache_dir: str = "data/.cache",
        rate: int = 1,
        period: int = 3,
        concurrency: int = 3,
        timeout: float = 30.0,
    ):
        super().__init__(
            name="arxiv",
            base_url="https://export.arxiv.org",
            rate=rate,
            period=period,
            concurrency=concurrency,
            timeout=timeout,
            cache_ttl=86400,
            cache_dir=cache_dir,
            api_key=None,
            extra_headers={"Accept": "application/atom+xml"},
        )
        self.html_client = httpx.AsyncClient(
            base_url="https://arxiv.org",
            timeout=httpx.Timeout(timeout, connect=5.0, pool=10.0),
            limits=httpx.Limits(
                max_keepalive_connections=concurrency,
                max_connections=concurrency,
            ),
            http2=True,
            headers={
                "User-Agent": "research-explorer/0.1 (mailto:research@example.com)",
                "Accept": "text/html",
            },
            event_hooks={
                "request": [self._log_request],
                "response": [self._log_response],
            },
        )

    async def search(self, query: str, limit: int = 10) -> list[PaperSummary]:
        root = await self._get_xml(
            "/api/query", search_query=self._build_search_query(query),
            start=0, max_results=limit,
        )
        if root is None:
            return []
        return [self._parse_summary(e) for e in root.findall("atom:entry", _NS)]

    async def get_paper(self, paper_id: str, id_type: str = "auto") -> Paper | None:
        aid = self._format_id(paper_id, id_type)
        root = await self._get_xml("/api/query", id_list=aid)
        if root is None:
            return None
        entries = root.findall("atom:entry", _NS)
        if not entries:
            return None
        return self._parse_paper(entries[0])

    async def get_references(self, paper_id: str, limit: int = 50) -> list[PaperSummary]:
        return []

    async def get_citations(self, paper_id: str, limit: int = 50) -> list[PaperSummary]:
        return []

    async def get_abstract(self, paper_id: str) -> str | None:
        paper = await self.get_paper(paper_id)
        return paper.abstract if paper else None

    async def get_fulltext_and_refs(
        self, paper_id: str, max_chars: int = 16000, ref_limit: int = 100
    ) -> tuple[str, list[str]] | None:
        """Fetch the full text and return (truncated plain text, reference entries).

        Strategy:
          1. Try the LaTeXML HTML rendering (clean bibliography, ~30% of papers).
          2. Fall back to the PDF (100% of papers) — extract text with pypdf
             and parse the references section with regex.

        Returns None only if both HTML and PDF are unavailable.
        """
        aid = self._format_id(paper_id, "auto")

        # 1. Try HTML
        html = await self._get_html(f"/html/{aid}")
        if html is not None:
            return self._html_to_text_and_refs(html, max_chars, ref_limit)

        # 2. Fall back to PDF
        pdf = await self._get_pdf(f"/pdf/{aid}")
        if pdf is not None:
            return self._pdf_to_text_and_refs(pdf, max_chars, ref_limit)

        return None

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=_wait_retry_after,
        retry=retry_if_exception_type(RETRYABLE),
        before_sleep=before_sleep_log(log, logging.WARNING),
    )
    async def _attempt_get_html(self, path: str, **params: Any) -> httpx.Response:
        async with self.sem, self.limiter:
            resp = await self.html_client.get(path, params=params)
            _raise_for_retryable(resp)
            return resp

    async def _get_html(self, path: str, **params: Any) -> str | None:
        key = self._cache_key(path, params)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        try:
            resp = await self.breaker.call(self._attempt_get_html, path, **params)
        except aiobreaker.CircuitBreakerError:
            log.warning("circuit_open", provider=self.name, path=path)
            return None
        except RETRYABLE:
            log.warning("request_failed", provider=self.name, path=path)
            return None
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                log.warning("http_error", provider=self.name, path=path, status=e.response.status_code)
            return None
        if resp.status_code == 200 and resp.text:
            text = resp.text
            self.cache.set(key, text, expire=self.cache_ttl)
            return text
        return None

    async def _get_pdf(self, path: str, **params: Any) -> bytes | None:
        """Fetch a PDF and return its raw bytes (cached)."""
        key = self._cache_key(path, params)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        try:
            resp = await self.breaker.call(self._attempt_get_html, path, **params)
        except aiobreaker.CircuitBreakerError:
            log.warning("circuit_open", provider=self.name, path=path)
            return None
        except RETRYABLE:
            log.warning("request_failed", provider=self.name, path=path)
            return None
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                log.warning("http_error", provider=self.name, path=path, status=e.response.status_code)
            return None
        if resp.status_code == 200 and resp.content:
            data = resp.content
            self.cache.set(key, data, expire=self.cache_ttl)
            return data
        return None

    def _pdf_to_text_and_refs(
        self, pdf_bytes: bytes, max_chars: int, ref_limit: int
    ) -> tuple[str, list[str]]:
        """Extract text and references from a PDF via pypdf."""
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
        except Exception as e:
            log.warning("pdf_parse_failed", error=str(e))
            return "", []
        full_text = ""
        for page in reader.pages:
            try:
                full_text += page.extract_text() + "\n"
            except Exception:
                continue
        refs = self._extract_pdf_refs(full_text, ref_limit)
        body = _WHITESPACE.sub(" ", full_text).strip()[:max_chars]
        return body, refs

    def _extract_pdf_refs(self, text: str, ref_limit: int) -> list[str]:
        """Parse references from PDF-extracted text using multiple strategies."""
        ref_section = self._find_pdf_ref_section(text)
        if ref_section is None:
            return []

        # Strategy 1: split by [N] bracket numbering
        entries = _BRACKET_SPLIT.split(ref_section)
        refs = [re.sub(r"\s+", " ", e).strip() for e in entries if e.strip() and len(e.strip()) > 20]

        # Strategy 2: split by N. dot numbering
        if len(refs) <= 1:
            entries = _DOT_SPLIT.split(ref_section)
            refs = [re.sub(r"\s+", " ", e).strip() for e in entries if e.strip() and len(e.strip()) > 20]

        # Strategy 3: split by double newline (paragraph-style references)
        if len(refs) <= 1:
            refs = [
                re.sub(r"\s+", " ", p).strip()
                for p in ref_section.split("\n\n")
                if p.strip() and len(p.strip()) > 20
            ]

        return [r for r in refs if r][:ref_limit]

    def _find_pdf_ref_section(self, text: str) -> str | None:
        """Locate the references section in PDF-extracted text."""
        # Strategy 1: find "References" heading, take everything after the last one
        matches = list(_REF_HEADING.finditer(text))
        for m in reversed(matches):
            after = text[m.end():]
            # Verify this is actually the references section (contains [1] or 1. soon after)
            if _BRACKET_REF.search(after[:500]) or re.search(r'\n\s*1\.\s+', after[:500]):
                return after

        # Strategy 2: find [1] in the last 50% of the text
        search_start = int(len(text) * 0.5)
        bracket_match = _BRACKET_REF.search(text[search_start:])
        if bracket_match:
            return text[search_start + bracket_match.start():]

        # Strategy 3: find "References" heading without verification (last match)
        if matches:
            return text[matches[-1].end():]

        return None

    def _html_to_text_and_refs(
        self, html: str, max_chars: int, ref_limit: int
    ) -> tuple[str, list[str]]:
        body = _HEAD.sub(" ", html)
        body = _SCRIPT.sub(" ", body)
        plain = _WHITESPACE.sub(" ", _TAG.sub(" ", body)).strip()
        if len(plain) > max_chars:
            plain = plain[:max_chars]
        refs = self._extract_bib_entries(html, ref_limit)
        return plain, refs

    def _extract_bib_entries(self, html: str, ref_limit: int) -> list[str]:
        section = _BIB_SECTION.search(html)
        if section is None:
            return []
        entries: list[str] = []
        for m in _BIB_ITEM.finditer(section.group(1)):
            raw = _WHITESPACE.sub(" ", _TAG.sub(" ", m.group(1))).strip()
            if raw:
                entries.append(raw)
            if len(entries) >= ref_limit:
                break
        return entries

    async def aclose(self) -> None:
        await self.html_client.aclose()
        await super().aclose()

    async def _get_xml(self, path: str, **params: Any) -> ET.Element | None:
        key = self._cache_key(path, params)
        xml_text = self.cache.get(key)
        if xml_text is None:
            try:
                resp = await self.breaker.call(self._attempt_get, path, **params)
            except aiobreaker.CircuitBreakerError as exc:
                if self.strict_transient:
                    raise TransientProviderError(
                        self.name, "circuit_open", path
                    ) from exc
                log.warning("circuit_open", provider=self.name, path=redact_secrets(path))
                return None
            except RETRYABLE as exc:
                if self.strict_transient:
                    url = getattr(exc, "url", None) or path
                    raise TransientProviderError(
                        self.name,
                        "retry_exhausted",
                        url,
                        status=getattr(exc, "status", None),
                    ) from exc
                log.warning("request_failed", provider=self.name, path=redact_secrets(path))
                return None
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status != 404:
                    log.warning(
                        "http_error",
                        provider=self.name,
                        path=redact_secrets(path),
                        status=status,
                    )
                    if self.strict_transient:
                        raise TransientProviderError(
                            self.name, "http_error", path, status=status
                        ) from exc
                return None
            if resp.status_code != 200:
                if resp.status_code != 404 and self.strict_transient:
                    raise TransientProviderError(
                        self.name, "http_error", path, status=resp.status_code
                    )
                return None
            xml_text = resp.text
            if not xml_text:
                if self.strict_transient:
                    raise TransientProviderError(
                        self.name, "malformed_response", path, status=200
                    )
                return None
            self.cache.set(key, xml_text, expire=self.cache_ttl)
        root = self._parse_xml(xml_text)
        if root is None:
            if self.strict_transient:
                raise TransientProviderError(
                    self.name, "malformed_response", path, status=200
                )
            return None
        return root

    @staticmethod
    def _parse_xml(xml_text: str) -> ET.Element | None:
        try:
            return ET.fromstring(xml_text)
        except ET.ParseError:
            return None

    def _build_search_query(self, query: str) -> str:
        if _FIELD_PREFIX.match(query):
            return query
        return f"all:{query}"

    def _format_id(self, paper_id: str, id_type: str) -> str:
        if id_type == "auto":
            lower = paper_id.lower()
            if lower.startswith("arxiv:"):
                return paper_id[len("arxiv:"):]
            return paper_id
        return paper_id

    def _find_text(self, entry: ET.Element, tag: str) -> str | None:
        el = entry.find(tag, _NS)
        if el is None or el.text is None:
            return None
        return el.text

    def _clean_text(self, text: str | None) -> str:
        if text is None:
            return ""
        return " ".join(text.split())

    def _extract_id(self, entry: ET.Element) -> str:
        raw = self._find_text(entry, "atom:id") or ""
        aid = raw.split("/abs/")[-1] if "/abs/" in raw else raw
        return _VERSION.sub("", aid)

    def _extract_year(self, entry: ET.Element) -> int | None:
        published = self._find_text(entry, "atom:published")
        if published and len(published) >= 4:
            try:
                return int(published[:4])
            except ValueError:
                return None
        return None

    def _extract_authors(self, entry: ET.Element) -> list[str]:
        authors: list[str] = []
        for au in entry.findall("atom:author", _NS):
            name = au.find("atom:name", _NS)
            if name is not None and name.text:
                authors.append(name.text)
        return authors

    def _extract_categories(self, entry: ET.Element) -> list[str]:
        categories = [c.get("term", "") for c in entry.findall("atom:category", _NS)]
        primary = entry.find("arxiv:primary_category", _NS)
        if primary is not None and primary.get("term"):
            categories = [primary.get("term", ""), *categories]
        return [c for c in categories if c]

    def _parse_summary(self, entry: ET.Element) -> PaperSummary:
        return PaperSummary(
            id=self._extract_id(entry),
            doi=self._find_text(entry, "arxiv:doi"),
            title=self._clean_text(self._find_text(entry, "atom:title")),
            year=self._extract_year(entry),
            authors=self._extract_authors(entry),
            citation_count=None,
            abstract=self._clean_text(self._find_text(entry, "atom:summary")) or None,
            provider="arxiv",
        )

    def _parse_paper(self, entry: ET.Element) -> Paper:
        arxiv_id = self._extract_id(entry)
        doi = self._find_text(entry, "arxiv:doi")
        external_ids: dict[str, str] = {}
        if doi:
            external_ids["DOI"] = doi
        return Paper(
            id=arxiv_id,
            doi=doi or None,
            title=self._clean_text(self._find_text(entry, "atom:title")),
            year=self._extract_year(entry),
            authors=self._extract_authors(entry),
            citation_count=None,
            abstract=self._clean_text(self._find_text(entry, "atom:summary")) or None,
            provider="arxiv",
            external_ids=external_ids,
            fields_of_study=self._extract_categories(entry),
            references=[],
            citations=[],
        )
