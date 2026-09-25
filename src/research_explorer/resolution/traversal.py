"""Canonical neighbor expansion for graph traversal.

OpenAlex is the primary expansion source for both directions — incoming
cited-by via the `cited_by:<id>` filter and outgoing via `referenced_works`
(batch-hydrated) — with Semantic Scholar as fallback and enrichment. Every
candidate neighbor is verified through the IdentityResolver before it enters
the graph: only verified candidates get canonical ids, aliases, edges with
provenance, and frontier entries; rejections never touch the graph.

Expansion works purely on provider metadata, independent of full text, which
is what allows metadata-only nodes to remain expandable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from research_explorer.config import Config, ResolutionConfig
from research_explorer.graph.models import Paper, PaperSummary, parse_normalized_id
from research_explorer.graph.store import GraphStore
from research_explorer.logging_setup import get_logger
from research_explorer.providers.base import ResilientProvider
from research_explorer.replay.trace import RunTracer
from research_explorer.resolution.matcher import MatchThresholds
from research_explorer.resolution.models import (
    BibliographicEntry,
    EntrySource,
    RejectReason,
)
from research_explorer.resolution.providers import build_verification_providers
from research_explorer.resolution.resolver import (
    IdentityResolver,
    candidate_dedup_key,
    normalize_arxiv,
)

log = get_logger("resolution.traversal")

INCOMING = "incoming"
OUTGOING = "outgoing"
ARXIV_DOI_PREFIX = "10.48550/arxiv."


class DirectionExpansion(BaseModel):
    """Outcome of expanding one direction of one node."""

    direction: str
    provider: str | None = None
    fallback_used: bool = False
    discovered: int = 0
    rejected: int = 0
    node_ids: list[str] = Field(default_factory=list)


class ExpansionResult(BaseModel):
    """Outcome of expanding both directions of one node."""

    node_id: str
    incoming: DirectionExpansion
    outgoing: DirectionExpansion


class NeighborExpander:
    """Provider-verified, cross-provider deduplicated neighbor expansion."""

    def __init__(
        self,
        providers: dict[str, ResilientProvider],
        resolver: IdentityResolver,
        store: GraphStore,
        config: ResolutionConfig,
    ):
        self.providers = providers
        self.resolver = resolver
        self.store = store
        self.config = config
        self._order = [config.primary, *config.fallbacks]

    async def expand(
        self,
        node_id: str,
        paper: Paper | None = None,
        extracted: list[PaperSummary] | None = None,
        tracer: RunTracer | None = None,
    ) -> ExpansionResult:
        """Expand both directions of node_id; records verified edges in the store."""
        incoming = await self._expand(node_id, INCOMING, paper, [], tracer)
        outgoing = await self._expand(node_id, OUTGOING, paper, extracted or [], tracer)
        return ExpansionResult(node_id=node_id, incoming=incoming, outgoing=outgoing)

    def _fetchable(self) -> list[str]:
        return [name for name in self._order if name in self.providers]

    def _limit(self, direction: str) -> int:
        if direction == INCOMING:
            enabled, limit = self.config.incoming_enabled, self.config.incoming_limit
        else:
            enabled, limit = self.config.outgoing_enabled, self.config.outgoing_limit
        return limit if enabled else 0

    async def _expand(
        self,
        node_id: str,
        direction: str,
        paper: Paper | None,
        extracted: list[PaperSummary],
        tracer: RunTracer | None,
    ) -> DirectionExpansion:
        limit = self._limit(direction)
        if limit <= 0:
            return DirectionExpansion(direction=direction)

        summary = self.store.get_paper_summary(node_id)
        origin_provider, native = parse_normalized_id(node_id)

        candidates: list[PaperSummary] = []
        used_provider: str | None = None
        fallback_used = False
        fetched = await self._fetch_from_providers(
            node_id, direction, origin_provider, native, summary, limit, tracer
        )
        if fetched is not None:
            source_name, source_candidates, source_fallback = fetched
            candidates.extend(source_candidates[:limit])
            used_provider = source_name
            fallback_used = source_fallback
        if direction == OUTGOING:
            if paper is not None and paper.references:
                candidates.extend(paper.references[:limit])
            candidates.extend(extracted[:limit])
        elif paper is not None and paper.citations:
            candidates.extend(paper.citations[:limit])

        candidates = await self._hydrate_openalex_stubs(candidates)
        candidates = self._dedupe_raw(candidates)

        resolved, discovered, rejected = await self._resolve_candidates(
            node_id, direction, candidates, limit, tracer
        )
        attribution = used_provider
        if attribution is None and paper is not None and (paper.references or paper.citations):
            attribution = paper.provider

        for canonical, _candidate in resolved:
            if direction == OUTGOING:
                self.store.record_edge(
                    node_id, canonical, attribution or "resolution", "references"
                )
            else:
                self.store.record_edge(
                    canonical, node_id, attribution or "resolution", "cited_by"
                )
        self.store.commit()

        if tracer is not None:
            tracer.emit(
                "neighbors_expanded",
                node_id=node_id,
                direction=direction,
                provider=attribution,
                discovered=discovered,
                resolved=len(resolved),
                rejected=rejected,
                fallback_used=fallback_used,
            )
        return DirectionExpansion(
            direction=direction,
            provider=attribution,
            fallback_used=fallback_used,
            discovered=discovered,
            rejected=rejected,
            node_ids=[canonical for canonical, _ in resolved],
        )

    def _dedupe_raw(self, candidates: list[PaperSummary]) -> list[PaperSummary]:
        seen: set[str] = set()
        unique: list[PaperSummary] = []
        for candidate in candidates:
            key = candidate_dedup_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    async def _fetch_from_providers(
        self,
        node_id: str,
        direction: str,
        origin_provider: str,
        native: str,
        summary: PaperSummary | None,
        limit: int,
        tracer: RunTracer | None,
    ) -> tuple[str, list[PaperSummary], bool] | None:
        names = self._fetchable()
        for index, name in enumerate(names):
            provider = self.providers[name]
            candidates = await self._fetch_direction(
                provider, direction, origin_provider, native, summary, limit
            )
            if candidates:
                return name, candidates, name != self.config.primary
            if tracer is not None and index < len(names) - 1:
                tracer.emit(
                    "provider_fallback",
                    node_id=node_id,
                    direction=direction,
                    from_provider=name,
                    to_provider=names[index + 1],
                    reason="empty" if candidates is not None else "unavailable",
                )
            log.debug(
                "expansion_source_skipped",
                provider=name,
                direction=direction,
                node=node_id,
                reason="empty" if candidates is not None else "unavailable",
            )
        return None

    async def _fetch_direction(
        self,
        provider: ResilientProvider,
        direction: str,
        origin_provider: str,
        native: str,
        summary: PaperSummary | None,
        limit: int,
    ) -> list[PaperSummary] | None:
        try:
            if direction == INCOMING:
                return await self._fetch_incoming(
                    provider, origin_provider, native, summary, limit
                )
            return await self._fetch_outgoing(
                provider, origin_provider, native, summary, limit
            )
        except Exception as exc:
            log.warning(
                "expansion_fetch_failed",
                provider=provider.name,
                direction=direction,
                error=str(exc),
            )
            return None

    async def _fetch_incoming(
        self,
        provider: ResilientProvider,
        origin_provider: str,
        native: str,
        summary: PaperSummary | None,
        limit: int,
    ) -> list[PaperSummary] | None:
        key = await self._direction_key(provider, origin_provider, native, summary)
        if key is None:
            return None
        if provider.name == "openalex":
            openalex_id = await self._openalex_id(provider, key)
            if openalex_id is None:
                return None
            return await provider.get_citations(openalex_id, limit=limit)
        return await provider.get_citations(key, limit=limit)

    async def _fetch_outgoing(
        self,
        provider: ResilientProvider,
        origin_provider: str,
        native: str,
        summary: PaperSummary | None,
        limit: int,
    ) -> list[PaperSummary] | None:
        key = await self._direction_key(provider, origin_provider, native, summary)
        if key is None:
            return None
        if provider.name == "openalex":
            openalex_id = await self._openalex_id(provider, key)
            if openalex_id is None:
                return None
            return await provider.get_references(openalex_id, limit=limit)
        return await provider.get_references(key, limit=limit)

    async def _direction_key(
        self,
        provider: ResilientProvider,
        origin_provider: str,
        native: str,
        summary: PaperSummary | None,
    ) -> str | None:
        if provider.name == origin_provider:
            return native
        if summary is None:
            return None
        if provider.name == "semantic_scholar":
            if summary.doi:
                return f"DOI:{summary.doi}"
            if summary.arxiv_id:
                return f"arXiv:{normalize_arxiv(summary.arxiv_id)}"
            return None
        if provider.name == "openalex":
            if summary.doi:
                return summary.doi
            if summary.arxiv_id:
                return f"{ARXIV_DOI_PREFIX}{normalize_arxiv(summary.arxiv_id)}"
            return None
        return None

    async def _openalex_id(self, provider: ResilientProvider, key: str) -> str | None:
        if key.startswith("W"):
            return key
        paper = await provider.get_paper(key)
        return paper.id if paper is not None else None

    async def _hydrate_openalex_stubs(
        self, candidates: list[PaperSummary]
    ) -> list[PaperSummary]:
        openalex = self.providers.get("openalex")
        if openalex is None or not hasattr(openalex, "get_works_batch"):
            return candidates
        stub_ids = [c.id for c in candidates if c.provider == "openalex" and not c.title]
        if not stub_ids:
            return candidates
        hydrated = await openalex.get_works_batch(stub_ids, limit=len(stub_ids))
        by_id = {h.id: h for h in hydrated}
        return [by_id.get(c.id, c) if c.provider == "openalex" else c for c in candidates]

    async def _resolve_candidates(
        self,
        node_id: str,
        direction: str,
        candidates: list[PaperSummary],
        limit: int,
        tracer: RunTracer | None,
    ) -> tuple[list[tuple[str, PaperSummary]], int, int]:
        seen: set[str] = set()
        resolved: list[tuple[str, PaperSummary]] = []
        discovered = 0
        rejected = 0
        for candidate in candidates:
            discovered += 1
            entry = self._entry_from_candidate(candidate)
            if not self._resolvable(entry):
                rejected += 1
                if tracer is not None:
                    tracer.emit(
                        "resolution_started",
                        node_id=node_id,
                        direction=direction,
                        title=entry.title,
                        doi=entry.doi,
                        arxiv_id=entry.arxiv_id,
                        provider=candidate.provider,
                    )
                    tracer.emit(
                        "resolution_rejected",
                        node_id=node_id,
                        direction=direction,
                        title=entry.title,
                        provider=candidate.provider,
                        reason=RejectReason.UNVERIFIABLE_IDENTIFIER.value,
                    )
                continue
            if tracer is not None:
                tracer.emit(
                    "resolution_started",
                    node_id=node_id,
                    direction=direction,
                    title=entry.title,
                    doi=entry.doi,
                    arxiv_id=entry.arxiv_id,
                    provider=candidate.provider,
                )
            result = await self.resolver.resolve(entry)
            if not result.accepted or result.canonical_id is None:
                rejected += 1
                if tracer is not None:
                    tracer.emit(
                        "resolution_rejected",
                        node_id=node_id,
                        direction=direction,
                        title=entry.title,
                        provider=result.provider,
                        reason=result.reject_reason.value if result.reject_reason else None,
                    )
                continue
            canonical = result.canonical_id
            if canonical == node_id or canonical in seen:
                continue
            seen.add(canonical)
            resolved.append((canonical, candidate))
            if tracer is not None:
                tracer.emit(
                    "resolution_resolved",
                    node_id=node_id,
                    direction=direction,
                    canonical_id=canonical,
                    method=result.method.value if result.method else None,
                    provider=result.provider,
                    confidence=round(result.confidence, 4),
                )
            if len(resolved) >= limit:
                break
        return resolved, discovered, rejected

    def _resolvable(self, entry: BibliographicEntry) -> bool:
        """A candidate needs at least one piece of evidence to verify against."""
        return bool(entry.title or entry.doi or entry.arxiv_id)

    def _entry_from_candidate(self, candidate: PaperSummary) -> BibliographicEntry:
        arxiv_id = candidate.arxiv_id
        if arxiv_id is None and candidate.provider == "arxiv":
            arxiv_id = candidate.id
        return BibliographicEntry(
            title=candidate.title,
            authors=candidate.authors,
            year=candidate.year,
            doi=candidate.doi,
            arxiv_id=arxiv_id,
            source=EntrySource.PROVIDER,
        )


def build_neighbor_expander(
    config: Config,
    providers: dict[str, ResilientProvider],
    store: GraphStore,
) -> NeighborExpander | None:
    """Build the expander when resolution is enabled and a source is available."""
    res = getattr(config, "resolution", None)
    if res is None or not res.enabled:
        return None
    order = [res.primary, *res.fallbacks]
    available = [name for name in order if name in providers]
    if not available:
        return None
    verification = build_verification_providers(
        available, providers, title_limit=res.title_search_limit
    )
    if not verification:
        return None
    resolver = IdentityResolver(
        providers=verification,
        store=store,
        thresholds=MatchThresholds.from_config(res),
    )
    return NeighborExpander(
        providers=providers, resolver=resolver, store=store, config=res
    )
