"""ReferenceGraphBuilder — shared, paper-level bibliography mapping service.

This service is the single owner of graph construction for full-text papers:
it acquires/accepts the raw bibliography, maps every entry through a dedicated
LLM contract, resolves identities deterministically, and commits canonical or
provisional nodes, edges, aliases, and provenance to the shared ``GraphStore``.

It is single-flight per mapping revision (SQLite job lease + an in-process
lock) and idempotent, so fifteen concurrent agents encountering the same paper
produce one job and one set of LLM batch calls (FRG-6). Narrative integration
stays in the agent and never gates graph construction (FRG-5/FRG-6).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import uuid
from typing import TYPE_CHECKING

from research_explorer.aco.frontier import SharedFrontier
from research_explorer.agents.llm_client import LLMClient
from research_explorer.config import Config, ReferenceMappingConfig, ResolutionConfig
from research_explorer.graph.models import Paper, PaperSummary, normalize_id
from research_explorer.graph.store import GraphStore
from research_explorer.logging_setup import get_logger
from research_explorer.providers.base import ResilientProvider
from research_explorer.redaction import redact_secrets
from research_explorer.references.bibliography import (
    provisional_reference_id,
    raw_entry_hash,
    segment_bibliography,
    sha256_text,
    source_content_hash,
    trim_raw_entry,
)
from research_explorer.references.mapper import BatchMapResult, ReferenceMapper
from research_explorer.references.models import (
    MappedEntry,
    MappingStatus,
    RawBibliographyEntry,
    ReferenceAccounting,
    ResolutionState,
)
from research_explorer.resolution.matcher import MatchThresholds
from research_explorer.resolution.models import BibliographicEntry, RejectReason
from research_explorer.resolution.providers import build_verification_providers
from research_explorer.resolution.resolver import IdentityResolver

if TYPE_CHECKING:
    from research_explorer.replay.trace import RunTracer

log = get_logger("references.builder")

_FULLTEXT_EVIDENCE = "fulltext_bibliography"
_REUSABLE_STATUSES = ("completed", "incomplete", "failed")
_WAIT_POLL_SECONDS = 0.1
_WAIT_MAX_POLLS = 600


def _as_text(raw: object) -> str:
    """Coerce a provider/legacy bibliography entry into raw text."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        for key in ("raw_text", "text", "title"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str)
    return str(raw)


class ReferenceGraphBuilder:
    """Application service that builds a paper's outgoing citation graph once."""

    def __init__(
        self,
        store: GraphStore,
        resolver: IdentityResolver,
        config: ReferenceMappingConfig,
        *,
        mapper: ReferenceMapper,
        frontier: SharedFrontier | None = None,
        visited: set[str] | None = None,
        owner: str | None = None,
    ) -> None:
        self.store = store
        self.resolver = resolver
        self.cfg = config
        self.mapper = mapper
        self.frontier = frontier if frontier is not None else SharedFrontier()
        self.visited = visited if visited is not None else set()
        self.owner = owner or f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.tracer: RunTracer | None = None
        self._locks: dict[str, asyncio.Lock] = {}

    # ---- Public API -------------------------------------------------------

    async def build(
        self,
        paper: Paper,
        raw_entries: list[str] | None = None,
        *,
        tracer: RunTracer | None = None,
    ) -> ReferenceAccounting:
        """Map and commit the full bibliography of ``paper`` exactly once."""
        tracer = tracer if tracer is not None else self.tracer
        raw = list(raw_entries if raw_entries is not None else paper.ref_entries)
        entries = [_as_text(e) for e in raw]
        source_id = self.store.canonical_id(normalize_id(paper.provider, paper.id))
        segmentation_error = getattr(paper, "bibliography_error", None)
        if not entries and not segmentation_error and paper.fulltext:
            entries = segment_bibliography(paper.fulltext)
        if not entries and segmentation_error:
            self._emit(
                tracer,
                "reference_mapping_failed",
                paper_id=source_id,
                error_code=segmentation_error,
            )
            return ReferenceAccounting(
                source_id=source_id, job_id="", status="failed", observed=0
            )
        observed = len(entries)
        if observed == 0:
            self._emit(tracer, "reference_mapping_completed", paper_id=source_id, observed=0)
            return ReferenceAccounting(
                source_id=source_id, job_id="", status="completed", observed=0
            )

        content_hash = source_content_hash(entries)
        job_id = _job_id(source_id, content_hash, self.cfg.mapper_version, self.mapper.prompt_hash)
        self.store.ensure_mapping_job(
            job_id,
            source_id,
            content_hash,
            self.cfg.mapper_version,
            self.mapper.prompt_hash,
        )

        lock = self._locks.setdefault(job_id, asyncio.Lock())
        async with lock:
            job = self.store.get_mapping_job(job_id)
            if job is not None and job["status"] in _REUSABLE_STATUSES:
                accounting = self._account(job_id, source_id, reused=True)
                self._emit(
                    tracer,
                    "reference_mapping_reused",
                    paper_id=source_id,
                    job_id=job_id,
                    status=accounting.status,
                    observed=accounting.observed,
                    processed=accounting.processed,
                    mapped=accounting.mapped,
                    unparsed=accounting.unparsed,
                    resolved=accounting.resolved,
                    provisional=accounting.provisional,
                    failed=accounting.failed,
                    traversable=accounting.traversable,
                )
                return accounting

            if not self.store.claim_mapping_job(job_id, self.owner, self.cfg.lease_seconds):
                # Another worker owns the lease. Wait for its terminal result, or
                # for the lease to expire so this worker can take over. Never map
                # without owning the lease (FRG-6).
                while True:
                    job = await self._wait_for_terminal(job_id)
                    if job is not None and job["status"] in _REUSABLE_STATUSES:
                        accounting = self._account(job_id, source_id, reused=True)
                        self._emit(
                            tracer,
                            "reference_mapping_reused",
                            paper_id=source_id,
                            job_id=job_id,
                            status=accounting.status,
                            observed=accounting.observed,
                            mapped=accounting.mapped,
                            resolved=accounting.resolved,
                            provisional=accounting.provisional,
                            failed=accounting.failed,
                            traversable=accounting.traversable,
                        )
                        return accounting
                    if self.store.claim_mapping_job(
                        job_id, self.owner, self.cfg.lease_seconds
                    ):
                        break

            self._emit(
                tracer,
                "reference_mapping_started",
                paper_id=source_id,
                job_id=job_id,
                observed=observed,
            )
            self._persist_raw_entries(job_id, source_id, entries)
            await self._map_pending(job_id, tracer)
            await self._resolve_pending(job_id, source_id, content_hash, tracer)
            status = self._finalize(job_id, observed)
            self._emit(
                tracer,
                "reference_graph_committed",
                paper_id=source_id,
                job_id=job_id,
                status=status,
            )
            accounting = self._account(job_id, source_id, reused=False)
            self._emit(
                tracer,
                "reference_mapping_completed",
                paper_id=source_id,
                job_id=job_id,
                status=accounting.status,
                observed=accounting.observed,
                processed=accounting.processed,
                mapped=accounting.mapped,
                unparsed=accounting.unparsed,
                resolved=accounting.resolved,
                provisional=accounting.provisional,
                failed=accounting.failed,
                traversable=accounting.traversable,
            )
            return accounting

    # ---- Persistence ------------------------------------------------------

    def _persist_raw_entries(self, job_id: str, source_id: str, entries: list[str]) -> None:
        records = []
        for ordinal, raw in enumerate(entries, 1):
            entry_hash = raw_entry_hash(raw)
            records.append(
                {
                    "id": f"{job_id}:{ordinal:05d}",
                    "ordinal": ordinal,
                    "raw_text": trim_raw_entry(raw),
                    "raw_hash": entry_hash,
                }
            )
        self.store.upsert_bibliography_entries(job_id, source_id, records)
        self.store.update_mapping_job(job_id, entry_count=len(records), status="running")

    # ---- Mapping ----------------------------------------------------------

    async def _map_pending(self, job_id: str, tracer: RunTracer | None) -> None:
        max_entries = self.cfg.max_entries
        attempt = 0
        while attempt <= self.cfg.max_retries:
            pending = [
                r
                for r in self.store.get_bibliography_entries(job_id)
                if r["mapping_status"] == MappingStatus.PENDING.value
            ]
            if max_entries > 0:
                pending = [r for r in pending if r["ordinal"] <= max_entries]
            if not pending:
                break
            raw_models = [
                RawBibliographyEntry(
                    entry_id=r["id"],
                    ordinal=r["ordinal"],
                    raw_text=r["raw_text"],
                    raw_hash=r["raw_hash"],
                )
                for r in pending
            ]
            batches = self.mapper.build_batches(raw_models)
            limit = max(1, self.cfg.max_concurrent_batches)
            if limit == 1 or len(batches) == 1:
                for batch in batches:
                    await self._run_batch(job_id, batch, tracer)
            else:
                semaphore = asyncio.Semaphore(limit)

                async def run(
                    batch: list[RawBibliographyEntry],
                    sem: asyncio.Semaphore = semaphore,
                ) -> None:
                    async with sem:
                        await self._run_batch(job_id, batch, tracer)

                await asyncio.gather(*(run(batch) for batch in batches))
            attempt += 1

        remaining = [
            r
            for r in self.store.get_bibliography_entries(job_id)
            if r["mapping_status"] == MappingStatus.PENDING.value
        ]
        for row in remaining:
            self.store.update_bibliography_entry(
                row["id"],
                mapping_status=MappingStatus.FAILED.value,
                error_code="retry_exhausted",
            )

    async def _run_batch(
        self, job_id: str, batch: list[RawBibliographyEntry], tracer: RunTracer | None
    ) -> None:
        self._emit(
            tracer,
            "reference_batch_started",
            job_id=job_id,
            batch_size=len(batch),
            first_ordinal=batch[0].ordinal,
        )
        try:
            result = await self.mapper.map_batch(batch)
        except Exception as exc:  # transport/schema failure: retry next pass
            log.warning(
                "reference_batch_failed", job_id=job_id, error=redact_secrets(str(exc))
            )
            for entry in batch:
                self._bump_attempt(entry.entry_id, "transport_error")
            self._emit(
                tracer,
                "reference_batch_failed",
                job_id=job_id,
                error_code="transport_error",
                batch_size=len(batch),
            )
            return
        self._persist_batch(job_id, batch, result, tracer)
        self._emit(
            tracer,
            "reference_batch_completed",
            job_id=job_id,
            mapped=len(result.results),
            failed=len(result.failed_entry_ids),
            malformed=result.malformed,
        )

    def _persist_batch(
        self,
        job_id: str,
        batch: list[RawBibliographyEntry],
        result: BatchMapResult,
        tracer: RunTracer | None,
    ) -> None:
        for entry in batch:
            mapped = result.results.get(entry.entry_id)
            if mapped is None:
                self._bump_attempt(entry.entry_id, "missing_from_response")
                continue
            self._store_mapped(entry.entry_id, mapped)
            event = (
                "reference_entry_unparsed"
                if mapped.mapping_status is MappingStatus.UNPARSED
                else "reference_entry_mapped"
            )
            self._emit(
                tracer,
                event,
                job_id=job_id,
                entry_id=entry.entry_id,
                ordinal=entry.ordinal,
                confidence=round(mapped.parse_confidence, 4),
            )

    def _store_mapped(self, entry_id: str, mapped: MappedEntry) -> None:
        self.store.update_bibliography_entry(
            entry_id,
            entry_type=mapped.entry_type.value,
            title=mapped.title,
            authors=json.dumps(mapped.authors),
            year=mapped.year,
            venue=mapped.venue,
            volume=mapped.volume,
            issue=mapped.issue,
            pages=mapped.pages,
            doi=mapped.doi,
            arxiv_id=mapped.arxiv_id,
            pmid=mapped.pmid,
            parse_confidence=mapped.parse_confidence,
            parse_notes=mapped.parse_notes,
            mapping_status=mapped.mapping_status.value,
            resolution_status=ResolutionState.PENDING.value,
        )

    def _bump_attempt(self, entry_id: str, error_code: str) -> None:
        row = self.store.get_bibliography_entry(entry_id)
        attempts = int(row["attempt_count"]) + 1 if row else 1
        self.store.update_bibliography_entry(
            entry_id, attempt_count=attempts, error_code=error_code
        )

    # ---- Resolution + commit ---------------------------------------------

    async def _resolve_pending(
        self, job_id: str, source_id: str, content_hash: str, tracer: RunTracer | None
    ) -> None:
        rows = [
            r
            for r in self.store.get_bibliography_entries(job_id)
            if r["mapping_status"] == MappingStatus.MAPPED.value
            and r["resolution_status"]
            in (None, ResolutionState.PENDING.value, ResolutionState.RETRYABLE.value)
        ]
        for row in rows:
            confidence = float(row["parse_confidence"] or 0.0)
            prov_id = provisional_reference_id(content_hash, row["raw_hash"])
            if confidence < self.cfg.min_parse_confidence:
                self._mark_provisional(
                    row, source_id, prov_id, ResolutionState.PROVISIONAL, tracer,
                    reason=RejectReason.LOW_CONFIDENCE.value,
                )
                continue
            entry = BibliographicEntry(
                title=row["title"] or "",
                authors=json.loads(row["authors"]) if row["authors"] else [],
                year=row["year"],
                doi=row["doi"],
                arxiv_id=row["arxiv_id"],
                pmid=row["pmid"],
            )
            result = await self.resolver.resolve(entry)
            self._record_attempts(row["id"], result)
            if result.accepted and result.canonical_id:
                canonical = self.store.canonical_id(result.canonical_id)
                self.store.record_edge(
                    source_id,
                    canonical,
                    result.provider or "resolution",
                    "references",
                    evidence_type=_FULLTEXT_EVIDENCE,
                    source_id=source_id,
                    entry_id=row["id"],
                    mapping_revision=job_id,
                    method=result.method.value if result.method else None,
                    confidence=result.confidence,
                )
                self.store.commit()
                if self.store.get_provisional_node(prov_id) is not None:
                    self.store.add_alias(prov_id, canonical, "provisional")
                self.store.update_bibliography_entry(
                    row["id"],
                    resolution_status=ResolutionState.RESOLVED.value,
                    canonical_id=canonical,
                    resolution_confidence=result.confidence,
                )
                self._publish(canonical, source_id)
                if tracer is not None:
                    tracer.emit(
                        "reference_resolved",
                        job_id=job_id,
                        entry_id=row["id"],
                        canonical_id=canonical,
                        method=result.method.value if result.method else None,
                        confidence=round(result.confidence, 4),
                    )
            else:
                retryable = result.reject_reason is RejectReason.PROVIDER_UNAVAILABLE
                state = (
                    ResolutionState.RETRYABLE if retryable else ResolutionState.PROVISIONAL
                )
                self._mark_provisional(
                    row, source_id, prov_id, state, tracer,
                    reason=result.reject_reason.value if result.reject_reason else None,
                )

    def _mark_provisional(
        self,
        row: dict,
        source_id: str,
        prov_id: str,
        state: ResolutionState,
        tracer: RunTracer | None,
        *,
        reason: str | None = None,
    ) -> None:
        if self.cfg.allow_provisional_nodes:
            self.store.add_provisional_node(
                prov_id,
                source_id,
                row["raw_hash"],
                title=row["title"],
                authors=json.loads(row["authors"]) if row["authors"] else [],
                year=row["year"],
                raw_text=row["raw_text"],
            )
            self.store.record_provisional_edge(source_id, prov_id, row["id"])
            self.store.cache_summary_for_nid(
                prov_id,
                PaperSummary(
                    id=prov_id,
                    title=row["title"] or row["raw_text"][:80],
                    year=row["year"],
                    authors=json.loads(row["authors"]) if row["authors"] else [],
                    provider="provisional",
                ),
            )
        self.store.update_bibliography_entry(
            row["id"],
            resolution_status=state.value,
            canonical_id=None,
            resolution_confidence=row["parse_confidence"],
        )
        self._emit(
            tracer,
            "reference_provisional",
            entry_id=row["id"],
            state=state.value,
            reason=reason,
        )

    def _record_attempts(self, entry_id: str, result) -> None:
        expected = json.dumps(result.expected.model_dump(), default=str)
        for attempt in result.attempts:
            self.store.record_resolution_attempt(
                entry_id,
                method=attempt.method.value if attempt.method else None,
                provider=attempt.provider,
                candidate_id=attempt.canonical_id,
                status=attempt.status.value,
                confidence=result.confidence,
                reject_reason=attempt.reject_reason.value if attempt.reject_reason else None,
                expected_json=expected,
            )
        if result.actual is not None:
            self.store.record_resolution_attempt(
                entry_id,
                method=result.method.value if result.method else None,
                provider=result.provider,
                candidate_id=result.canonical_id,
                status=result.status.value,
                confidence=result.confidence,
                reject_reason=result.reject_reason.value if result.reject_reason else None,
                expected_json=expected,
                actual_json=json.dumps(result.actual.model_dump(), default=str),
            )

    def _publish(self, canonical: str, source_id: str) -> None:
        self.frontier.add([canonical], source=source_id, mode="ref", exclude=self.visited)

    # ---- Accounting -------------------------------------------------------

    def _finalize(self, job_id: str, observed: int) -> str:
        rows = self.store.get_bibliography_entries(job_id)
        mapped = sum(1 for r in rows if r["mapping_status"] == MappingStatus.MAPPED.value)
        unparsed = sum(
            1 for r in rows if r["mapping_status"] == MappingStatus.UNPARSED.value
        )
        failed = sum(
            1 for r in rows if r["mapping_status"] == MappingStatus.FAILED.value
        )
        pending = sum(
            1 for r in rows if r["mapping_status"] == MappingStatus.PENDING.value
        )
        retryable = sum(
            1 for r in rows if r["resolution_status"] == ResolutionState.RETRYABLE.value
        )
        traversable = len(
            {
                self.store.canonical_id(r["canonical_id"])
                for r in rows
                if r["resolution_status"] == ResolutionState.RESOLVED.value
                and r["canonical_id"]
            }
        )
        max_entries = self.cfg.max_entries
        incomplete = max_entries > 0 and observed > max_entries
        if observed == 0:
            status = "completed"
        elif incomplete:
            status = "incomplete"
        elif pending > 0:
            status = "partial"
        elif mapped == 0:
            status = "failed"
        elif retryable > 0:
            status = "partial"
        else:
            status = "completed"
        self.store.update_mapping_job(
            job_id,
            status=status,
            mapped_count=mapped,
            unparsed_count=unparsed,
            failed_count=failed,
            resolved_count=sum(
                1 for r in rows if r["resolution_status"] == ResolutionState.RESOLVED.value
            ),
            provisional_count=sum(
                1
                for r in rows
                if r["resolution_status"]
                in (
                    ResolutionState.PROVISIONAL.value,
                    ResolutionState.RETRYABLE.value,
                    ResolutionState.UNRESOLVED.value,
                )
            ),
            traversable_count=traversable,
        )
        return status

    def _account(self, job_id: str, source_id: str, *, reused: bool) -> ReferenceAccounting:
        job = self.store.get_mapping_job(job_id) or {}
        rows = self.store.get_bibliography_entries(job_id)
        resolved_ids: list[str] = []
        for row in rows:
            if (
                row["resolution_status"] == ResolutionState.RESOLVED.value
                and row["canonical_id"]
            ):
                canonical = self.store.canonical_id(row["canonical_id"])
                if canonical not in resolved_ids:
                    resolved_ids.append(canonical)
        summaries = [
            summary
            for cid in resolved_ids
            if (summary := self.store.get_paper_summary(cid)) is not None
        ]
        observed = int(job.get("entry_count") or len(rows))
        mapped = int(job.get("mapped_count") or 0)
        unparsed = int(job.get("unparsed_count") or 0)
        failed = int(job.get("failed_count") or 0)
        return ReferenceAccounting(
            source_id=source_id,
            job_id=job_id,
            status=str(job.get("status") or "completed"),
            reused=reused,
            observed=observed,
            processed=mapped + unparsed + failed,
            mapped=mapped,
            unparsed=unparsed,
            failed=failed,
            resolved=int(job.get("resolved_count") or 0),
            provisional=int(job.get("provisional_count") or 0),
            traversable=int(job.get("traversable_count") or len(resolved_ids)),
            resolved_summaries=summaries,
        )

    # ---- Waiting / events -------------------------------------------------

    async def _wait_for_terminal(self, job_id: str) -> dict | None:
        for _ in range(_WAIT_MAX_POLLS):
            job = self.store.get_mapping_job(job_id)
            if job is not None and job["status"] in _REUSABLE_STATUSES:
                return job
            await asyncio.sleep(_WAIT_POLL_SECONDS)
        return self.store.get_mapping_job(job_id)

    def _emit(self, tracer: RunTracer | None, event: str, **payload) -> None:
        if tracer is not None:
            with contextlib.suppress(Exception):
                tracer.emit(event, **payload)


def _job_id(
    source_id: str, content_hash: str, mapper_version: str, prompt_hash: str
) -> str:
    key = f"{source_id}|{content_hash}|{mapper_version}|{prompt_hash}"
    return "map-" + sha256_text(key)[:24]


def build_reference_builder(
    config: Config,
    providers: dict[str, ResilientProvider],
    store: GraphStore,
    llm: LLMClient,
    *,
    frontier: SharedFrontier | None = None,
    visited: set[str] | None = None,
) -> ReferenceGraphBuilder | None:
    """Build the shared reference builder when reference mapping is enabled."""
    mapping = getattr(config, "reference_mapping", None)
    if mapping is None or not mapping.enabled:
        return None
    resolution: ResolutionConfig = config.resolution
    order = [resolution.primary, *resolution.fallbacks]
    available = [name for name in order if name in providers]
    verification = build_verification_providers(
        available, providers, title_limit=resolution.title_search_limit
    )
    resolver = IdentityResolver(
        providers=verification,
        store=store,
        thresholds=MatchThresholds.from_config(resolution),
    )
    model = mapping.model or config.llm.explorer_model
    mapper = ReferenceMapper(llm, mapping, model)
    return ReferenceGraphBuilder(
        store,
        resolver,
        mapping,
        mapper=mapper,
        frontier=frontier,
        visited=visited,
    )
