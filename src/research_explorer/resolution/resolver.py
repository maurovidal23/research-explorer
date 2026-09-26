"""Deterministic identity resolution for bibliographic entries.

The resolver is the provider verification contract: providers implement
`VerificationProvider` (DOI lookup, arXiv lookup, title search) and the
resolver deterministically verifies, scores, tie-breaks, and dedups whatever
candidates they return. No network and no LLM logic lives here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from research_explorer.graph.models import PaperSummary, normalize_id
from research_explorer.graph.store import GraphStore
from research_explorer.logging_setup import get_logger
from research_explorer.resolution.matcher import (
    MatchThresholds,
    hard_gate_failures,
    normalize_title,
    score_candidate,
    surname,
)
from research_explorer.resolution.models import (
    BibliographicEntry,
    RecordMetadata,
    RejectReason,
    ResolutionAttempt,
    ResolutionEvidence,
    ResolutionMethod,
    ResolutionResult,
    ResolutionStatus,
    record_from_summary,
)

log = get_logger("resolution")

DOI_ALIAS_PREFIX = "doi:"
ARXIV_ALIAS_PREFIX = "arxiv:"

_DOI_URL_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi.org/",
    "doi:",
    "info:doi/",
)

_ARXIV_URL_PREFIXES = (
    "https://arxiv.org/abs/",
    "http://arxiv.org/abs/",
    "arxiv.org/abs/",
    "arxiv:",
    "arXiv:",
)

_ARXIV_VERSION_SUFFIX = re.compile(r"v\d+$", re.IGNORECASE)
_ARXIV_NEW_STYLE = re.compile(r"^\d{4}\.\d{4,5}$")
_ARXIV_OLD_STYLE = re.compile(r"^[a-z-]+(?:\.[a-z-]+)?/\d{7}$")
_DOI_SHAPE = re.compile(r"^10\.\d{4,9}/\S+$")

_REJECT_PRIORITY: tuple[RejectReason, ...] = (
    RejectReason.UNVERIFIABLE_IDENTIFIER,
    RejectReason.TITLE_MISMATCH,
    RejectReason.AUTHOR_MISMATCH,
    RejectReason.YEAR_MISMATCH,
    RejectReason.LOW_CONFIDENCE,
    RejectReason.MULTIPLE_CANDIDATES,
    RejectReason.NO_PROVIDER_MATCH,
    RejectReason.PROVIDER_UNAVAILABLE,
)


def normalize_doi(raw: str) -> str:
    """Normalize a DOI to lowercase bare form (whitespace and prefix free)."""
    text = re.sub(r"\s+", "", raw)
    changed = True
    while changed:
        changed = False
        lowered = text.lower()
        for prefix in _DOI_URL_PREFIXES:
            if lowered.startswith(prefix):
                text = text[len(prefix):]
                changed = True
                break
    return text.lower()


def is_valid_doi(doi: str) -> bool:
    """True when the normalized DOI has the required '10.<prefix>/<suffix>' shape."""
    return bool(_DOI_SHAPE.match(doi))


def normalize_arxiv(raw: str) -> str:
    """Normalize an arXiv identifier: strip URLs/prefixes and version suffixes."""
    text = re.sub(r"\s+", "", raw)
    changed = True
    while changed:
        changed = False
        lowered = text.lower()
        for prefix in _ARXIV_URL_PREFIXES:
            if lowered.startswith(prefix):
                text = text[len(prefix):]
                changed = True
                break
    text = _ARXIV_VERSION_SUFFIX.sub("", text)
    return text.lower()


def is_valid_arxiv(arxiv_id: str) -> bool:
    """True for new-style (2301.00001) or old-style (cs/0112017) arXiv ids."""
    stripped = _ARXIV_VERSION_SUFFIX.sub("", arxiv_id)
    return bool(_ARXIV_NEW_STYLE.match(stripped) or _ARXIV_OLD_STYLE.match(stripped))


def alias_key_for_doi(doi: str) -> str | None:
    normalized = normalize_doi(doi)
    return f"{DOI_ALIAS_PREFIX}{normalized}" if is_valid_doi(normalized) else None


def alias_key_for_arxiv(arxiv_id: str) -> str | None:
    normalized = normalize_arxiv(arxiv_id)
    return f"{ARXIV_ALIAS_PREFIX}{normalized}" if is_valid_arxiv(normalized) else None


def alias_keys_for_entry(entry: BibliographicEntry) -> list[str]:
    keys = []
    if entry.doi:
        key = alias_key_for_doi(entry.doi)
        if key:
            keys.append(key)
    if entry.arxiv_id:
        key = alias_key_for_arxiv(entry.arxiv_id)
        if key:
            keys.append(key)
    return keys


def alias_keys_for_summary(summary: PaperSummary) -> list[str]:
    keys = []
    if summary.doi:
        key = alias_key_for_doi(summary.doi)
        if key:
            keys.append(key)
    if summary.arxiv_id:
        key = alias_key_for_arxiv(summary.arxiv_id)
        if key:
            keys.append(key)
    return keys


def candidate_dedup_key(summary: PaperSummary) -> str:
    """Stable identity key used to collapse same-work candidates."""
    if summary.doi:
        return alias_key_for_doi(summary.doi) or f"doi:{normalize_doi(summary.doi)}"
    if summary.arxiv_id:
        key = alias_key_for_arxiv(summary.arxiv_id)
        if key:
            return key
    if not summary.title:
        return normalize_id(summary.provider, summary.id)
    head = min((surname(a) for a in summary.authors if surname(a)), default="")
    return f"title:{normalize_title(summary.title)}:{head}:{summary.year or ''}"


def metadata_completeness(summary: PaperSummary) -> int:
    value = 0
    if summary.title:
        value += 1
    if summary.doi:
        value += 1
    if summary.arxiv_id:
        value += 1
    if summary.year is not None:
        value += 1
    if summary.authors:
        value += 1
    if summary.abstract:
        value += 1
    return value


class VerificationProvider(Protocol):
    """Contract every identity source must satisfy.

    Implementations are pure adapters over a data source: they return candidate
    records (or None / empty lists) and may raise on transport failure. The
    resolver owns all verification, tie-breaking, and alias bookkeeping.
    """

    name: str

    async def lookup_doi(self, doi: str) -> PaperSummary | None: ...

    async def lookup_arxiv(self, arxiv_id: str) -> PaperSummary | None: ...

    async def search_title(self, title: str) -> list[PaperSummary]: ...


class IdentityResolver:
    """Verify bibliographic entries against providers and assign canonical ids."""

    def __init__(
        self,
        providers: Sequence[VerificationProvider] = (),
        store: GraphStore | None = None,
        thresholds: MatchThresholds | None = None,
    ) -> None:
        self.providers = list(providers)
        self.store = store
        self.thresholds = thresholds or MatchThresholds()

    async def resolve(self, entry: BibliographicEntry) -> ResolutionResult:
        expected = RecordMetadata(
            title=entry.title,
            authors=entry.authors,
            year=entry.year,
            doi=entry.doi,
            arxiv_id=entry.arxiv_id,
        )
        attempts: list[ResolutionAttempt] = []
        evidence: list[ResolutionEvidence] = []

        alias_hit = self._alias_lookup(entry)
        if alias_hit is not None:
            summary, alias_key = alias_hit
            attempts.append(
                ResolutionAttempt(
                    method=ResolutionMethod.ALIAS_CACHE,
                    status=ResolutionStatus.RESOLVED,
                    canonical_id=summary.id,
                )
            )
            return ResolutionResult(
                status=ResolutionStatus.RESOLVED,
                confidence=1.0,
                method=ResolutionMethod.ALIAS_CACHE,
                provider=summary.provider,
                canonical_id=summary.id,
                aliases=[alias_key, summary.id],
                summary=summary,
                expected=expected,
                actual=record_from_summary(summary),
                attempts=attempts,
            )

        evidence.extend(await self._doi_stage(entry, attempts))
        if not any(ev.accepted for ev in evidence):
            evidence.extend(await self._arxiv_stage(entry, attempts))
        if not any(ev.accepted for ev in evidence):
            evidence.extend(await self._title_stage(entry, attempts))

        return self._decide(entry, expected, attempts, evidence)

    def _alias_lookup(
        self, entry: BibliographicEntry
    ) -> tuple[PaperSummary, str] | None:
        if self.store is None:
            return None
        for key in alias_keys_for_entry(entry):
            canonical = self.store.get_canonical_id(key)
            if canonical is None:
                continue
            summary = self.store.get_paper_summary(canonical)
            if summary is not None:
                return summary, key
        return None

    async def _doi_stage(
        self, entry: BibliographicEntry, attempts: list[ResolutionAttempt]
    ) -> list[ResolutionEvidence]:
        evidence: list[ResolutionEvidence] = []
        if not entry.doi:
            return evidence
        normalized = normalize_doi(entry.doi)
        if not is_valid_doi(normalized):
            attempts.append(
                ResolutionAttempt(
                    method=ResolutionMethod.DOI_LOOKUP,
                    status=ResolutionStatus.REJECTED,
                    reject_reason=RejectReason.UNVERIFIABLE_IDENTIFIER,
                )
            )
            return evidence
        for provider in self.providers:
            try:
                candidate = await provider.lookup_doi(normalized)
            except Exception as exc:
                attempts.append(
                    ResolutionAttempt(
                        method=ResolutionMethod.DOI_LOOKUP,
                        provider=provider.name,
                        status=ResolutionStatus.REJECTED,
                        reject_reason=RejectReason.PROVIDER_UNAVAILABLE,
                        error=str(exc),
                    )
                )
                continue
            if candidate is None:
                attempts.append(
                    ResolutionAttempt(
                        method=ResolutionMethod.DOI_LOOKUP,
                        provider=provider.name,
                        status=ResolutionStatus.REJECTED,
                        reject_reason=RejectReason.NO_PROVIDER_MATCH,
                    )
                )
                continue
            evidence.append(
                self._verify(
                    entry,
                    candidate,
                    method=ResolutionMethod.DOI_LOOKUP,
                    provider_name=provider.name,
                )
            )
            if any(ev.accepted for ev in evidence):
                break
        return evidence

    async def _arxiv_stage(
        self, entry: BibliographicEntry, attempts: list[ResolutionAttempt]
    ) -> list[ResolutionEvidence]:
        evidence: list[ResolutionEvidence] = []
        if not entry.arxiv_id:
            return evidence
        normalized = normalize_arxiv(entry.arxiv_id)
        if not is_valid_arxiv(normalized):
            attempts.append(
                ResolutionAttempt(
                    method=ResolutionMethod.ARXIV_LOOKUP,
                    status=ResolutionStatus.REJECTED,
                    reject_reason=RejectReason.UNVERIFIABLE_IDENTIFIER,
                )
            )
            return evidence
        for provider in self.providers:
            try:
                candidate = await provider.lookup_arxiv(normalized)
            except Exception as exc:
                attempts.append(
                    ResolutionAttempt(
                        method=ResolutionMethod.ARXIV_LOOKUP,
                        provider=provider.name,
                        status=ResolutionStatus.REJECTED,
                        reject_reason=RejectReason.PROVIDER_UNAVAILABLE,
                        error=str(exc),
                    )
                )
                continue
            if candidate is None:
                attempts.append(
                    ResolutionAttempt(
                        method=ResolutionMethod.ARXIV_LOOKUP,
                        provider=provider.name,
                        status=ResolutionStatus.REJECTED,
                        reject_reason=RejectReason.NO_PROVIDER_MATCH,
                    )
                )
                continue
            evidence.append(
                self._verify(
                    entry,
                    candidate,
                    method=ResolutionMethod.ARXIV_LOOKUP,
                    provider_name=provider.name,
                )
            )
            if any(ev.accepted for ev in evidence):
                break
        return evidence

    async def _title_stage(
        self, entry: BibliographicEntry, attempts: list[ResolutionAttempt]
    ) -> list[ResolutionEvidence]:
        evidence: list[ResolutionEvidence] = []
        if not entry.title:
            return evidence
        for provider in self.providers:
            try:
                candidates = await provider.search_title(entry.title)
            except Exception as exc:
                attempts.append(
                    ResolutionAttempt(
                        method=ResolutionMethod.TITLE_SEARCH,
                        provider=provider.name,
                        status=ResolutionStatus.REJECTED,
                        reject_reason=RejectReason.PROVIDER_UNAVAILABLE,
                        error=str(exc),
                    )
                )
                continue
            attempts.append(
                ResolutionAttempt(
                    method=ResolutionMethod.TITLE_SEARCH,
                    provider=provider.name,
                    status=ResolutionStatus.REJECTED,
                    candidate_count=len(candidates),
                    reject_reason=RejectReason.NO_PROVIDER_MATCH
                    if not candidates
                    else None,
                )
            )
            stage = [
                self._verify(
                    entry,
                    candidate,
                    method=ResolutionMethod.TITLE_SEARCH,
                    provider_name=provider.name,
                )
                for candidate in candidates
            ]
            evidence.extend(stage)
            if any(ev.accepted for ev in stage):
                break
        return evidence

    def _identifier_match(
        self, entry: BibliographicEntry, candidate: PaperSummary
    ) -> bool:
        """True when the candidate carries the same exact DOI/arXiv identifier."""
        if entry.doi:
            candidate_doi = candidate.doi
            if candidate_doi and normalize_doi(candidate_doi) == normalize_doi(entry.doi):
                return True
        if entry.arxiv_id:
            normalized = normalize_arxiv(entry.arxiv_id)
            candidate_arxiv = candidate.arxiv_id
            if candidate_arxiv and normalize_arxiv(candidate_arxiv) == normalized:
                return True
            candidate_doi = candidate.doi
            if candidate_doi and normalize_doi(candidate_doi) == f"10.48550/arxiv.{normalized}":
                return True
        return False

    def _verify(
        self,
        entry: BibliographicEntry,
        candidate: PaperSummary,
        method: ResolutionMethod,
        provider_name: str,
    ) -> ResolutionEvidence:
        score = score_candidate(
            entry,
            candidate.title,
            candidate.authors,
            candidate.year,
            self.thresholds,
        )
        failures = hard_gate_failures(score, self.thresholds)
        if failures:
            return ResolutionEvidence(
                method=method,
                provider=provider_name,
                candidate=candidate,
                score=score,
                accepted=False,
                reject_reason=failures[0],
            )
        descriptive_checked = (
            score.title_checked or score.authors_checked or score.year_checked
        )
        identifier_match = method in (
            ResolutionMethod.DOI_LOOKUP,
            ResolutionMethod.ARXIV_LOOKUP,
        ) and self._identifier_match(entry, candidate)
        if not descriptive_checked and identifier_match:
            # Exact identifier lookup is sufficient evidence on its own,
            # even when the provider returns no descriptive metadata (STAB-4).
            score.confidence = 1.0
            return ResolutionEvidence(
                method=method,
                provider=provider_name,
                candidate=candidate,
                score=score,
                accepted=True,
            )
        if score.confidence < self.thresholds.min_confidence:
            return ResolutionEvidence(
                method=method,
                provider=provider_name,
                candidate=candidate,
                score=score,
                accepted=False,
                reject_reason=RejectReason.LOW_CONFIDENCE,
            )
        return ResolutionEvidence(
            method=method,
            provider=provider_name,
            candidate=candidate,
            score=score,
            accepted=True,
        )

    def _tie_break_key(self, evidence: ResolutionEvidence) -> tuple:
        provider_rank = {p.name: i for i, p in enumerate(self.providers)}
        candidate = evidence.candidate
        score = evidence.score
        identifier_match = (
            1
            if evidence.method
            in (ResolutionMethod.DOI_LOOKUP, ResolutionMethod.ARXIV_LOOKUP)
            else 0
        )
        return (
            -round(score.confidence, 6),
            -round(score.title_similarity, 6),
            -round(score.author_overlap, 6),
            -identifier_match,
            -metadata_completeness(candidate),
            provider_rank.get(evidence.provider or "", len(provider_rank)),
            normalize_id(candidate.provider, candidate.id),
        )

    def _decide(
        self,
        entry: BibliographicEntry,
        expected: RecordMetadata,
        attempts: list[ResolutionAttempt],
        evidence: list[ResolutionEvidence],
    ) -> ResolutionResult:
        accepted = [ev for ev in evidence if ev.accepted]
        if accepted:
            clusters: dict[str, list[ResolutionEvidence]] = {}
            for ev in accepted:
                clusters.setdefault(candidate_dedup_key(ev.candidate), []).append(ev)
            if len(clusters) > 1:
                best = min(accepted, key=self._tie_break_key)
                return ResolutionResult(
                    status=ResolutionStatus.AMBIGUOUS,
                    confidence=best.score.confidence,
                    method=best.method,
                    provider=best.provider,
                    expected=expected,
                    actual=record_from_summary(best.candidate),
                    reject_reason=RejectReason.MULTIPLE_CANDIDATES,
                    attempts=attempts,
                    evidence=evidence,
                )
            cluster = clusters[candidate_dedup_key(accepted[0].candidate)]
            best = min(cluster, key=self._tie_break_key)
            canonical = normalize_id(best.candidate.provider, best.candidate.id)
            if self.store is not None:
                self.store.cache_summary_for_nid(canonical, best.candidate)
            aliases = self._register_aliases(entry, canonical, cluster)
            return ResolutionResult(
                status=ResolutionStatus.RESOLVED,
                confidence=best.score.confidence,
                method=best.method,
                provider=best.provider,
                canonical_id=canonical,
                aliases=aliases,
                summary=best.candidate,
                expected=expected,
                actual=record_from_summary(best.candidate),
                attempts=attempts,
                evidence=evidence,
            )

        if evidence:
            best = min(evidence, key=self._tie_break_key)
            log.debug(
                "resolution_rejected",
                status="rejected",
                reason=best.reject_reason.value if best.reject_reason else None,
            )
            return ResolutionResult(
                status=ResolutionStatus.REJECTED,
                confidence=best.score.confidence,
                method=best.method,
                provider=best.provider,
                expected=expected,
                actual=record_from_summary(best.candidate),
                reject_reason=best.reject_reason,
                attempts=attempts,
                evidence=evidence,
            )

        reasons = {
            attempt.reject_reason
            for attempt in attempts
            if attempt.reject_reason is not None
        }
        reason = next(
            (r for r in _REJECT_PRIORITY if r in reasons), RejectReason.NO_PROVIDER_MATCH
        )
        return ResolutionResult(
            status=ResolutionStatus.REJECTED,
            confidence=0.0,
            expected=expected,
            reject_reason=reason,
            attempts=attempts,
        )

    def _register_aliases(
        self,
        entry: BibliographicEntry,
        canonical_id: str,
        cluster: list[ResolutionEvidence],
    ) -> list[str]:
        keys: set[str] = set(alias_keys_for_entry(entry))
        for ev in cluster:
            keys.update(alias_keys_for_summary(ev.candidate))
            keys.add(normalize_id(ev.candidate.provider, ev.candidate.id))
        keys.discard(canonical_id)
        sorted_keys = sorted(keys)
        if self.store is not None:
            for key in sorted_keys:
                self.store.add_alias(key, canonical_id)
            self.store.commit()
        return sorted_keys
