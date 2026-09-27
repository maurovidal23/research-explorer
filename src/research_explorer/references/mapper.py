"""Dedicated LLM mapper for bibliography entries (FRG-2).

Batches are deterministic and ordinal-based; reconciliation guarantees exactly
one result per input entry regardless of missing, duplicated, or reordered LLM
output. Claimed identifiers are verified against the raw entry before they can
reach the graph.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from research_explorer.agents.llm_client import LLMClient
from research_explorer.agents.prompts import MAP_BIBLIOGRAPHY_SYSTEM, map_bibliography_batch
from research_explorer.config import ReferenceMappingConfig
from research_explorer.logging_setup import get_logger
from research_explorer.references.bibliography import (
    extract_arxiv_candidates,
    extract_doi_candidates,
    extract_pmid_candidates,
    sha256_text,
)
from research_explorer.references.models import (
    EntryType,
    MappedEntry,
    MappingStatus,
    RawBibliographyEntry,
)
from research_explorer.resolution.resolver import (
    is_valid_arxiv,
    is_valid_doi,
    is_valid_pmid,
    normalize_arxiv,
    normalize_doi,
    normalize_pmid,
)

log = get_logger("references.mapper")

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)

_TOKENS_PER_ENTRY = 1024

MAPPING_BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entry_id": {"type": "string"},
                    "ordinal": {"type": "integer"},
                    "title": {"type": "string"},
                    "authors": {"type": "array", "items": {"type": "string"}},
                    "year": {"type": ["integer", "null"]},
                    "venue": {"type": ["string", "null"]},
                    "volume": {"type": ["string", "null"]},
                    "issue": {"type": ["string", "null"]},
                    "pages": {"type": ["string", "null"]},
                    "doi": {"type": ["string", "null"]},
                    "arxiv_id": {"type": ["string", "null"]},
                    "pmid": {"type": ["string", "null"]},
                    "entry_type": {"type": "string"},
                    "parse_confidence": {"type": "number"},
                    "parse_notes": {"type": "string"},
                    "mapping_status": {"type": "string"},
                },
                "required": [
                    "entry_id",
                    "ordinal",
                    "title",
                    "authors",
                    "year",
                    "venue",
                    "volume",
                    "issue",
                    "pages",
                    "doi",
                    "arxiv_id",
                    "pmid",
                    "entry_type",
                    "parse_confidence",
                    "parse_notes",
                    "mapping_status",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["entries"],
    "additionalProperties": False,
}


@dataclass
class BatchMapResult:
    """Outcome of mapping one batch of entries."""

    results: dict[str, MappedEntry] = field(default_factory=dict)
    failed_entry_ids: list[str] = field(default_factory=list)
    malformed: bool = False


def prompt_hash() -> str:
    return sha256_text(MAP_BIBLIOGRAPHY_SYSTEM)[:16]


class ReferenceMapper:
    """Map raw bibliography entries with a dedicated, schema-constrained call."""

    def __init__(self, llm: LLMClient, config: ReferenceMappingConfig, model: str) -> None:
        self.llm = llm
        self.cfg = config
        self.model = model
        self.prompt_hash = prompt_hash()

    @property
    def batch_size(self) -> int:
        return max(1, self.cfg.batch_size)

    def token_budget_for(self, batch_size: int) -> int:
        """Completion-token budget sized from the batch and the response schema.

        The configured value is a floor; a length-truncated response would
        otherwise look like a failed batch for every paper (FRG-2).
        """
        return max(self.cfg.max_completion_tokens_per_batch, batch_size * _TOKENS_PER_ENTRY)

    def build_batches(
        self, entries: Sequence[RawBibliographyEntry]
    ) -> list[list[RawBibliographyEntry]]:
        """Deterministic ordinal batches, independent of research relevance."""
        ordered = sorted(entries, key=lambda e: e.ordinal)
        size = self.batch_size
        return [ordered[i : i + size] for i in range(0, len(ordered), size)]

    async def map_batch(self, batch: Sequence[RawBibliographyEntry]) -> BatchMapResult:
        if not batch:
            return BatchMapResult()
        payload = [
            {"entry_id": e.entry_id, "ordinal": e.ordinal, "raw_text": e.raw_text}
            for e in batch
        ]
        messages = map_bibliography_batch(payload)
        max_tokens = self.token_budget_for(len(batch))
        chat_json = getattr(self.llm, "chat_json", None)
        if callable(chat_json):
            result = await chat_json(
                messages,
                model=self.model,
                schema=MAPPING_BATCH_SCHEMA,
                temperature=0.0,
                max_tokens=max_tokens,
                purpose="reference_mapping",
            )
            raw = json.dumps(result)
        else:
            raw = await self.llm.chat(
                messages,
                model=self.model,
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                purpose="reference_mapping",
            )
        return self.parse_batch_response(raw, batch)

    def parse_batch_response(
        self, raw: str, batch: Sequence[RawBibliographyEntry]
    ) -> BatchMapResult:
        """Parse and reconcile one LLM response against the input batch."""
        by_id = {e.entry_id: e for e in batch}
        obj = _parse_json_object(raw)
        if obj is None:
            return BatchMapResult(
                results={}, failed_entry_ids=[e.entry_id for e in batch], malformed=True
            )
        items = obj.get("entries")
        if not isinstance(items, list):
            return BatchMapResult(
                results={}, failed_entry_ids=[e.entry_id for e in batch], malformed=True
            )

        by_ordinal = {e.ordinal: e for e in batch}
        reconciled: dict[str, MappedEntry] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            entry_id = str(item.get("entry_id") or "")
            ordinal = item.get("ordinal")
            target = by_id.get(entry_id)
            if target is None and isinstance(ordinal, int):
                target = by_ordinal.get(ordinal)
            if target is None or target.entry_id in reconciled:
                continue
            mapped = self._coerce(item, target)
            verified = self._verify_identifiers(mapped, target.raw_text)
            if self._is_effectively_empty(verified):
                verified.mapping_status = MappingStatus.UNPARSED
            reconciled[target.entry_id] = verified

        missing = [e.entry_id for e in batch if e.entry_id not in reconciled]
        return BatchMapResult(results=reconciled, failed_entry_ids=missing)

    def _coerce(self, item: dict, target: RawBibliographyEntry) -> MappedEntry:
        status_raw = str(item.get("mapping_status") or "mapped").lower()
        status = (
            MappingStatus.UNPARSED
            if status_raw in ("unparsed", "failed", "mapping_failed")
            else MappingStatus.MAPPED
        )
        entry_type_raw = str(item.get("entry_type") or "other").lower()
        try:
            entry_type = EntryType(entry_type_raw)
        except ValueError:
            entry_type = EntryType.OTHER
        authors = item.get("authors") or []
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(";") if a.strip()]
        elif not isinstance(authors, list):
            authors = []
        try:
            confidence = float(item.get("parse_confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        year = item.get("year")
        if isinstance(year, str) and year.isdigit():
            year = int(year)
        if not isinstance(year, int):
            year = None
        return MappedEntry(
            entry_id=target.entry_id,
            ordinal=target.ordinal,
            title=str(item.get("title") or "").strip(),
            authors=[str(a) for a in authors],
            year=year,
            venue=_opt_str(item.get("venue")),
            volume=_opt_str(item.get("volume")),
            issue=_opt_str(item.get("issue")),
            pages=_opt_str(item.get("pages")),
            doi=_opt_str(item.get("doi")),
            arxiv_id=_opt_str(item.get("arxiv_id")),
            pmid=_opt_str(item.get("pmid")),
            entry_type=entry_type,
            parse_confidence=confidence,
            parse_notes=str(item.get("parse_notes") or "")[:500],
            mapping_status=status,
        )

    def _verify_identifiers(self, mapped: MappedEntry, raw_text: str) -> MappedEntry:
        """Reject claimed identifiers absent from the raw entry (FRG-2)."""
        notes: list[str] = [mapped.parse_notes] if mapped.parse_notes else []
        confidence = mapped.parse_confidence
        if mapped.doi:
            normalized = normalize_doi(mapped.doi)
            candidates = {normalize_doi(c) for c in extract_doi_candidates(raw_text)}
            if is_valid_doi(normalized) and normalized in candidates:
                mapped.doi = normalized
            else:
                mapped.doi = None
                confidence *= 0.5
                notes.append("rejected unverified doi")
        if mapped.arxiv_id:
            normalized = normalize_arxiv(mapped.arxiv_id)
            candidates = {normalize_arxiv(c) for c in extract_arxiv_candidates(raw_text)}
            if is_valid_arxiv(normalized) and normalized in candidates:
                mapped.arxiv_id = normalized
            else:
                mapped.arxiv_id = None
                confidence *= 0.5
                notes.append("rejected unverified arxiv_id")
        if mapped.pmid:
            normalized = normalize_pmid(mapped.pmid)
            candidates = {normalize_pmid(c) for c in extract_pmid_candidates(raw_text)}
            if is_valid_pmid(normalized) and normalized in candidates:
                mapped.pmid = normalized
            else:
                mapped.pmid = None
                confidence *= 0.5
                notes.append("rejected unverified pmid")
        mapped.parse_confidence = max(0.0, min(1.0, confidence))
        mapped.parse_notes = "; ".join(n for n in notes if n)[:500]
        return mapped

    @staticmethod
    def _is_effectively_empty(mapped: MappedEntry) -> bool:
        return not mapped.title and not mapped.doi and not mapped.arxiv_id and not (
            mapped.pmid
        )


def _parse_json_object(raw: str) -> dict | None:
    cleaned = _JSON_FENCE.sub("", (raw or "").strip())
    with contextlib.suppress(json.JSONDecodeError):
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
    return None


def _opt_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
