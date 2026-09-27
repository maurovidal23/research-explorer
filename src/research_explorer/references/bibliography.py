"""Bibliography hashing, segmentation, and identifier recovery.

Deterministic helpers shared by the mapper and the graph builder. No network,
no LLM.
"""

from __future__ import annotations

import hashlib
import re

_DOI_IN_TEXT = re.compile(r"10\.\d{4,9}/[^\s,;)\]}<>\"']+", re.IGNORECASE)
_ARXIV_IN_TEXT = re.compile(
    r"(?:arxiv[:/\s]*|abs/)"
    r"(\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z-]+)?/\d{7})(?:v\d+)?",
    re.IGNORECASE,
)
_PMID_IN_TEXT = re.compile(r"(?:pmid[:\s]*)(\d{1,9})", re.IGNORECASE)
_BRACKET_SPLIT = re.compile(r"\n\s*\[\d+\]\s*")
_DOT_SPLIT = re.compile(r"\n\s*\d+\.\s+")
_REF_HEADING = re.compile(r"\n[ \t]*References[ \t]*\n", re.IGNORECASE)
_WS = re.compile(r"\s+")

MAX_RAW_TEXT_CHARS = 2000


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def source_content_hash(raw_entries: list[str]) -> str:
    """Stable hash of the acquired bibliography (defines a mapping revision)."""
    payload = "\n---\n".join(raw_entries)
    return sha256_text(payload)


def raw_entry_hash(raw_text: str) -> str:
    return sha256_text(_WS.sub(" ", raw_text).strip())


def provisional_reference_id(source_hash: str, entry_hash: str) -> str:
    """Synthetic, provider-non-authoritative id for an unresolved reference."""
    return f"bib:{source_hash[:16]}:{entry_hash[:16]}"


def trim_raw_entry(raw_text: str, limit: int = MAX_RAW_TEXT_CHARS) -> str:
    """Collapse whitespace and length-bound a raw entry for storage/display."""
    collapsed = _WS.sub(" ", raw_text).strip()
    return collapsed[:limit]


def extract_doi_candidates(raw_text: str) -> list[str]:
    return [m.group(0).rstrip(".,;") for m in _DOI_IN_TEXT.finditer(raw_text)]


def extract_arxiv_candidates(raw_text: str) -> list[str]:
    return [m.group(1) for m in _ARXIV_IN_TEXT.finditer(raw_text)]


def extract_pmid_candidates(raw_text: str) -> list[str]:
    return [m.group(1) for m in _PMID_IN_TEXT.finditer(raw_text)]


def segment_bibliography(text: str, max_entries: int = 0) -> list[str]:
    """Bound a separate bibliography-segmentation stage for unstructured text.

    Used only when a provider cannot supply entry boundaries. Never send a whole
    document to the mapping call to rediscover boundaries (FRG-1).
    """
    section = _find_reference_section(text)
    if section is None:
        return []
    entries = [e.strip() for e in _BRACKET_SPLIT.split(section) if len(e.strip()) > 20]
    if len(entries) <= 1:
        entries = [e.strip() for e in _DOT_SPLIT.split(section) if len(e.strip()) > 20]
    if len(entries) <= 1:
        entries = [
            _WS.sub(" ", p).strip()
            for p in section.split("\n\n")
            if len(p.strip()) > 20
        ]
    cleaned = [e for e in entries if e]
    return cleaned[:max_entries] if max_entries > 0 else cleaned


def _find_reference_section(text: str) -> str | None:
    matches = list(_REF_HEADING.finditer(text))
    if not matches:
        return None
    return text[matches[-1].end():]
