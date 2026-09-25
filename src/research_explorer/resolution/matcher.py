"""Deterministic title/author/year matching for identity resolution.

No LLM, no network: pure normalization and scoring with explicit thresholds.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any

from research_explorer.resolution.models import (
    BibliographicEntry,
    CandidateScore,
    RejectReason,
)

_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)
_WS = re.compile(r"\s+", re.UNICODE)

STOP_WORDS = frozenset(
    {"a", "an", "the", "of", "for", "and", "in", "on", "to", "with", "by", "from"}
)


def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    t = _PUNCT.sub(" ", title.casefold())
    return _WS.sub(" ", t).strip()


def title_tokens(title: str) -> set[str]:
    return {
        t for t in normalize_title(title).split() if t and t not in STOP_WORDS
    }


def title_similarity(a: str, b: str) -> float:
    """Blend of SequenceMatcher ratio and token Jaccard over normalized titles."""
    if not a or not b:
        return 0.0
    na, nb = normalize_title(a), normalize_title(b)
    if na == nb:
        return 1.0
    seq = difflib.SequenceMatcher(None, na, nb).ratio()
    ta, tb = title_tokens(a), title_tokens(b)
    jac = len(ta & tb) / len(ta | tb) if ta and tb else 0.0
    return 0.5 * seq + 0.5 * jac


def surname(name: str) -> str:
    """Last token of a personal name; handles 'Vaswani, Ashish' and 'Ashish Vaswani'."""
    cleaned = name.strip()
    if "," in cleaned:
        cleaned = cleaned.split(",")[0]
    return _PUNCT.sub("", cleaned.split()[-1]).casefold() if cleaned.split() else ""


def author_overlap(expected: list[str], actual: list[str]) -> float:
    """Fraction of the expected author surnames found among the actual ones."""
    exp = {surname(a) for a in expected if a and surname(a)}
    act = {surname(a) for a in actual if a and surname(a)}
    if not exp or not act:
        return 0.0
    return len(exp & act) / len(exp)


def year_matches(expected: int | None, actual: int | None, tolerance: int) -> bool:
    if expected is None or actual is None:
        return True
    return abs(expected - actual) <= tolerance


@dataclass(frozen=True)
class MatchThresholds:
    min_title_similarity: float = 0.82
    min_author_overlap: float = 0.34
    min_confidence: float = 0.75
    year_tolerance: int = 2

    @classmethod
    def from_config(cls, cfg: Any) -> MatchThresholds:
        return cls(
            min_title_similarity=float(
                getattr(cfg, "min_title_similarity", cls.min_title_similarity)
            ),
            min_author_overlap=float(
                getattr(cfg, "min_author_overlap", cls.min_author_overlap)
            ),
            min_confidence=float(getattr(cfg, "min_confidence", cls.min_confidence)),
            year_tolerance=int(getattr(cfg, "year_tolerance", cls.year_tolerance)),
        )


W_TITLE = 0.6
W_AUTHOR = 0.25
W_YEAR = 0.15


def score_candidate(
    entry: BibliographicEntry,
    actual_title: str,
    actual_authors: list[str],
    actual_year: int | None,
    thresholds: MatchThresholds,
) -> CandidateScore:
    """Score one candidate against an entry, re-normalizing over available evidence."""
    score = CandidateScore()
    components: list[tuple[float, float]] = []

    if entry.title and actual_title:
        score.title_similarity = title_similarity(entry.title, actual_title)
        score.title_checked = True
        components.append((W_TITLE, score.title_similarity))

    if entry.authors and actual_authors:
        score.author_overlap = author_overlap(entry.authors, actual_authors)
        score.authors_checked = True
        components.append((W_AUTHOR, score.author_overlap))

    if entry.year is not None and actual_year is not None:
        score.year_checked = True
        score.year_ok = year_matches(entry.year, actual_year, thresholds.year_tolerance)
        components.append((W_YEAR, 1.0 if score.year_ok else 0.0))

    total_w = sum(w for w, _ in components)
    score.confidence = (
        sum(w * v for w, v in components) / total_w if total_w > 0 else 0.0
    )
    return score


def hard_gate_failures(
    score: CandidateScore, thresholds: MatchThresholds
) -> list[RejectReason]:
    """Explicit threshold failures that reject a candidate regardless of confidence."""
    failures: list[RejectReason] = []
    if score.title_checked and score.title_similarity < thresholds.min_title_similarity:
        failures.append(RejectReason.TITLE_MISMATCH)
    if score.authors_checked and score.author_overlap < thresholds.min_author_overlap:
        failures.append(RejectReason.AUTHOR_MISMATCH)
    if score.year_checked and not score.year_ok:
        failures.append(RejectReason.YEAR_MISMATCH)
    return failures
