"""Effective research scope resolution (SURV-1).

The positional CLI text is an optional research scope, not the question the
survivor is trained to answer. When blank, a deterministic bounded topic
profile is derived from the seed title, abstract, and identifiers.
"""

from __future__ import annotations

from research_explorer.graph.models import Paper

MAX_SCOPE_WORDS = 40


def derive_scope(paper: Paper) -> str:
    """Deterministic bounded topic profile from a seed paper."""
    parts: list[str] = [paper.title or ""]
    if paper.year:
        parts.append(str(paper.year))
    text = paper.tldr or paper.abstract or ""
    if text:
        parts.append(" ".join(text.split()[:MAX_SCOPE_WORDS]))
    identifiers = [paper.doi, paper.arxiv_id, paper.pmid]
    parts.extend(ident for ident in identifiers if ident)
    scope = " ".join(part for part in parts if part).strip()
    return " ".join(scope.split()[:MAX_SCOPE_WORDS])


def resolve_scope(seed_query: str, paper: Paper) -> tuple[str, str]:
    """Return ``(effective_scope, origin)``; origin is ``user`` or ``derived``."""
    cleaned = (seed_query or "").strip()
    if cleaned:
        return cleaned, "user"
    return derive_scope(paper), "derived"
