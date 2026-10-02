"""Narrative synthesis derived from structured research memory (SURV-4).

The synthesis is a read-only *view*. Regenerating it never mutates dossiers,
claims, evidence, or examination state. Length is bounded by a configurable
word budget with a safe default appropriate for a detailed research briefing.
"""

from __future__ import annotations

from research_explorer.memory.models import ClaimStatus, ResearchMemory

DEFAULT_SYNTHESIS_WORDS = 2000


def _words(text: str) -> int:
    return len(text.split())


def _truncate(text: str, remaining: int) -> tuple[str, int]:
    if remaining <= 0:
        return "", 0
    tokens = text.split()
    if len(tokens) <= remaining:
        return text, remaining - len(tokens)
    return " ".join(tokens[:remaining]), 0


def synthesize_memory(
    memory: ResearchMemory,
    max_words: int = DEFAULT_SYNTHESIS_WORDS,
) -> str:
    """Render a readable long-form synthesis that distinguishes evidence classes."""
    if max_words <= 0:
        max_words = DEFAULT_SYNTHESIS_WORDS
    remaining = max_words
    lines: list[str] = []
    budget: list[int] = [remaining]

    def emit(text: str) -> None:
        chunk, budget[0] = _truncate(text, budget[0])
        if chunk:
            lines.append(chunk)

    scope_label = memory.scope.strip() or "the seed paper's research context"
    emit(
        f"# Research synthesis\n\nScope ({memory.scope_origin}): {scope_label}\n"
    )

    established = [
        c
        for c in sorted(memory.claims.values(), key=lambda c: c.id)
        if c.status is ClaimStatus.SUPPORTED
    ]
    provisional = [
        c
        for c in sorted(memory.claims.values(), key=lambda c: c.id)
        if c.status is ClaimStatus.PROPOSED and c.has_resolving_support
    ]
    disputes = [
        c
        for c in sorted(memory.claims.values(), key=lambda c: c.id)
        if c.status is ClaimStatus.DISPUTED
    ]

    def section(title: str, claims) -> None:
        if not claims:
            return
        emit(f"\n## {title}\n")
        for claim in claims:
            sources = sorted({ref.paper_id for ref in claim.supporting if ref.paper_id})
            suffix = f" [{', '.join(sources)}]" if sources else ""
            emit(f"- {claim.text}{suffix}\n")

    section("Established findings", established)
    section("Provisional interpretations", provisional)
    section("Disputes", disputes)

    limitations: list[str] = []
    for dossier in sorted(memory.dossiers.values(), key=lambda d: d.paper_id):
        limitations.extend(dossier.limitations)
    if limitations:
        emit("\n## Limitations and threats to validity\n")
        for item in sorted(set(limitations)):
            emit(f"- {item}\n")

    if memory.gaps:
        emit("\n## Unknowns and open gaps\n")
        for gap in sorted(memory.gaps, key=lambda g: g.id):
            emit(f"- {gap.text}\n")

    unknowns = [
        c
        for c in sorted(memory.claims.values(), key=lambda c: c.id)
        if c.status is ClaimStatus.UNKNOWN
    ]
    if unknowns:
        emit("\n## Open questions\n")
        for claim in unknowns:
            emit(f"- {claim.text}\n")

    text = "".join(lines).strip()
    if _words(text) > max_words:
        text = " ".join(text.split()[:max_words])
    return text + "\n"


def synthesis_word_count(text: str) -> int:
    return _words(text)
