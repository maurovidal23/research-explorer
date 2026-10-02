"""Optional research scope resolution (SURV-1)."""

from __future__ import annotations

from research_explorer.graph.models import Paper
from research_explorer.memory.scope import derive_scope, resolve_scope


def _seed() -> Paper:
    return Paper(
        id="1234.5678",
        arxiv_id="1234.5678",
        title="A study of citation graph exploration",
        abstract="We study heuristic exploration of scientific citation graphs. " * 20,
        provider="arxiv",
        year=2024,
    )


def test_user_supplied_scope_is_preserved() -> None:
    scope, origin = resolve_scope("  transformer interpretability  ", _seed())
    assert scope == "transformer interpretability"
    assert origin == "user"


def test_blank_scope_is_derived_and_bounded_deterministically() -> None:
    first = derive_scope(_seed())
    second = derive_scope(_seed())
    assert first == second
    assert first
    assert len(first.split()) <= 40
    assert "citation graph exploration" in first
    scope, origin = resolve_scope("   ", _seed())
    assert scope == first
    assert origin == "derived"


def test_scope_derivation_is_deterministic_across_calls() -> None:
    assert resolve_scope("", _seed()) == resolve_scope("", _seed())
