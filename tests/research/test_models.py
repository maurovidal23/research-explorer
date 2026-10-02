"""Domain model validation and canonical serialization."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_explorer.research.models import (
    AgentNotebook,
    BudgetState,
    Claim,
    ClaimStatus,
    EvidenceRef,
    FinalAnswer,
    OpenQuestion,
    ResearchObjective,
    canonical_json,
    state_hash,
)


def _objective(**overrides: object) -> ResearchObjective:
    values: dict[str, object] = {
        "run_id": "run1",
        "seed_paper_id": "openalex:W1",
        "question": "How do extracellular fields originate?",
    }
    values.update(overrides)
    return ResearchObjective(**values)  # type: ignore[arg-type]


def test_question_is_required_and_non_blank() -> None:
    with pytest.raises(ValidationError):
        _objective(question="   ")


def test_claim_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        Claim(id="c1", text="x", confidence=1.5)


def test_budget_remaining_and_exhaustion() -> None:
    budget = BudgetState(max_fetches=2, fetches_used=2)
    assert budget.fetches_remaining == 0
    assert budget.exhausted


def test_canonical_json_and_hash_are_stable() -> None:
    state = {
        "b": [1, 2],
        "a": {"y": 1, "x": 2},
    }
    assert canonical_json(state) == '{"a":{"x":2,"y":1},"b":[1,2]}'
    assert state_hash(state) == state_hash(dict(reversed(list(state.items()))))


def test_notebook_revision_is_monotonic() -> None:
    notebook = AgentNotebook(agent_id="a")
    notebook.bump()
    notebook.bump()
    assert notebook.revision == 2


def test_final_answer_render_excludes_nothing_but_labels() -> None:
    answer = FinalAnswer(
        question="Q?",
        supported_conclusions=["S [W1]"],
        plausible_interpretations=["P [W2]"],
        unknowns=["U"],
        citations=["W1", "W2"],
    )
    rendered = answer.render_markdown()
    assert "Supported conclusions" in rendered
    assert "Plausible interpretations" in rendered
    assert "## Citations" in rendered


def test_open_question_priority_bounds() -> None:
    with pytest.raises(ValidationError):
        OpenQuestion(id="q1", text="x", priority=2.0)


def test_evidence_ref_is_a_pointer_only() -> None:
    ref = EvidenceRef(paper_id="openalex:W1", locator="p.3", content_hash="abc")
    assert ref.paper_id == "openalex:W1"
    assert set(ref.model_dump()) == {
        "paper_id",
        "locator",
        "content_hash",
        "acquisition_event",
        "setting",
    }


def test_supported_claim_evidence_predicate() -> None:
    claim = Claim(
        id="c1",
        text="x",
        status=ClaimStatus.SUPPORTED,
        supporting=[EvidenceRef(paper_id="openalex:W1")],
    )
    assert claim.has_evidence
