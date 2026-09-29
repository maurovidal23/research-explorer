"""Evidence-pack eligibility and independent item validation (EXAM-1/3/4)."""

from __future__ import annotations

from research_explorer.examination.models import (
    CATEGORY_CONCEPTS,
    DIFFICULTY_EASY,
    ExamItem,
    ExamOption,
)
from research_explorer.examination.pack import (
    REASON_DUPLICATE as REASON_DUPLICATE_CONTENT,
)
from research_explorer.examination.pack import (
    REASON_METADATA_ONLY,
    REASON_NO_EVIDENCE,
    build_evidence_pack,
)
from research_explorer.examination.synthetic import build_synthetic_dossiers
from research_explorer.examination.validation import (
    REASON_ANSWER_LEAKAGE,
    REASON_INVALID_SCHEMA,
    REASON_KEY_NOT_IN_OPTIONS,
    REASON_MULTIPLE_DEFENSIBLE,
    REASON_UNRESOLVED_EVIDENCE,
    validate_bank,
    validate_item,
)
from research_explorer.examination.validation import (
    REASON_DUPLICATE as REASON_DUPLICATE_QUESTION,
)
from research_explorer.graph.models import Paper
from research_explorer.memory.extract import dossier_from_paper
from research_explorer.memory.models import ContentKind, PaperDossier


def _pack():
    dossiers = build_synthetic_dossiers(count=4)
    return build_evidence_pack(
        sorted(dossiers)[0], "scope", dossiers
    ).freeze(), dossiers


def _valid_item(source_id: str, question: str = "Which statement is entailed?") -> ExamItem:
    return ExamItem(
        question_id="q-ok",
        category=CATEGORY_CONCEPTS,
        difficulty=DIFFICULTY_EASY,
        question=question,
        options=[
            ExamOption(id="A", text=f"Entailed statement about {source_id}."),
            ExamOption(id="B", text=f"False statement about {source_id}."),
            ExamOption(id="C", text=f"Incomplete statement about {source_id}."),
            ExamOption(id="D", text=f"Unrelated statement about {source_id}."),
        ],
        correct_option_id="A",
        evidence_refs=[source_id],
    )


def test_pack_excludes_metadata_only_and_evidence_less_sources() -> None:
    dossiers = build_synthetic_dossiers(count=2)
    metadata = PaperDossier(paper_id="arxiv:meta", content_kind=ContentKind.METADATA)
    no_evidence = PaperDossier(
        paper_id="arxiv:empty", content_kind=ContentKind.ABSTRACT
    )
    dossiers[metadata.paper_id] = metadata
    dossiers[no_evidence.paper_id] = no_evidence
    pack = build_evidence_pack("arxiv:syn-00", "scope", dossiers).freeze()
    reasons = {exclusion.reason for exclusion in pack.excluded}
    assert REASON_METADATA_ONLY in reasons
    assert REASON_NO_EVIDENCE in reasons
    assert all(e.source_id not in {"arxiv:meta", "arxiv:empty"} for e in pack.sources)


def test_pack_deduplicates_identical_content_by_hash() -> None:
    body = "Identical acquired body. " * 60
    first = dossier_from_paper(
        Paper(id="dup-1", title="First", provider="arxiv", fulltext=body)
    )
    second = dossier_from_paper(
        Paper(id="dup-2", title="Second", provider="arxiv", fulltext=body)
    )
    pack = build_evidence_pack(
        first.paper_id, "scope", {first.paper_id: first, second.paper_id: second}
    ).freeze()
    assert len(pack.sources) == 1
    assert any(x.reason == REASON_DUPLICATE_CONTENT for x in pack.excluded)


def test_pack_hash_is_stable_and_public_annotations_are_stripped() -> None:
    dossiers = build_synthetic_dossiers(count=3)
    first = build_evidence_pack("arxiv:syn-00", "scope", dossiers).freeze()
    second = build_evidence_pack("arxiv:syn-00", "scope", dossiers).freeze()
    assert first.pack_hash == second.pack_hash
    public = first.without_annotations()
    assert all("excerpt" not in source for source in public["sources"])
    assert all("content_hash" not in source for source in public["sources"])


def test_valid_item_is_accepted_but_insufficient_evidence_option_is_allowed() -> None:
    pack, dossiers = _pack()
    source_id = sorted(dossiers)[0]
    item = _valid_item(source_id)
    assert validate_item(item, pack) == []
    item.has_insufficient_evidence_option = True
    assert validate_item(item, pack) == []
    accepted, rejected, _reasons = validate_bank([item], pack)
    assert len(accepted) == 1
    assert not rejected


def test_item_validation_rejects_schema_key_and_leakage_failures() -> None:
    pack, dossiers = _pack()
    source_id = sorted(dossiers)[0]

    assert REASON_KEY_NOT_IN_OPTIONS in validate_item(
        _valid_item(source_id).model_copy(update={"correct_option_id": "Z"}), pack
    )
    assert REASON_INVALID_SCHEMA in validate_item(
        _valid_item(source_id).model_copy(update={"options": [ExamOption(id="A", text="only")]}),
        pack,
    )
    assert REASON_ANSWER_LEAKAGE in validate_item(
        _valid_item(source_id).model_copy(
            update={
                "options": [
                    ExamOption(id="A", text="All of the above"),
                    ExamOption(id="B", text=f"False {source_id}"),
                    ExamOption(id="C", text=f"Incomplete {source_id}"),
                    ExamOption(id="D", text=f"Unrelated {source_id}"),
                ]
            }
        ),
        pack,
    )
    assert REASON_MULTIPLE_DEFENSIBLE in validate_item(
        _valid_item(source_id).model_copy(update={"defensible_option_ids": ["A", "B"]}), pack
    )


def test_item_validation_rejects_unresolved_and_duplicate_questions() -> None:
    pack, dossiers = _pack()
    source_id = sorted(dossiers)[0]
    assert REASON_UNRESOLVED_EVIDENCE in validate_item(
        _valid_item(source_id).model_copy(update={"evidence_refs": ["ghost"]}), pack
    )

    first = _valid_item(source_id, question="Which of these statements is entailed?")
    near_duplicate = _valid_item(
        source_id, question="Which of these statements is entailed"
    ).model_copy(update={"question_id": "q-dup"})
    assert REASON_DUPLICATE_QUESTION in validate_item(near_duplicate, pack, existing=[first])
    accepted, rejected, reasons = validate_bank([first, near_duplicate], pack)
    assert len(accepted) == 1
    assert len(rejected) == 1
    assert REASON_DUPLICATE_QUESTION in reasons
