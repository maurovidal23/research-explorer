"""Human-readable benchmark report and public/private artifact serialization."""

from __future__ import annotations

import json
from typing import Any

from research_explorer.examination.models import (
    AnswerKey,
    BenchmarkResult,
    EvidencePack,
    ExamBank,
)


def public_exam_artifact(bank: ExamBank) -> str:
    """Public exam artifact without keys, rationales, or evidence annotations."""
    return json.dumps(
        {
            "exam_version": bank.exam_version,
            "partition_seed": bank.partition_seed,
            "selection_ids": list(bank.selection_ids),
            "holdout_ids": list(bank.holdout_ids),
            "questions": bank.public_payload(),
        },
        indent=2,
        sort_keys=True,
    )


def private_key_artifact(key: AnswerKey) -> str:
    """Restricted answer key; never written to the ordinary event stream."""
    return key.model_dump_json(indent=2)


def _accuracy(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.4f}"


def _breakdown(label: str, score: Any) -> list[str]:
    lines: list[str] = []
    if score is None:
        return lines
    for name, bucket in sorted(getattr(score, label).items()):
        lines.append(f"- {name}: {bucket.correct}/{bucket.total}")
    return lines


def benchmark_report_markdown(
    result: BenchmarkResult,
    pack: EvidencePack,
    scope: str = "",
    scope_origin: str = "derived",
    evidence_bearing: int = 0,
    dossier_count: int = 0,
    bank: ExamBank | None = None,
) -> str:
    lines: list[str] = ["# Survivor benchmark report", ""]
    lines.append(f"**Outcome:** {result.outcome}")
    if result.reason_code:
        lines.append(f"**Reason:** {result.reason_code} — {result.reason}")
    lines.append(f"**Effective scope ({scope_origin}):** {scope or '(derived)'}")
    lines.append(f"**Survivor:** {result.survivor_id or 'none'}")
    if result.selection is not None:
        sel = result.selection
        lines.append(
            "**Terminal score:** "
            f"{sel.terminal_score:.4f} "
            f"(E_selection={_accuracy(sel.selection_accuracy)}, "
            f"Q_process={sel.process_score:.4f}, G={sel.grounding_score:.4f})"
        )
        lines.append(f"**Candidate ranking:** {', '.join(sel.ranking) or 'none'}")
        lines.append(f"**Process-peak agent:** {sel.process_peak_agent or 'none'}")
    lines.extend(
        [
            "",
            "## Evidence coverage",
            f"- Pack hash: {pack.pack_hash}",
            f"- Eligible sources: {len(pack.sources)}",
            f"- Excluded sources: {len(pack.excluded)}",
            f"- Dossiers: {dossier_count} (evidence-bearing: {evidence_bearing})",
        ]
    )
    if bank is not None:
        lines.extend(
            [
                "",
                "## Bank and partitions",
                f"- Exam version: {bank.exam_version}",
                f"- Accepted: {bank.accepted_count} (rejected: {bank.rejected_count})",
                f"- Rejection reasons: {', '.join(bank.rejection_reasons) or 'none'}",
                f"- Partition seed: {bank.partition_seed}",
                f"- Selection items: {len(bank.selection_ids)}",
                f"- Holdout items: {len(bank.holdout_ids)}",
                f"- Partition frozen: {bank.partition_frozen}",
            ]
        )
    lines.extend(
        [
            "",
            "## Holdout benchmark",
            f"- Survivor accuracy: {_accuracy(result.survivor_accuracy)}",
            f"- Naive accuracy: {_accuracy(result.naive_accuracy)}",
            (
                f"- Descriptive uplift (percentage points): "
                f"{'unavailable' if result.uplift is None else f'{result.uplift * 100:.2f}'}"
            ),
            f"- Survivor arm unavailable: {result.survivor_unavailable or 'no'}",
            f"- Naive arm unavailable: {result.naive_unavailable or 'no'}",
            "",
            "## Survivor breakdown",
            *(_breakdown("by_category", result.survivor_score) or ["- unavailable"]),
            *(_breakdown("by_difficulty", result.survivor_score) or []),
            *(_breakdown("by_source_distance", result.survivor_score) or []),
            "",
            "## Configuration",
            f"- Fingerprint: {result.config_fingerprint or 'unset'}",
            f"- Models: {json.dumps(result.model_ids, sort_keys=True)}",
        ]
    )
    if result.survivor_score is not None:
        lines.append(
            f"- Selection bank: {len(result.survivor_score.item_outcomes)} holdout items"
        )
    return "\n".join(lines).strip() + "\n"


def benchmark_result_json(result: BenchmarkResult) -> str:
    return result.model_dump_json(indent=2)
