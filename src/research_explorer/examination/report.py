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
from research_explorer.examination.scoring import across_run_confidence_interval


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


def _uplift_interval(result: BenchmarkResult) -> str:
    if result.uplift_ci_low is None or result.uplift_ci_high is None:
        return "unavailable"
    return (
        f"[{result.uplift_ci_low * 100:.2f}, {result.uplift_ci_high * 100:.2f}] pp "
        "(descriptive, single run)"
    )


def _humanize(name: str) -> str:
    return name.replace("_", " ") or "unlabeled"


def _cost_text(result: BenchmarkResult) -> str:
    return "unavailable (no pricing configured)" if result.cost is None else f"{result.cost:.6f}"


def _fallback_text(result: BenchmarkResult) -> str:
    if result.examiner_fallback_used:
        return f"used ({result.examiner_fallback_model or 'unknown'})"
    if result.examiner_fallback_model:
        return f"configured, not used ({result.examiner_fallback_model})"
    return "not configured"


def _across_run_interval(result: BenchmarkResult) -> list[str]:
    interval = across_run_confidence_interval(result.replicate_uplifts)
    if interval is None:
        return []
    low, high = interval
    lines = [
        f"- Across-run paired 95% interval: [{low * 100:.2f}, {high * 100:.2f}] pp "
        f"({len(result.replicate_uplifts)} runs)"
    ]
    if result.replicate_fingerprints:
        lines.append(
            f"- Run fingerprints: {', '.join(result.replicate_fingerprints)}"
        )
    return lines


def _bucket_line(name: str, bucket: Any) -> str:
    accuracy = bucket.accuracy
    share = "" if accuracy is None else f" ({accuracy * 100:.1f}%)"
    return f"- {_humanize(name)}: {bucket.correct}/{bucket.total}{share}"


def _breakdown(label: str, score: Any) -> list[str]:
    if score is None:
        return []
    buckets = sorted(getattr(score, label).items())
    return [_bucket_line(name, bucket) for name, bucket in buckets]


def _arm_breakdown(title: str, score: Any) -> list[str]:
    if score is None:
        return [f"### {title}", "- unavailable"]
    sections: list[str] = [f"### {title}"]
    groups = (
        ("By category", "by_category"),
        ("By difficulty", "by_difficulty"),
        ("By content kind", "by_content_kind"),
        ("By source distance", "by_source_distance"),
    )
    wrote_any = False
    for heading, label in groups:
        rows = _breakdown(label, score)
        if not rows:
            continue
        wrote_any = True
        sections.append(f"#### {heading}")
        sections.extend(rows)
    if not wrote_any:
        sections.append("- unavailable")
    return sections


def _paired_summary(result: BenchmarkResult) -> list[str]:
    if not result.paired_outcomes:
        return []
    both = survivor_only = naive_only = neither = 0
    for outcome in result.paired_outcomes.values():
        survivor = bool(outcome.get("survivor"))
        naive = bool(outcome.get("naive"))
        if survivor and naive:
            both += 1
        elif survivor:
            survivor_only += 1
        elif naive:
            naive_only += 1
        else:
            neither += 1
    return [
        "#### Paired item outcomes",
        f"- both correct: {both}",
        f"- survivor only: {survivor_only}",
        f"- naive only: {naive_only}",
        f"- neither correct: {neither}",
    ]


def benchmark_report_markdown(
    result: BenchmarkResult,
    pack: EvidencePack,
    scope: str = "",
    scope_origin: str = "derived",
    evidence_bearing: int = 0,
    dossier_count: int = 0,
    bank: ExamBank | None = None,
) -> str:
    lines: list[str] = ["# Research Explorer — Survivor Benchmark Report", ""]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Outcome: {result.outcome}")
    if result.reason_code:
        lines.append(f"- Reason: {result.reason_code} — {result.reason}")
    lines.append(f"- Effective scope ({scope_origin}): {scope or '(derived)'}")
    lines.append(f"- Survivor: {result.survivor_id or 'none'}")
    if result.selection is not None:
        sel = result.selection
        lines.append(
            "- Terminal score: "
            f"{sel.terminal_score:.4f} "
            f"(E_selection={_accuracy(sel.selection_accuracy)}, "
            f"Q_process={sel.process_score:.4f}, G={sel.grounding_score:.4f})"
        )
        lines.append(f"- Candidate ranking: {', '.join(sel.ranking) or 'none'}")
        lines.append(f"- Process-peak agent: {sel.process_peak_agent or 'none'}")
    lines.extend(
        [
            "",
            "## Evidence coverage",
            "",
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
                "",
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
            "",
            f"- Survivor accuracy: {_accuracy(result.survivor_accuracy)}",
            f"- Naive accuracy: {_accuracy(result.naive_accuracy)}",
            (
                f"- Descriptive uplift (percentage points): "
                f"{'unavailable' if result.uplift is None else f'{result.uplift * 100:.2f}'}"
            ),
            f"- Paired 95% interval (item-level): {_uplift_interval(result)}",
            *_across_run_interval(result),
            f"- Survivor arm unavailable: {result.survivor_unavailable or 'no'}",
            f"- Naive arm unavailable: {result.naive_unavailable or 'no'}",
            "",
            "## Breakdowns",
            "",
            *_arm_breakdown("Survivor arm", result.survivor_score),
        ]
    )
    if result.naive_score is not None:
        lines.extend(["", *_arm_breakdown("Naive arm", result.naive_score)])
    paired = _paired_summary(result)
    if paired:
        lines.extend(["", *paired])
    lines.extend(
        [
            "",
            "## Configuration",
            "",
            f"- Fingerprint: {result.config_fingerprint or 'unset'}",
            f"- Models: {json.dumps(result.model_ids, sort_keys=True)}",
            f"- Token usage: {json.dumps(result.token_usage, sort_keys=True) or '{}'}",
            f"- Answer latency (s): {result.latency_seconds:.4f}",
            f"- Cost (USD): {_cost_text(result)}",
            f"- Examiner fallback: {_fallback_text(result)}",
        ]
    )
    if result.survivor_score is not None:
        lines.append(
            f"- Holdout bank: {len(result.survivor_score.item_outcomes)} items scored"
        )
    return "\n".join(lines).strip() + "\n"


def benchmark_result_json(result: BenchmarkResult) -> str:
    return result.model_dump_json(indent=2)
