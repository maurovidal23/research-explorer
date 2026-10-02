"""Frozen, network-free end-to-end survivor benchmark (PRD §4, EXAM-8).

Proves that the expected survivor uplift and terminal score are reproduced from
stored artifacts, without any network or LLM access.
"""

from __future__ import annotations

import asyncio
import json

from research_explorer.examination.benchmark import (
    OUTCOME_BENCHMARKED,
    BenchmarkConfig,
    BenchmarkRunner,
)
from research_explorer.examination.clients import FakeAnswerClient
from research_explorer.examination.generator import FakeExaminer
from research_explorer.examination.models import SelectionWeights
from research_explorer.examination.report import (
    benchmark_report_markdown,
    benchmark_result_json,
    private_key_artifact,
    public_exam_artifact,
)
from research_explorer.examination.synthetic import build_synthetic_corpus


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        selection_count=6,
        holdout_count=4,
        partition_seed=13,
        weights=SelectionWeights(),
        min_coverage=0.5,
        model_ids={
            "examiner_model": "fake-examiner-v1",
            "answer_model": "fake-answer-v1",
        },
        prompt_versions={"examiner": "v1", "answer": "v1"},
    )


def _run() -> tuple[BenchmarkRunner, object]:
    pack, candidates, acquired = build_synthetic_corpus()
    runner = BenchmarkRunner(pack, FakeExaminer(), FakeAnswerClient(), _config())
    result = asyncio.run(runner.run(candidates, acquired))
    return runner, result


def test_frozen_network_free_end_to_end_benchmark(tmp_path) -> None:
    runner, result = _run()

    assert result.outcome == OUTCOME_BENCHMARKED
    assert result.survivor_id
    assert runner.bank is not None
    assert runner.answer_key is not None

    # Deterministic bank with exactly one rejected ambiguous item.
    assert runner.bank.rejected_count >= 1
    assert runner.bank.accepted_count == 10

    # Disjoint partitions.
    assert not (set(runner.bank.selection_ids) & set(runner.bank.holdout_ids))
    assert len(runner.bank.selection_ids) == 6
    assert len(runner.bank.holdout_ids) == 4

    # Uplift and terminal score reproduced from stored artifacts.
    assert result.survivor_accuracy is not None
    assert result.naive_accuracy is not None
    assert result.uplift == round(result.survivor_accuracy - result.naive_accuracy, 6)
    assert result.uplift > 0

    assert result.selection is not None
    sel = result.selection
    expected_terminal = round(
        sel.weights.selection * (sel.selection_accuracy or 0.0)
        + sel.weights.process * sel.process_score
        + sel.weights.grounding * sel.grounding_score,
        6,
    )
    assert abs(sel.terminal_score - expected_terminal) < 1e-9

    # Persist public/private artifacts and read them back.
    public_path = tmp_path / "exam_public.json"
    key_path = tmp_path / "exam_key.private.json"
    result_path = tmp_path / "result.json"
    report_path = tmp_path / "report.md"
    public_path.write_text(public_exam_artifact(runner.bank), encoding="utf-8")
    key_path.write_text(private_key_artifact(runner.answer_key), encoding="utf-8")
    result_path.write_text(benchmark_result_json(result), encoding="utf-8")
    report_path.write_text(
        benchmark_report_markdown(result, runner.pack, bank=runner.bank), encoding="utf-8"
    )

    public = json.loads(public_path.read_text(encoding="utf-8"))
    stored = json.loads(result_path.read_text(encoding="utf-8"))
    assert stored["outcome"] == OUTCOME_BENCHMARKED
    assert stored["uplift"] == result.uplift
    assert stored["survivor_id"] == result.survivor_id
    assert not (set(public["selection_ids"]) & set(public["holdout_ids"]))
    assert "correct_option_id" not in public_path.read_text(encoding="utf-8")
    assert "correct_option_id" in key_path.read_text(encoding="utf-8")
    assert "Descriptive uplift" in report_path.read_text(encoding="utf-8")


def test_benchmark_is_deterministic() -> None:
    _runner_a, result_a = _run()
    _runner_b, result_b = _run()
    assert result_a.survivor_id == result_b.survivor_id
    assert result_a.uplift == result_b.uplift
    assert result_a.selection is not None and result_b.selection is not None
    assert result_a.selection.terminal_score == result_b.selection.terminal_score
