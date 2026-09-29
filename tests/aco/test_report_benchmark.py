"""The exploration report reconstructs durable benchmark results (OBS-1).

PRD §4 case 22: the final report reconstructs scores, partitions, models, and
reason codes from durable state instead of losing them to the benchmark artifact.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from research_explorer.config import Config
from research_explorer.examination.models import (
    BenchmarkResult,
    ExamScore,
    SurvivorSelection,
)
from research_explorer.orchestrator.report import build_report
from research_explorer.orchestrator.runner import Orchestrator


def _result() -> BenchmarkResult:
    return BenchmarkResult(
        survivor_id="a0",
        outcome="completed_benchmarked",
        survivor_accuracy=0.75,
        naive_accuracy=0.25,
        uplift=0.5,
        selection_count=6,
        holdout_count=4,
        state_hash="deadbeef",
        config_fingerprint="cfg-42",
        model_ids={"answer_model": "fake-answer-v1", "examiner_model": "fake-examiner-v1"},
        survivor_score=ExamScore(total=4, correct=3),
        naive_score=ExamScore(total=4, correct=1),
        selection=SurvivorSelection(
            survivor_id="a0",
            eligible=True,
            terminal_score=0.8,
            selection_accuracy=0.7,
            process_score=0.6,
            grounding_score=0.5,
            ranking=["a0", "a1"],
            process_peak_agent="a1",
        ),
    )


def _empty_report(**kwargs) -> str:
    colony = SimpleNamespace(
        best_narrative="",
        best_snapshot_agent="",
        best_quality=0.1,
        best_snapshot_oleada=0,
        agents=[],
        shared_frontier=SimpleNamespace(papers=[]),
        active_candidates=lambda: [],
    )
    return build_report(
        config=Config(),
        colony=colony,
        scheduler=SimpleNamespace(oleada_count=1, total_fetches=2, history=[]),
        convergence=SimpleNamespace(state=SimpleNamespace(quality_history=[])),
        graph=SimpleNamespace(get_all_paper_summaries=lambda: []),
        seed_paper_id="arxiv:seed",
        seed_query="scope",
        elapsed=12.0,
        **kwargs,
    )


def test_report_includes_benchmark_when_present() -> None:
    report = _empty_report(
        benchmark_result=_result(),
        effective_scope="citation graph",
        scope_origin="user",
    )
    assert "## Survivor Benchmark" in report
    assert "**Survivor:** a0" in report
    assert "0.7000" in report and "0.6000" in report
    assert "**Selection items:** 6" in report
    assert "**Holdout items:** 4" in report
    assert "**Survivor accuracy:** 0.7500" in report
    assert "**Naive accuracy:** 0.2500" in report
    assert "50.00 pp" in report
    assert "fake-answer-v1" in report
    assert "deadbeef" in report
    assert "**Effective scope (user):** citation graph" in report


def test_report_omits_benchmark_section_without_a_result() -> None:
    report = _empty_report()
    assert "## Survivor Benchmark" not in report
    assert "## Winning Narrative" in report


def test_report_renders_frozen_survivor_synthesis() -> None:
    report = _empty_report(
        benchmark_result=_result(),
        survivor_synthesis="Established findings\n- synthesized view",
    )
    assert "## Frozen Survivor Synthesis" in report
    assert "synthesized view" in report


def test_live_answer_model_fails_closed_when_unset() -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    with pytest.raises(RuntimeError):
        orchestrator._require_answer_model(SimpleNamespace(answer_model=""))
    assert (
        orchestrator._require_answer_model(SimpleNamespace(answer_model="answer-x"))
        == "answer-x"
    )
