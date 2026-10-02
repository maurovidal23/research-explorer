import asyncio
from pathlib import Path

from research_explorer.fixed_benchmark.batch import (
    FixedResearchBatchRunner,
    PaperRunResult,
    utc_now,
)
from research_explorer.fixed_benchmark.models import BenchmarkManifest, BenchmarkPaper


def _manifest() -> BenchmarkManifest:
    papers = [
        BenchmarkPaper(
            paper_id=f"paper-{index}",
            arxiv_id=f"1234.0000{index}",
            version=1,
            title=f"Paper {index}",
            cluster="test",
            abstract_url=f"https://arxiv.org/abs/1234.0000{index}v1",
            pdf_url=f"https://arxiv.org/pdf/1234.0000{index}v1",
            source_sha256=str(index) * 64,
        )
        for index in range(3)
    ]
    return BenchmarkManifest(
        benchmark_id="fixture",
        version="1.0.0",
        description="fixture",
        question_count_per_paper=1,
        options_per_question=3,
        papers=papers,
        categories={"method": 1},
        difficulties={"hard": 1},
        correct_option_counts={"1": 1},
        generator_model="generator",
        critic_model="critic",
        prompt_version="v1",
        created_at="2026-10-01T00:00:00Z",
    )


def test_batch_is_bounded_and_resumes_completed_papers(tmp_path: Path) -> None:
    calls: list[str] = []
    active = 0
    peak = 0

    async def execute(paper: BenchmarkPaper, paper_dir: Path) -> PaperRunResult:
        nonlocal active, peak
        calls.append(paper.paper_id)
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        bundle = paper_dir / "survivor_bundle.private.json"
        bundle.parent.mkdir(parents=True, exist_ok=True)
        bundle.write_text("{}", encoding="utf-8")
        return PaperRunResult(
            paper_id=paper.paper_id,
            benchmark_version="1.0.0",
            source_sha256=paper.source_sha256,
            config_fingerprint="cfg",
            status="completed",
            started_at=utc_now(),
            finished_at=utc_now(),
            survivor_bundle_path=str(bundle),
        )

    runner = FixedResearchBatchRunner(
        _manifest(), tmp_path, "cfg", execute, max_concurrent=2
    )
    first = asyncio.run(runner.run())
    second = asyncio.run(runner.run())

    assert first.completed == 3
    assert second.completed == 3
    assert peak == 2
    assert calls == ["paper-0", "paper-1", "paper-2"]


def test_batch_records_failures_without_cancelling_other_papers(tmp_path: Path) -> None:
    async def execute(paper: BenchmarkPaper, paper_dir: Path) -> PaperRunResult:
        if paper.paper_id == "paper-1":
            raise RuntimeError("provider unavailable")
        bundle = paper_dir / "bundle.json"
        bundle.parent.mkdir(parents=True, exist_ok=True)
        bundle.write_text("{}", encoding="utf-8")
        return PaperRunResult(
            paper_id=paper.paper_id,
            benchmark_version="1.0.0",
            source_sha256=paper.source_sha256,
            config_fingerprint="cfg",
            status="completed",
            started_at=utc_now(),
            finished_at=utc_now(),
            survivor_bundle_path=str(bundle),
        )

    summary = asyncio.run(
        FixedResearchBatchRunner(_manifest(), tmp_path, "cfg", execute).run()
    )

    assert summary.completed == 2
    assert summary.failed == 1
    assert "provider unavailable" in summary.papers["paper-1"].error
    assert (tmp_path / "summary.json").exists()
