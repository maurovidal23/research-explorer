from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from research_explorer.fixed_benchmark.models import BenchmarkManifest, BenchmarkPaper
from research_explorer.redaction import redact_secrets


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PaperRunResult(BaseModel):
    paper_id: str
    benchmark_version: str
    source_sha256: str
    config_fingerprint: str
    status: str
    started_at: str
    finished_at: str = ""
    run_id: str = ""
    survivor_id: str = ""
    survivor_state_hash: str = ""
    survivor_bundle_path: str = ""
    report_path: str = ""
    error: str = ""


class BatchSummary(BaseModel):
    benchmark_id: str
    benchmark_version: str
    config_fingerprint: str
    created_at: str = Field(default_factory=utc_now)
    total: int = 0
    completed: int = 0
    failed: int = 0
    pending: int = 0
    papers: dict[str, PaperRunResult] = Field(default_factory=dict)


PaperExecutor = Callable[[BenchmarkPaper, Path], Awaitable[PaperRunResult]]


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-")
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_result(path: Path, result: PaperRunResult) -> None:
    _atomic_write(path, result.model_dump_json(indent=2) + "\n")


def _load_result(path: Path) -> PaperRunResult | None:
    if not path.exists():
        return None
    try:
        return PaperRunResult.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


class FixedResearchBatchRunner:
    def __init__(
        self,
        manifest: BenchmarkManifest,
        output_dir: str | Path,
        config_fingerprint: str,
        executor: PaperExecutor,
        max_concurrent: int = 2,
        retry_failed: bool = False,
    ) -> None:
        self.manifest = manifest
        self.output_dir = Path(output_dir)
        self.config_fingerprint = config_fingerprint
        self.executor = executor
        self.max_concurrent = max(1, max_concurrent)
        self.retry_failed = retry_failed

    def _reusable(self, paper: BenchmarkPaper, result: PaperRunResult | None) -> bool:
        if result is None:
            return False
        if result.status == "failed" and not self.retry_failed:
            return True
        return bool(
            result.status == "completed"
            and result.benchmark_version == self.manifest.version
            and result.source_sha256 == paper.source_sha256
            and result.config_fingerprint == self.config_fingerprint
            and result.survivor_bundle_path
            and Path(result.survivor_bundle_path).exists()
        )

    async def _run_paper(
        self, paper: BenchmarkPaper, semaphore: asyncio.Semaphore
    ) -> PaperRunResult:
        paper_dir = self.output_dir / "papers" / paper.paper_id
        result_path = paper_dir / "run.json"
        existing = _load_result(result_path)
        if self._reusable(paper, existing):
            assert existing is not None
            return existing
        async with semaphore:
            started = PaperRunResult(
                paper_id=paper.paper_id,
                benchmark_version=self.manifest.version,
                source_sha256=paper.source_sha256,
                config_fingerprint=self.config_fingerprint,
                status="running",
                started_at=utc_now(),
            )
            _write_result(result_path, started)
            try:
                result = await self.executor(paper, paper_dir)
                if result.status != "completed" or not result.survivor_bundle_path:
                    raise RuntimeError("paper executor did not produce a survivor bundle")
                if not Path(result.survivor_bundle_path).exists():
                    raise RuntimeError("reported survivor bundle does not exist")
            except Exception as exc:
                result = started.model_copy(
                    update={
                        "status": "failed",
                        "finished_at": utc_now(),
                        "error": redact_secrets(str(exc)),
                    }
                )
            _write_result(result_path, result)
            return result

    async def run(self) -> BatchSummary:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        semaphore = asyncio.Semaphore(self.max_concurrent)
        summary = BatchSummary(
            benchmark_id=self.manifest.benchmark_id,
            benchmark_version=self.manifest.version,
            config_fingerprint=self.config_fingerprint,
            total=len(self.manifest.papers),
            pending=len(self.manifest.papers),
        )
        _atomic_write(
            self.output_dir / "summary.json", summary.model_dump_json(indent=2) + "\n"
        )
        tasks = [
            asyncio.create_task(self._run_paper(paper, semaphore))
            for paper in self.manifest.papers
        ]
        for task in asyncio.as_completed(tasks):
            result = await task
            summary.papers[result.paper_id] = result
            summary.completed = sum(
                item.status == "completed" for item in summary.papers.values()
            )
            summary.failed = sum(
                item.status == "failed" for item in summary.papers.values()
            )
            summary.pending = summary.total - len(summary.papers)
            _atomic_write(
                self.output_dir / "summary.json",
                summary.model_dump_json(indent=2) + "\n",
            )
        return summary
