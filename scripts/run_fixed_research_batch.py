#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import copy
import dataclasses
import os
from pathlib import Path

from research_explorer.config import Config, load_config
from research_explorer.examination import SelectionWeights, build_evidence_pack
from research_explorer.examination.benchmark import config_fingerprint
from research_explorer.fixed_benchmark.batch import (
    FixedResearchBatchRunner,
    PaperRunResult,
    utc_now,
)
from research_explorer.fixed_benchmark.io import load_manifest
from research_explorer.fixed_benchmark.models import BenchmarkPaper
from research_explorer.memory.extract import memory_from_state
from research_explorer.orchestrator.runner import Orchestrator
from research_explorer.survivor.bundle import build_bundle
from research_explorer.survivor.models import SelectionMetadata

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks" / "research_explorer_v1" / "manifest.json"


def _paper_config(base: Config, paper_dir: Path) -> Config:
    config = copy.deepcopy(base)
    storage = paper_dir / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    config.storage.db_path = str(storage / "graph.db")
    config.storage.trace_db_path = str(storage / "replay.db")
    config.storage.research_db_path = str(storage / "research.db")
    config.storage.cache_dir = str(storage / "cache")
    config.examination.enabled = False
    return config


def _freeze_process_winner(orch: Orchestrator):
    winner = orch.colony.best_agent
    if winner is None:
        raise RuntimeError("research run completed without a winning agent")
    memory = memory_from_state(winner.state)
    if not memory.dossiers:
        raise RuntimeError("winning agent has no evidence-bearing dossier")
    seed_id = orch.colony.seed_id
    pack = build_evidence_pack(
        seed_id,
        orch.effective_scope,
        memory.dossiers,
    )
    if not pack.sources:
        raise RuntimeError("winning agent has no eligible evidence source")
    selection = SelectionMetadata(
        terminal_score=winner.state.quality,
        process_score=winner.state.quality,
        weights=SelectionWeights(selection=0.0, process=1.0, grounding=0.0),
        formula="Q_process",
        candidate_ranking=[winner.state.id],
        process_peak_agent=winner.state.id,
    )
    synthesis = winner.state.synthesis or winner.state.narrative
    return build_bundle(
        agent_id=winner.state.id,
        memory=memory,
        synthesis=synthesis,
        pack=pack,
        selection=selection,
        config_fingerprint=config_fingerprint(dataclasses.asdict(orch.cfg)),
        model_ids={
            "explorer_model": orch.cfg.llm.explorer_model,
            "judge_model": orch.cfg.llm.judge_model,
        },
        prompt_versions={"research": "v1"},
        schema_versions={"bundle": "survivor-bundle/1", "benchmark": "fixed/1"},
    )


def _executor(base: Config, fingerprint: str, benchmark_version: str):
    async def execute(paper: BenchmarkPaper, paper_dir: Path) -> PaperRunResult:
        started_at = utc_now()
        attempt_dir = paper_dir / "attempts" / fingerprint
        config = _paper_config(base, attempt_dir)
        orch = Orchestrator(config)
        try:
            await orch.run(paper.versioned_arxiv_id, "")
            bundle = orch.survivor_bundle or _freeze_process_winner(orch)
            report = orch.generate_report(paper.versioned_arxiv_id, "")
            bundle_path = (attempt_dir / "survivor_bundle.private.json").resolve()
            report_path = (attempt_dir / "research_report.md").resolve()
            bundle_path.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
            os.chmod(bundle_path, 0o600)
            report_path.write_text(report, encoding="utf-8")
            return PaperRunResult(
                paper_id=paper.paper_id,
                benchmark_version=benchmark_version,
                source_sha256=paper.source_sha256,
                config_fingerprint=fingerprint,
                status="completed",
                started_at=started_at,
                finished_at=utc_now(),
                run_id=orch.run_id,
                survivor_id=bundle.survivor_id,
                survivor_state_hash=bundle.state_hash,
                survivor_bundle_path=str(bundle_path),
                report_path=str(report_path),
            )
        finally:
            await orch.aclose()

    return execute


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--config", default="config/profiles/opencode_exam.toml")
    parser.add_argument("--output", default="data/fixed_benchmark/runs")
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--max-fetches", type=int, default=0)
    parser.add_argument("--max-reference-entries", type=int, default=0)
    parser.add_argument("--reference-mapping-tokens", type=int, default=0)
    parser.add_argument("--arxiv-period", type=int, default=0)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    config = load_config(args.config)
    config.examination.enabled = False
    if args.max_fetches > 0:
        config.budget.max_fetches = args.max_fetches
    if args.max_reference_entries > 0:
        config.reference_mapping.max_entries = args.max_reference_entries
    if args.reference_mapping_tokens > 0:
        config.reference_mapping.max_completion_tokens_per_batch = (
            args.reference_mapping_tokens
        )
    if args.arxiv_period > 0 and "arxiv" in config.providers.entries:
        config.providers.entries["arxiv"].period = args.arxiv_period
    fingerprint = config_fingerprint(dataclasses.asdict(config))
    executor = _executor(config, fingerprint, manifest.version)
    runner = FixedResearchBatchRunner(
        manifest=manifest,
        output_dir=args.output,
        config_fingerprint=fingerprint,
        executor=executor,
        max_concurrent=args.max_concurrent,
        retry_failed=args.retry_failed,
    )
    summary = asyncio.run(runner.run())
    print(
        f"fixed research batch: {summary.completed}/{summary.total} completed, "
        f"{summary.failed} failed"
    )
    if summary.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
