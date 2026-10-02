from __future__ import annotations

import json
from pathlib import Path

from research_explorer.fixed_benchmark.models import BenchmarkManifest, BenchmarkQuestion


def load_manifest(path: str | Path) -> BenchmarkManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return BenchmarkManifest.model_validate(payload)


def load_bank(path: str | Path) -> list[BenchmarkQuestion]:
    questions: list[BenchmarkQuestion] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                questions.append(BenchmarkQuestion.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"invalid question at line {line_number}: {exc}") from exc
    return questions
