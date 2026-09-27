"""Evaluation replay: typed evaluation records, run trace store, and the
FastAPI + static web replay UI.

``RunTracer``/``RunTraceStore`` are exposed lazily to avoid an import cycle:
the trace store imports the UI-independent event models, which in turn import
:mod:`research_explorer.replay.models`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from research_explorer.replay.models import (
    CandidateScore,
    CandidateSelection,
    DetailedEvaluation,
)

if TYPE_CHECKING:
    from research_explorer.replay.trace import RunTracer, RunTraceStore

__all__ = [
    "CandidateScore",
    "CandidateSelection",
    "DetailedEvaluation",
    "RunTraceStore",
    "RunTracer",
]


def __getattr__(name: str):
    if name in ("RunTracer", "RunTraceStore"):
        from research_explorer.replay.trace import RunTracer, RunTraceStore

        return {"RunTracer": RunTracer, "RunTraceStore": RunTraceStore}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
