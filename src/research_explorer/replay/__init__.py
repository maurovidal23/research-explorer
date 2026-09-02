"""Evaluation replay: typed evaluation records, run trace store, and the
FastAPI + static web replay UI."""

from research_explorer.replay.models import DetailedEvaluation
from research_explorer.replay.trace import RunTracer, RunTraceStore

__all__ = ["DetailedEvaluation", "RunTraceStore", "RunTracer"]
