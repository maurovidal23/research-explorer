"""Agent boundary: turns a compact prompt into a validated ``AgentBrief``.

Malformed top-level and nested JSON is contained here: a bad entry is ignored,
and only a non-object payload or a transport failure aborts the turn (which the
controller records without terminating the run).
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import ValidationError

from research_explorer.agents.llm_client import LLMClient
from research_explorer.graph.models import Paper, PaperSummary
from research_explorer.research.context import PromptBundle
from research_explorer.research.models import (
    AgentBrief,
    ClaimMutation,
    EvidenceRef,
    ResearchObjective,
    ResearchState,
)


class AgentOutputError(Exception):
    """Raised when the agent produced unusable (non-object) output."""


class ReferenceMappingError(Exception):
    """Raised when the reference mapper cannot produce usable output."""


def reference_schema() -> dict[str, Any]:
    """JSON schema for schema-constrained reference-mapper output."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "references": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "authors": {"type": "array", "items": {"type": "string"}},
                        "year": {"type": ["integer", "null"]},
                        "arxiv_id": {"type": ["string", "null"]},
                        "doi": {"type": ["string", "null"]},
                    },
                    "required": ["title", "authors", "year", "arxiv_id", "doi"],
                },
            }
        },
        "required": ["references"],
    }


class ResearchAgent(Protocol):
    async def propose_brief(
        self, objective: ResearchObjective, state: ResearchState, prompt: PromptBundle
    ) -> AgentBrief: ...


class ReferenceMapper(Protocol):
    async def map_references(
        self, paper: Paper, question: str, limit: int
    ) -> list[PaperSummary]: ...


def parse_agent_brief(raw: Any) -> AgentBrief:
    """Validate top-level and nested JSON types, ignoring malformed entries."""
    if not isinstance(raw, dict):
        raise AgentOutputError(f"agent output must be an object, got {type(raw).__name__}")

    def _text(value: Any) -> str:
        return value if isinstance(value, str) else ""

    def _text_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str)]

    mutations: list[ClaimMutation] = []
    raw_mutations = raw.get("claim_mutations")
    if isinstance(raw_mutations, list):
        for entry in raw_mutations:
            if not isinstance(entry, dict):
                continue
            try:
                mutations.append(ClaimMutation.model_validate(entry))
            except ValidationError:
                continue

    evidence: list[EvidenceRef] = []
    raw_evidence = raw.get("evidence")
    if isinstance(raw_evidence, list):
        for entry in raw_evidence:
            if not isinstance(entry, dict):
                continue
            try:
                evidence.append(EvidenceRef.model_validate(entry))
            except ValidationError:
                continue

    return AgentBrief(
        action_summary=_text(raw.get("action_summary")),
        claim_mutations=mutations,
        evidence=evidence,
        changed_understanding=_text(raw.get("changed_understanding")),
        contradictions=_text_list(raw.get("contradictions")),
        remaining_uncertainty=_text(raw.get("remaining_uncertainty")),
        proposed_next_actions=_text_list(raw.get("proposed_next_actions")),
        shareable_findings=_text_list(raw.get("shareable_findings")),
    )


class LLMResearchAgent:
    """LLM-backed agent returning a validated AgentBrief."""

    def __init__(
        self,
        llm: LLMClient,
        model: str = "qwen3.6",
        temperature: float = 0.4,
        max_tokens: int = 1500,
    ) -> None:
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def propose_brief(
        self, objective: ResearchObjective, state: ResearchState, prompt: PromptBundle
    ) -> AgentBrief:
        try:
            raw = await self.llm.chat_json(
                prompt.messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=min(self.max_tokens, prompt.output_reserve or self.max_tokens),
            )
        except Exception as exc:  # transport/decoding failures are contained by the caller
            raise AgentOutputError(str(exc)) from exc
        return parse_agent_brief(raw)


class LLMReferenceMapper:
    def __init__(
        self,
        llm: LLMClient,
        model: str = "qwen3.6",
        max_tokens: int = 2000,
        doi_provider: str | None = None,
        max_attempts: int = 2,
    ) -> None:
        self.llm = llm
        self.model = model
        self.max_tokens = max_tokens
        self.doi_provider = doi_provider
        self.max_attempts = max(1, max_attempts)

    async def map_references(
        self, paper: Paper, question: str, limit: int
    ) -> list[PaperSummary]:
        bibliography = paper.ref_entries[:limit]
        if not bibliography:
            return []
        messages = [
            {
                "role": "system",
                "content": (
                    "Extract only real works present in the supplied bibliography. "
                    "Return a JSON object with a references array. Each item may contain "
                    "title, authors, year, arxiv_id, and doi. Never invent identifiers."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Research question: {question}\n"
                    f"Source paper: {paper.title}\n"
                    f"Bibliography:\n" + "\n".join(bibliography)
                ),
            },
        ]
        raw: dict | None = None
        last_error: Exception | None = None
        for _ in range(self.max_attempts):
            try:
                raw = await self.llm.chat_json(
                    messages,
                    model=self.model,
                    schema=reference_schema(),
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                )
                break
            except Exception as exc:
                last_error = exc
                raw = None
        if raw is None:
            raise ReferenceMappingError(str(last_error)) from last_error
        entries = raw.get("references") if isinstance(raw, dict) else None
        if not isinstance(entries, list):
            raise ReferenceMappingError("reference mapping output lacked a references array")
        summaries: list[PaperSummary] = []
        for entry in entries:
            summary = self._parse_reference(entry)
            if summary is not None:
                summaries.append(summary)
            if len(summaries) >= limit:
                break
        return summaries

    def _parse_reference(self, entry: Any) -> PaperSummary | None:
        if not isinstance(entry, dict):
            return None
        title = entry.get("title")
        if not isinstance(title, str) or not title.strip():
            return None
        authors = entry.get("authors")
        if isinstance(authors, str):
            authors = [part.strip() for part in authors.split(",") if part.strip()]
        if not isinstance(authors, list):
            authors = []
        authors = [author for author in authors if isinstance(author, str)]
        year = entry.get("year")
        if isinstance(year, str) and year.isdigit():
            year = int(year)
        if not isinstance(year, int):
            year = None
        arxiv_id = entry.get("arxiv_id")
        if isinstance(arxiv_id, str) and arxiv_id.strip():
            native = arxiv_id.strip()
            if native.lower().startswith("arxiv:"):
                native = native.split(":", 1)[1]
            return PaperSummary(
                id=native,
                arxiv_id=native,
                title=title.strip(),
                year=year,
                authors=authors,
                provider="arxiv",
            )
        doi = entry.get("doi")
        if self.doi_provider and isinstance(doi, str) and doi.strip():
            native = doi.strip()
            return PaperSummary(
                id=native,
                doi=native,
                title=title.strip(),
                year=year,
                authors=authors,
                provider=self.doi_provider,
            )
        return None
