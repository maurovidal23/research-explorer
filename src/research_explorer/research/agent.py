"""Agent boundary: turns a compact prompt into a validated ``AgentBrief``.

Malformed top-level and nested JSON is contained here: a bad entry is ignored,
and only a non-object payload or a transport failure aborts the turn (which the
controller records without terminating the run).
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import ValidationError

from research_explorer.agents.llm_client import LLMClient
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


class ResearchAgent(Protocol):
    async def propose_brief(
        self, objective: ResearchObjective, state: ResearchState, prompt: PromptBundle
    ) -> AgentBrief: ...


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
