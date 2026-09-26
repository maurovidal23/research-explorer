"""Prompt/context assembly from structured research models.

Prompts always carry the explicit research question, prefer the compact
notebook over full event history, and record which context identifiers were
included, omitted, or truncated plus an approximate input token count. Output
capacity is reserved rather than filling a fixed fraction of the window.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from research_explorer.research.models import (
    AgentBrief,
    ClaimStatus,
    OpenQuestionStatus,
    ResearchObjective,
    ResearchState,
)

PROMPT_TEMPLATE_VERSION = "research-kernel-agent/v1"
EVALUATION_TEMPLATE_VERSION = "research-kernel-eval/v1"

RUBRIC_DIMENSIONS = (
    "relevance",
    "coverage",
    "mechanistic_depth",
    "methodological_understanding",
    "evidence_traceability",
    "counterevidence",
    "uncertainty_calibration",
    "novelty",
    "redundancy",
)


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class PromptBundle:
    messages: list[dict[str, str]]
    template_version: str
    selected_ids: list[str] = field(default_factory=list)
    omitted_ids: list[str] = field(default_factory=list)
    approx_input_tokens: int = 0
    output_reserve: int = 0
    truncated: bool = False

    def as_event_payload(self) -> dict[str, Any]:
        return {
            "template_version": self.template_version,
            "selected_ids": list(self.selected_ids),
            "omitted_ids": list(self.omitted_ids),
            "approx_input_tokens": self.approx_input_tokens,
            "output_reserve": self.output_reserve,
            "truncated": self.truncated,
        }


def brief_schema_text() -> str:
    return json.dumps(
        {
            "action_summary": "str",
            "claim_mutations": [
                {
                    "op": "propose|support|dispute|reject|supersede|open_question|answer_question",
                    "claim_id": "str|null",
                    "question_id": "str|null",
                    "text": "str",
                    "status": "proposed|supported|disputed|rejected|superseded|null",
                    "confidence": "0..1|null",
                    "evidence": [
                        {"paper_id": "str", "locator": "str|null", "content_hash": "str|null"}
                    ],
                    "reason": "str",
                }
            ],
            "evidence": [{"paper_id": "str"}],
            "changed_understanding": "str",
            "contradictions": ["str"],
            "remaining_uncertainty": "str",
            "proposed_next_actions": ["str"],
            "shareable_findings": ["str"],
        },
        indent=2,
    )


def _compact_claims(state: ResearchState, limit: int) -> tuple[list[str], list[str]]:
    selected: list[str] = []
    omitted: list[str] = []
    accepted = [
        c for c in state.claims.values() if c.status is ClaimStatus.SUPPORTED
    ]
    others = [
        c for c in state.claims.values() if c.status is not ClaimStatus.SUPPORTED
    ]
    for claim in (accepted + others):
        if len(selected) >= limit:
            omitted.append(claim.id)
            continue
        selected.append(
            f"[{claim.id}] ({claim.status.value}, conf={claim.confidence:.2f}) {claim.text}"
        )
    return selected, omitted


def build_agent_prompt(
    objective: ResearchObjective,
    state: ResearchState,
    selected_paper_id: str,
    selected_title: str,
    *,
    selected_content: str = "",
    input_target: int = 6000,
    output_reserve: int = 1500,
    max_claims: int = 20,
    max_questions: int = 10,
) -> PromptBundle:
    selected_ids = [selected_paper_id]
    omitted_ids: list[str] = []

    claims, omitted_claims = _compact_claims(state, max_claims)
    omitted_ids.extend(omitted_claims)
    selected_ids.extend(c for c in state.claims if c not in omitted_claims)

    questions: list[str] = []
    for question in state.questions.values():
        if question.status in (OpenQuestionStatus.OPEN, OpenQuestionStatus.INVESTIGATING):
            if len(questions) >= max_questions:
                omitted_ids.append(question.id)
                continue
            questions.append(f"[{question.id}] (p={question.priority:.2f}) {question.text}")
            selected_ids.append(question.id)

    evidence_ids = sorted({ref.paper_id for ref in state.evidence})
    selected_ids.extend(evidence_ids[:50])
    if len(evidence_ids) > 50:
        omitted_ids.extend(evidence_ids[50:])

    content_limit = max(1000, input_target * 2)
    context_block = {
        "research_question": objective.question,
        "scope": objective.scope,
        "thesis": state.notebook.thesis,
        "claims": claims,
        "open_questions": questions,
        "selected_source": {
            "paper_id": selected_paper_id,
            "title": selected_title,
            "content": selected_content[:content_limit],
        },
        "evidence_papers": evidence_ids[:50],
        "planned_actions": [a.model_dump(mode="json") for a in state.notebook.planned_actions],
    }
    system = (
        "You are a rigorous research explorer. Return ONLY a JSON object matching "
        "the AgentBrief schema below. Never present an unsupported claim as fact.\n\n"
        f"AgentBrief schema:\n{brief_schema_text()}"
    )
    user = (
        f"Research question (must be addressed): {objective.question}\n\n"
        f"Current structured context:\n{json.dumps(context_block, indent=2, sort_keys=True)}\n\n"
        "Propose at most one new claim per turn and attach concrete evidence paper ids."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    serialized = json.dumps(messages, sort_keys=True)
    tokens = approx_tokens(serialized)

    truncated = False
    if tokens > max(input_target, 1):
        # Keep the question and selected source; drop planned actions first.
        context_block["planned_actions"] = []
        truncated = True
        user = (
            f"Research question (must be addressed): {objective.question}\n\n"
            f"Compacted context:\n{json.dumps(context_block, indent=2, sort_keys=True)}"
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        tokens = approx_tokens(json.dumps(messages, sort_keys=True))

    return PromptBundle(
        messages=messages,
        template_version=PROMPT_TEMPLATE_VERSION,
        selected_ids=selected_ids,
        omitted_ids=omitted_ids,
        approx_input_tokens=tokens,
        output_reserve=output_reserve,
        truncated=truncated,
    )


def build_evaluation_prompt(
    objective: ResearchObjective,
    state: ResearchState,
    *,
    input_target: int = 6000,
    output_reserve: int = 800,
) -> PromptBundle:
    claims = [c.model_dump(mode="json") for c in state.claims.values()]
    questions = [q.model_dump(mode="json") for q in state.questions.values()]
    context_block = {
        "research_question": objective.question,
        "claims": claims,
        "open_questions": questions,
        "evidence_papers": sorted({ref.paper_id for ref in state.evidence}),
    }
    system = (
        "You are an impartial research evaluator. Score each rubric dimension in "
        "[0,1] and return JSON: "
        '{"dimension_scores": {'
        + ", ".join(f'"{d}": 0.0' for d in RUBRIC_DIMENSIONS)
        + "}, \"missing_knowledge\": [], \"unsupported_claims\": [], "
        '"contradictions": [], "recommended_questions": []}. '
        "You are not told any controller score or policy name."
    )
    user = (
        f"Research question: {objective.question}\n\n"
        f"Evaluate this state:\n{json.dumps(context_block, indent=2, sort_keys=True)}"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return PromptBundle(
        messages=messages,
        template_version=EVALUATION_TEMPLATE_VERSION,
        selected_ids=sorted(state.claims) + sorted(state.questions),
        approx_input_tokens=approx_tokens(json.dumps(messages, sort_keys=True)),
        output_reserve=output_reserve,
    )


__all__ = [
    "EVALUATION_TEMPLATE_VERSION",
    "PROMPT_TEMPLATE_VERSION",
    "RUBRIC_DIMENSIONS",
    "AgentBrief",
    "PromptBundle",
    "build_agent_prompt",
    "build_evaluation_prompt",
]
