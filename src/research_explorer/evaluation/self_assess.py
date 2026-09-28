"""Self-assessment component (S) of the quality function Q.

The agent evaluates its own narrative. Uses the same model as the explorer.
"""

from __future__ import annotations

from research_explorer.agents.llm_client import LLMClient
from research_explorer.agents.prompts import self_assess
from research_explorer.config import Config
from research_explorer.evaluation import availability
from research_explorer.logging_setup import get_logger
from research_explorer.replay.models import SelfAssessmentDetail

log = get_logger("eval.self")

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}


class SelfAssessment:
    """Self-assessment (S) — the agent judges its own narrative."""

    def __init__(self, llm: LLMClient, config: Config):
        self.llm = llm
        self.cfg = config

    async def score(self, narrative: str, seed_query: str) -> float:
        """Return S ∈ [0, 1] — the agent's self-assessment of its narrative."""
        return (await self.score_detail(narrative, seed_query)).score

    async def score_detail(self, narrative: str, seed_query: str) -> SelfAssessmentDetail:
        """Return the S component with the agent's explicit reasoning."""
        if not narrative.strip():
            return SelfAssessmentDetail(
                score=0.0,
                reasoning="empty narrative",
                available=False,
                unavailable_reason=availability.REASON_EMPTY_NARRATIVE,
            )
        messages = self_assess(narrative, seed_query)
        try:
            result = await self.llm.chat_json(
                messages,
                model=self.cfg.llm.explorer_model,
                schema=SCORE_SCHEMA,
                temperature=0.3,
                max_tokens=1000,
                purpose="self_assessment",
            )
            score = float(result.get("score", 0.5))
            score = max(0.0, min(1.0, score))
            reasoning = str(result.get("reasoning", ""))
            return SelfAssessmentDetail(score=score, reasoning=reasoning)
        except Exception as e:
            log.warning("self_assess_failed", error=str(e))
            return SelfAssessmentDetail(
                score=0.0,
                reasoning=f"self assessment failed: {e}",
                available=False,
                unavailable_reason=availability.classify_failure(e),
            )
