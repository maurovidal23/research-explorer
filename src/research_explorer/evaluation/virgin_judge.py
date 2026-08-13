"""Virgin judge component (J) of the quality function Q.

A fresh LLM with no accumulated context evaluates the narrative impartially.
Uses a DIFFERENT model than the explorers to reduce shared bias.
"""

from __future__ import annotations

from research_explorer.agents.llm_client import LLMClient
from research_explorer.agents.prompts import virgin_judge
from research_explorer.config import Config
from research_explorer.logging_setup import get_logger

log = get_logger("eval.virgin")

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "coverage": {"type": "string"},
        "gaps": {"type": "string"},
    },
    "required": ["score", "coverage", "gaps"],
    "additionalProperties": False,
}


class VirginJudge:
    """Virgin judge (J) — impartial LLM evaluates the narrative with no prior context."""

    def __init__(self, llm: LLMClient, config: Config):
        self.llm = llm
        self.cfg = config

    async def judge(self, narrative: str, seed_query: str) -> float:
        """Return J ∈ [0, 1] — the virgin judge's impartial score."""
        if not narrative.strip():
            return 0.0
        messages = virgin_judge(narrative, seed_query)
        try:
            result = await self.llm.chat_json(
                messages,
                model=self.cfg.llm.judge_model,  # different model — reduces bias
                schema=JUDGE_SCHEMA,
                temperature=0.2,
                max_tokens=1500,
            )
            score = float(result.get("score", 0.5))
            return max(0.0, min(1.0, score))
        except Exception as e:
            log.warning("virgin_judge_failed", error=str(e))
            return 0.5
