"""Research kernel: structured single-agent research loop and its contracts."""

from research_explorer.research.agent import (
    AgentOutputError,
    LLMResearchAgent,
    ResearchAgent,
    parse_agent_brief,
)
from research_explorer.research.answer import build_final_answer
from research_explorer.research.context import build_agent_prompt, build_evaluation_prompt
from research_explorer.research.evaluator import (
    CompositeEvaluator,
    DeterministicIntegrity,
    LLMRubricEvaluator,
)
from research_explorer.research.loop import (
    Acquisition,
    EvidenceGateway,
    GraphEvidenceGateway,
    KernelOptions,
    ResearchKernel,
)
from research_explorer.research.models import (
    ActionKind,
    AgentBrief,
    AgentNotebook,
    BudgetState,
    CandidateAction,
    Claim,
    ClaimMutation,
    ClaimStatus,
    EvidenceRef,
    FinalAnswer,
    OpenQuestion,
    OpenQuestionStatus,
    ProviderOutcome,
    ResearchAction,
    ResearchEvaluation,
    ResearchEvent,
    ResearchObjective,
    ResearchState,
    SlotState,
)
from research_explorer.research.policy import ExplorationPolicy, GreedyPolicy
from research_explorer.research.store import ResearchStore

__all__ = [
    "Acquisition",
    "ActionKind",
    "AgentBrief",
    "AgentNotebook",
    "AgentOutputError",
    "BudgetState",
    "CandidateAction",
    "Claim",
    "ClaimMutation",
    "ClaimStatus",
    "CompositeEvaluator",
    "DeterministicIntegrity",
    "EvidenceGateway",
    "EvidenceRef",
    "ExplorationPolicy",
    "FinalAnswer",
    "GraphEvidenceGateway",
    "GreedyPolicy",
    "KernelOptions",
    "LLMResearchAgent",
    "LLMRubricEvaluator",
    "OpenQuestion",
    "OpenQuestionStatus",
    "ProviderOutcome",
    "ResearchAction",
    "ResearchAgent",
    "ResearchEvaluation",
    "ResearchEvent",
    "ResearchKernel",
    "ResearchObjective",
    "ResearchState",
    "ResearchStore",
    "SlotState",
    "build_agent_prompt",
    "build_evaluation_prompt",
    "build_final_answer",
    "parse_agent_brief",
]
