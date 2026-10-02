from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class BenchmarkPaper(BaseModel):
    paper_id: str
    arxiv_id: str
    version: int = Field(ge=1)
    title: str
    cluster: str
    abstract_url: str
    pdf_url: str
    source_sha256: str = ""
    text_sha256: str = ""
    page_count: int = Field(default=0, ge=0)

    @property
    def versioned_arxiv_id(self) -> str:
        return f"{self.arxiv_id}v{self.version}"


class QuestionOption(BaseModel):
    option_id: str
    text: str = Field(min_length=1)


class EvidenceLocator(BaseModel):
    paper_id: str
    page: int = Field(ge=1)
    section: str = ""
    quote: str = Field(min_length=1)


class BenchmarkQuestion(BaseModel):
    question_id: str
    paper_id: str
    category: str
    difficulty: str
    question: str = Field(min_length=1)
    options: list[QuestionOption]
    correct_option_ids: list[str]
    evidence: list[EvidenceLocator]
    rationale: str = Field(min_length=1)
    related_paper_ids: list[str] = Field(default_factory=list)
    trap_type: str = ""
    generator_model: str = ""
    critic_model: str = ""

    @field_validator("correct_option_ids")
    @classmethod
    def unique_keys(cls, value: list[str]) -> list[str]:
        return sorted(set(value))

    def public_payload(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "paper_id": self.paper_id,
            "category": self.category,
            "difficulty": self.difficulty,
            "question": self.question,
            "options": [option.model_dump() for option in self.options],
            "response_contract": "select_one_or_more",
        }


class CriticVerdict(BaseModel):
    question_id: str
    accepted: bool
    defensible_option_ids: list[str] = Field(default_factory=list)
    evidence_supported: bool = False
    distractors_valid: bool = False
    issue: str = ""


class BenchmarkManifest(BaseModel):
    benchmark_id: str
    version: str
    description: str
    question_count_per_paper: int = Field(default=100, ge=1)
    options_per_question: int = Field(default=5, ge=2)
    papers: list[BenchmarkPaper]
    categories: dict[str, int]
    difficulties: dict[str, int]
    correct_option_counts: dict[str, int]
    generator_model: str
    critic_model: str
    prompt_version: str
    created_at: str
    bank_sha256: str = ""

    @property
    def expected_question_count(self) -> int:
        return len(self.papers) * self.question_count_per_paper
