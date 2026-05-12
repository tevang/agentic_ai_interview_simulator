from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ProjectConfig(BaseModel):
    name: str = "interview_practice"
    storage_dir: str = "./.interview_memory"
    log_dir: str = "./session_logs"


class ModelConfig(BaseModel):
    chat_model: str = "gpt-4o"
    evaluator_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    temperature_question: float = 0.45
    temperature_evaluation: float = 0.1


class RuntimeConfig(BaseModel):
    max_interviewers: int = 3
    max_questions: int = 10
    question_order: str = "round_robin"
    print_expected_answer: bool = True
    allow_followups: bool = True
    passing_score: float = 0.75
    retrieval_k: int = 6

    @field_validator("max_interviewers")
    @classmethod
    def validate_max_interviewers(cls, value: int) -> int:
        if value < 1 or value > 3:
            raise ValueError("max_interviewers must be between 1 and 3")
        return value


class CompanyConfig(BaseModel):
    name: str
    website: str


class CandidateConfig(BaseModel):
    name: str
    cv_pdf: str


class InterviewerConfig(BaseModel):
    name: str
    role_hint: str = ""
    style_hint: str = ""
    profile_pdfs: List[str]
    extra_pdfs: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_at_least_one_profile_pdf(self):
        if not self.profile_pdfs:
            raise ValueError(f"Interviewer {self.name} must have at least one profile PDF")
        return self


class DocumentConfig(BaseModel):
    company: CompanyConfig
    job_description_pdf: str
    interview_character_pdf: Optional[str] = None
    candidate: CandidateConfig
    interviewers: List[InterviewerConfig]

    @field_validator("interviewers")
    @classmethod
    def validate_interviewer_count(cls, value: List[InterviewerConfig]) -> List[InterviewerConfig]:
        if not value:
            raise ValueError("At least one interviewer is required")
        if len(value) > 3:
            raise ValueError("A maximum of three interviewers is supported")
        return value


class WebResearchConfig(BaseModel):
    enabled: bool = True
    provider: str = "tavily"
    search_depth: str = "basic"
    max_results_per_query: int = 4
    include_answer: bool = True
    include_raw_content: bool = False
    max_job_skill_queries: int = 8
    max_interviewer_skill_queries: int = 3
    skip_research_if_interviewer_registered: bool = True
    company_queries: List[str] = Field(default_factory=list)


class GuardrailsConfig(BaseModel):
    forbid_name_search: bool = True
    candidate_names: List[str] = Field(default_factory=list)
    interviewer_names: List[str] = Field(default_factory=list)
    forbidden_extra_terms: List[str] = Field(default_factory=list)


class PromptConfig(BaseModel):
    interview_goal: str = "Emulate a realistic interview."


class AppConfig(BaseModel):
    project: ProjectConfig
    models: ModelConfig
    runtime: RuntimeConfig
    documents: DocumentConfig
    web_research: WebResearchConfig = Field(default_factory=WebResearchConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    prompts: PromptConfig = Field(default_factory=PromptConfig)

    @model_validator(mode="after")
    def sync_interviewer_names(self):
        names = [i.name for i in self.documents.interviewers]
        merged = list(dict.fromkeys([*self.guardrails.interviewer_names, *names]))
        self.guardrails.interviewer_names = merged
        if len(names) > self.runtime.max_interviewers:
            raise ValueError("documents.interviewers exceeds runtime.max_interviewers")
        return self


def load_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)