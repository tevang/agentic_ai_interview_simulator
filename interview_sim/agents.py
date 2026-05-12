from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .config import AppConfig, InterviewerConfig
from .llm import LLMClient
from .models import EVALUATION_SCHEMA, QUESTION_SCHEMA
from .text_utils import normalize_name
from .vector_store import MemoryStore


@dataclass
class InterviewTurn:
    interviewer: str
    question: str
    expected_answer: str
    candidate_answer: str = ""
    evaluation: Dict[str, Any] = field(default_factory=dict)


class InterviewerAgent:
    def __init__(
        self,
        interviewer: InterviewerConfig,
        config: AppConfig,
        memory: MemoryStore,
        llm: LLMClient,
    ):
        self.interviewer = interviewer
        self.config = config
        self.memory = memory
        self.llm = llm
        self.asked_questions: list[str] = []

    @property
    def name(self) -> str:
        return self.interviewer.name

    def ask_question(self, transcript: List[InterviewTurn]) -> Dict[str, Any]:
        context = self._retrieve_context(transcript)
        asked = "\n".join(f"- {q}" for q in self.asked_questions[-8:]) or "No questions asked yet."
        transcript_summary = self._transcript_summary(transcript)

        system = f"""
You are an interview-simulation agent. You emulate the likely interviewing perspective of {self.name},
not the real person. Use only the supplied context and do not invent private facts.

Interviewer role hint: {self.interviewer.role_hint}
Interviewer style hint: {self.interviewer.style_hint}

Global interview goal:
{self.config.prompts.interview_goal}

Rules:
- Ask exactly one question.
- The question must fit this interviewer's background and likely priorities.
- React to the candidate's previous answers when useful.
- For a first-round interview, favor background, scope of experience, architecture judgment, and light technical probes.
- Do not ask trivia. Ask questions that reveal senior engineering judgment.
- Include the expected answer and a concise rubric so the program can evaluate the candidate after they answer.
""".strip()

        user = f"""
Relevant retrieved context:
{context}

Previous transcript summary:
{transcript_summary}

Questions already asked by this interviewer:
{asked}

Generate the next interview question as JSON.
""".strip()

        question = self.llm.json(
            model=self.config.models.chat_model,
            system=system,
            user=user,
            schema=QUESTION_SCHEMA,
            schema_name="interview_question",
            temperature=self.config.models.temperature_question,
        )

        self.asked_questions.append(question["question"])
        return question

    def evaluate_answer(
        self,
        question: Dict[str, Any],
        candidate_answer: str,
        transcript: List[InterviewTurn],
    ) -> Dict[str, Any]:
        context = self._retrieve_context(transcript)

        system = f"""
You are evaluating a candidate answer in an interview simulation for {self.name}.
Be fair, specific, and useful. Evaluate against the expected answer and rubric,
while giving credit for equivalent wording or valid alternative designs.
Return JSON only.
""".strip()

        user = f"""
Context:
{context}

Question:
{question['question']}

Expected answer prepared by interviewer agent:
{question['expected_answer']}

Rubric:
{question['evaluation_rubric']}

Candidate answer:
{candidate_answer}

Score from 0.0 to 1.0. Include corrected_or_expected_answer that the candidate can study.
""".strip()

        evaluation = self.llm.json(
            model=self.config.models.evaluator_model,
            system=system,
            user=user,
            schema=EVALUATION_SCHEMA,
            schema_name="answer_evaluation",
            temperature=self.config.models.temperature_evaluation,
        )

        score = max(0.0, min(1.0, float(evaluation.get("score", 0.0))))
        evaluation["score"] = score
        return evaluation

    def _retrieve_context(self, transcript: List[InterviewTurn]) -> str:
        retrieval_k = self.config.runtime.retrieval_k
        name_key = normalize_name(self.name)
        role = self.interviewer.role_hint
        latest = transcript[-1].candidate_answer if transcript else ""

        interviewer_context = self.memory.query(
            f"{self.name} {role} interviewer profile professional experience likely interview focus",
            n_results=retrieval_k,
            where={"owner_name": name_key},
        )

        job_context = self.memory.query(
            "Senior AI Developer job responsibilities skills RAG LLM Python MLOps APIs observability",
            n_results=retrieval_k,
            where={"owner_type": "job"},
        )

        candidate_context = self.memory.query(
            f"candidate CV strengths gaps AI ML Python software drug discovery leadership {latest}",
            n_results=retrieval_k,
            where={"owner_type": "candidate"},
        )

        interview_context = self.memory.query(
            "interview character first round agenda technical depth background discussion",
            n_results=3,
            where={"owner_type": "interview"},
        )

        company_context = self.memory.query(
            "MSD Czech Republic AI engineering healthcare production ready systems company culture Prague",
            n_results=4,
            where={"owner_type": "company"},
        )

        general_research = self.memory.query(
            "production RAG LLM vector database evaluation MLOps observability senior AI developer",
            n_results=4,
        )

        return "\n\n".join(
            section
            for section in [
                "## Interviewer context\n" + interviewer_context,
                "## Job context\n" + job_context,
                "## Candidate context\n" + candidate_context,
                "## Interview character context\n" + interview_context,
                "## Company context\n" + company_context,
                "## Technical research context\n" + general_research,
            ]
            if section.strip()
        )

    @staticmethod
    def _transcript_summary(transcript: List[InterviewTurn]) -> str:
        if not transcript:
            return "No previous turns."

        lines = []
        for turn in transcript[-8:]:
            score = turn.evaluation.get("score", "n/a") if turn.evaluation else "n/a"
            feedback = turn.evaluation.get("feedback_to_candidate", "") if turn.evaluation else ""
            lines.append(
                f"{turn.interviewer}: Q={turn.question}\n"
                f"Candidate answer={turn.candidate_answer[:700]}\n"
                f"Score={score}; feedback={feedback}"
            )

        return "\n\n".join(lines)