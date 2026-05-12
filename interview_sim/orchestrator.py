from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .agents import InterviewTurn, InterviewerAgent
from .config import AppConfig
from .llm import LLMClient
from .vector_store import MemoryStore


class InterviewOrchestrator:
    def __init__(self, config: AppConfig, memory: MemoryStore, console: Console | None = None):
        self.config = config
        self.memory = memory
        self.console = console or Console()
        self.llm = LLMClient()
        self.agents = [
            InterviewerAgent(i, config, memory, self.llm)
            for i in config.documents.interviewers
        ]
        self.transcript: List[InterviewTurn] = []

    def run(self) -> None:
        self._print_intro()

        question_count = 0
        agent_index = 0

        while question_count < self.config.runtime.max_questions:
            agent = self.agents[agent_index % len(self.agents)]
            agent_index += 1
            question_count += 1

            question = agent.ask_question(self.transcript)

            self.console.print(
                Panel.fit(
                    f"[bold]{agent.name}[/bold] ({question['question_type']})\n\n"
                    f"{question['question']}\n\n"
                    f"[dim]Why this interviewer asks: {question['why_this_interviewer_asks']}[/dim]",
                    title=f"Question {question_count}/{self.config.runtime.max_questions}",
                )
            )

            try:
                answer = Prompt.ask("Your answer (type 'quit' to end)")
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[yellow]Interview ended before an answer was entered.[/yellow]")
                break

            if answer.strip().lower() in {"quit", "exit", "q"}:
                break

            evaluation = agent.evaluate_answer(question, answer, self.transcript)

            turn = InterviewTurn(
                interviewer=agent.name,
                question=question["question"],
                expected_answer=question["expected_answer"],
                candidate_answer=answer,
                evaluation=evaluation,
            )

            self.transcript.append(turn)
            self._print_evaluation(evaluation, question)

        self._print_summary()
        self._save_transcript()

    def _print_intro(self) -> None:
        names = ", ".join(a.name for a in self.agents)

        self.console.print(
            Panel.fit(
                f"Interview simulation: {self.config.project.name}\n"
                f"Candidate: {self.config.documents.candidate.name}\n"
                f"Company: {self.config.documents.company.name}\n"
                f"Interviewers: {names}\n\n"
                "Answer naturally. The system will evaluate each answer and then show the expected answer.",
                title="Senior AI Developer Interview Practice",
            )
        )

    def _print_evaluation(self, evaluation: dict, question: dict) -> None:
        table = Table(title="Evaluation")
        table.add_column("Metric")
        table.add_column("Value")

        table.add_row("Score", f"{evaluation['score']:.2f}")
        table.add_row("Verdict", evaluation["verdict"])
        table.add_row("Feedback", evaluation["feedback_to_candidate"])

        missing = "\n".join(f"- {x}" for x in evaluation.get("missing_points", [])) or "None"
        table.add_row("Missing points", missing)

        self.console.print(table)

        if self.config.runtime.print_expected_answer:
            self.console.print(
                Panel(
                    evaluation.get("corrected_or_expected_answer") or question["expected_answer"],
                    title="Expected / corrected answer",
                )
            )

        follow = evaluation.get("suggested_follow_up_question", "").strip()
        if (
            follow
            and self.config.runtime.allow_followups
            and evaluation["score"] < self.config.runtime.passing_score
        ):
            self.console.print(f"[dim]Potential follow-up to practice later: {follow}[/dim]")

    def _print_summary(self) -> None:
        if not self.transcript:
            self.console.print("No completed turns.")
            return

        scores = [float(t.evaluation.get("score", 0.0)) for t in self.transcript]
        avg = sum(scores) / len(scores)

        self.console.print(
            Panel.fit(
                f"Completed turns: {len(self.transcript)}\nAverage score: {avg:.2f}",
                title="Session summary",
            )
        )

    def _save_transcript(self) -> None:
        log_dir = Path(self.config.project.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        payload = [
            {
                "interviewer": t.interviewer,
                "question": t.question,
                "expected_answer": t.expected_answer,
                "candidate_answer": t.candidate_answer,
                "evaluation": t.evaluation,
            }
            for t in self.transcript
        ]

        path = log_dir / f"interview_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        self.console.print(f"[dim]Transcript saved locally to {path}[/dim]")
