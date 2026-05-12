from __future__ import annotations

from rich.console import Console

from .config import AppConfig
from .pdf_ingestion import extract_pdf_text, load_many_pdfs
from .vector_store import MemoryStore
from .web_research import (
    TavilyResearcher,
    skill_queries_from_interviewer_profile,
    skill_queries_from_job_text,
)


class Bootstrapper:
    def __init__(self, config: AppConfig, memory: MemoryStore, console: Console | None = None):
        self.config = config
        self.memory = memory
        self.console = console or Console()
        self.researcher = TavilyResearcher(config, memory)

    def run(self, *, force_reingest: bool = False) -> None:
        self._ingest_job(force_reingest=force_reingest)
        self._ingest_interview_character(force_reingest=force_reingest)
        self._ingest_candidate(force_reingest=force_reingest)
        self._ingest_company_research()
        self._ingest_interviewers(force_reingest=force_reingest)

    def _ingest_job(self, *, force_reingest: bool) -> None:
        loaded = extract_pdf_text(self.config.documents.job_description_pdf)
        source_id = f"job_pdf:{loaded.file_hash}"

        if force_reingest or not self.memory.exists_source(source_id):
            chunks = self.memory.upsert_text(
                owner_type="job",
                owner_name="Senior AI Developer",
                source_type="job_description",
                source_id=source_id,
                text=loaded.text,
                metadata={"path": loaded.path, "pages": loaded.page_count},
            )
            self.console.print(f"[green]Ingested job description:[/green] {chunks} chunks")
        else:
            self.console.print("[cyan]Job description already cached.[/cyan]")

        for query in skill_queries_from_job_text(
            loaded.text,
            self.config.web_research.max_job_skill_queries,
        ):
            try:
                result = self.researcher.search(
                    query,
                    owner_type="job",
                    owner_name="Senior AI Developer",
                )
                tag = "cached" if result.from_cache else "fetched"
                self.console.print(f"[dim]Job skill research {tag}: {query}[/dim]")
            except ValueError as exc:
                self.console.print(f"[yellow]Skipped unsafe job research query:[/yellow] {exc}")

    def _ingest_interview_character(self, *, force_reingest: bool) -> None:
        path = self.config.documents.interview_character_pdf
        if not path:
            return

        loaded = extract_pdf_text(path)
        source_id = f"interview_character_pdf:{loaded.file_hash}"

        if force_reingest or not self.memory.exists_source(source_id):
            chunks = self.memory.upsert_text(
                owner_type="interview",
                owner_name="Interview Character",
                source_type="interview_character",
                source_id=source_id,
                text=loaded.text,
                metadata={"path": loaded.path, "pages": loaded.page_count},
            )
            self.console.print(f"[green]Ingested interview character:[/green] {chunks} chunks")
        else:
            self.console.print("[cyan]Interview character already cached.[/cyan]")

    def _ingest_candidate(self, *, force_reingest: bool) -> None:
        loaded = extract_pdf_text(self.config.documents.candidate.cv_pdf)
        source_id = f"candidate_cv_pdf:{loaded.file_hash}"

        if force_reingest or not self.memory.exists_source(source_id):
            chunks = self.memory.upsert_text(
                owner_type="candidate",
                owner_name=self.config.documents.candidate.name,
                source_type="candidate_cv",
                source_id=source_id,
                text=loaded.text,
                metadata={"path": loaded.path, "pages": loaded.page_count},
            )
            self.console.print(f"[green]Ingested candidate CV:[/green] {chunks} chunks")
        else:
            self.console.print("[cyan]Candidate CV already cached.[/cyan]")

    def _ingest_company_research(self) -> None:
        company = self.config.documents.company

        try:
            result = self.researcher.extract_url(
                company.website,
                owner_type="company",
                owner_name=company.name,
            )
            tag = "cached" if result.from_cache else "fetched"
            self.console.print(f"[dim]Company website extraction {tag}: {company.website}[/dim]")
        except Exception as exc:
            self.console.print(f"[yellow]Company website extraction skipped:[/yellow] {exc}")

        for query in self.config.web_research.company_queries:
            try:
                result = self.researcher.search(
                    query,
                    owner_type="company",
                    owner_name=company.name,
                )
                tag = "cached" if result.from_cache else "fetched"
                self.console.print(f"[dim]Company research {tag}: {query}[/dim]")
            except ValueError as exc:
                self.console.print(f"[yellow]Skipped unsafe company research query:[/yellow] {exc}")

    def _ingest_interviewers(self, *, force_reingest: bool) -> None:
        for interviewer in self.config.documents.interviewers:
            already_registered = self.memory.has_interviewer_profile(interviewer.name)

            if already_registered and not force_reingest:
                self.console.print(
                    f"[cyan]Interviewer profile already cached; skipping re-ingest and web research:[/cyan] "
                    f"{interviewer.name}"
                )
                continue

            pdfs = load_many_pdfs([*interviewer.profile_pdfs, *interviewer.extra_pdfs])
            combined = "\n\n".join(p.text for p in pdfs)

            for loaded in pdfs:
                source_id = f"interviewer_pdf:{interviewer.name}:{loaded.file_hash}"

                self.memory.upsert_text(
                    owner_type="interviewer",
                    owner_name=interviewer.name,
                    source_type="interviewer_profile",
                    source_id=source_id,
                    text=loaded.text,
                    metadata={
                        "path": loaded.path,
                        "pages": loaded.page_count,
                        "role_hint": interviewer.role_hint,
                        "style_hint": interviewer.style_hint,
                    },
                )

            self.console.print(f"[green]Ingested interviewer profile:[/green] {interviewer.name}")

            for query in skill_queries_from_interviewer_profile(
                combined,
                interviewer.role_hint,
                self.config.web_research.max_interviewer_skill_queries,
            ):
                try:
                    result = self.researcher.search(
                        query,
                        owner_type="interviewer_research",
                        owner_name=interviewer.name,
                    )
                    tag = "cached" if result.from_cache else "fetched"
                    self.console.print(f"[dim]Interviewer background research {tag}: {query}[/dim]")
                except ValueError as exc:
                    self.console.print(f"[yellow]Skipped unsafe interviewer research query:[/yellow] {exc}")