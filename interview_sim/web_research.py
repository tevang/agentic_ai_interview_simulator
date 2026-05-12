from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable

from tavily import TavilyClient

from .config import AppConfig
from .text_utils import compact, short_hash, unique_preserving_order
from .vector_store import MemoryStore

KNOWN_SKILL_TERMS = [
    "Python",
    "RAG",
    "Retrieval-Augmented Generation",
    "LLM",
    "large language models",
    "vector database",
    "prompt engineering",
    "evaluation framework",
    "MLOps",
    "model observability",
    "CI/CD",
    "automated testing",
    "REST API",
    "AWS",
    "cloud services",
    "Docker",
    "Kubernetes",
    "data pipelines",
    "deep learning",
    "responsible AI",
    "LangChain",
    "Node.js",
    "React",
    "Next.js",
    "GraphQL",
    "Azure DevOps",
    "PostgreSQL",
    "MS SQL",
    "UiPath",
    "BluePrism",
    "RPA",
]


@dataclass(frozen=True)
class ResearchResult:
    query: str
    text: str
    from_cache: bool


class ResearchGuard:
    def __init__(self, forbidden_names: Iterable[str], extra_terms: Iterable[str] = ()):
        phrases: list[str] = []
        tokens: list[str] = []

        for name in forbidden_names:
            clean = name.strip()
            if not clean:
                continue

            phrases.append(clean.lower())

            for token in re.findall(r"[A-Za-zÀ-ž0-9]+", clean):
                # Block name parts except very short ambiguous tokens such as Jan.
                if len(token) >= 4:
                    tokens.append(token.lower())

        for term in extra_terms:
            if term.strip():
                phrases.append(term.strip().lower())

        self.forbidden_phrases = set(phrases)
        self.forbidden_tokens = set(tokens)

    def assert_safe_query(self, query: str) -> None:
        q = query.lower()

        for phrase in self.forbidden_phrases:
            if phrase and phrase in q:
                raise ValueError(f"Blocked web query because it contains a forbidden name/phrase: {phrase!r}")

        query_tokens = set(re.findall(r"[a-z0-9]+", q))
        overlap = query_tokens.intersection(self.forbidden_tokens)

        if overlap:
            raise ValueError(f"Blocked web query because it contains forbidden name token(s): {sorted(overlap)}")


class TavilyResearcher:
    def __init__(self, config: AppConfig, memory: MemoryStore):
        self.config = config
        self.memory = memory

        api_key = os.environ.get("TAVILY_API_KEY")
        self.client = TavilyClient(api_key=api_key) if api_key else None

        forbidden = [*config.guardrails.candidate_names, *config.guardrails.interviewer_names]
        self.guard = ResearchGuard(forbidden, config.guardrails.forbidden_extra_terms)

    def available(self) -> bool:
        return bool(self.config.web_research.enabled and self.client)

    def search(self, query: str, *, owner_type: str, owner_name: str) -> ResearchResult:
        if self.config.guardrails.forbid_name_search:
            self.guard.assert_safe_query(query)

        source_id = f"web:{short_hash(query.lower())}"

        if self.memory.exists_source(source_id):
            return ResearchResult(
                query=query,
                text=self.memory.get_by_source(source_id),
                from_cache=True,
            )

        if not self.available():
            return ResearchResult(
                query=query,
                text="[web research skipped: Tavily disabled or TAVILY_API_KEY missing]",
                from_cache=False,
            )

        response = self.client.search(
            query=query,
            search_depth=self.config.web_research.search_depth,
            max_results=self.config.web_research.max_results_per_query,
            include_answer=self.config.web_research.include_answer,
            include_raw_content=self.config.web_research.include_raw_content,
        )

        text = self._format_search_response(query, response)

        self.memory.upsert_text(
            owner_type=owner_type,
            owner_name=owner_name,
            source_type="web_research",
            source_id=source_id,
            text=text,
            metadata={"query": query},
        )

        return ResearchResult(query=query, text=text, from_cache=False)

    def extract_url(self, url: str, *, owner_type: str, owner_name: str) -> ResearchResult:
        source_id = f"web_extract:{short_hash(url.lower())}"

        if self.memory.exists_source(source_id):
            return ResearchResult(
                query=url,
                text=self.memory.get_by_source(source_id),
                from_cache=True,
            )

        if not self.available():
            return ResearchResult(
                query=url,
                text="[web extraction skipped: Tavily disabled or TAVILY_API_KEY missing]",
                from_cache=False,
            )

        response = self.client.extract(url)
        text = self._format_extract_response(url, response)

        self.memory.upsert_text(
            owner_type=owner_type,
            owner_name=owner_name,
            source_type="company_website_extract",
            source_id=source_id,
            text=text,
            metadata={"url": url},
        )

        return ResearchResult(query=url, text=text, from_cache=False)

    @staticmethod
    def _format_search_response(query: str, response: dict) -> str:
        parts = [f"TAVILY SEARCH QUERY: {query}"]

        answer = response.get("answer")
        if answer:
            parts.append(f"SUMMARY ANSWER:\n{answer}")

        for i, item in enumerate(response.get("results", []), start=1):
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            content = item.get("raw_content") or item.get("content") or ""
            parts.append(
                f"RESULT {i}: {title}\n"
                f"URL: {url}\n"
                f"CONTENT: {compact(content, 1600)}"
            )

        return "\n\n".join(parts)

    @staticmethod
    def _format_extract_response(url: str, response: dict) -> str:
        parts = [f"TAVILY EXTRACT URL: {url}"]

        results = response.get("results") or []
        for i, item in enumerate(results, start=1):
            content = item.get("raw_content") or item.get("content") or ""
            parts.append(
                f"EXTRACT {i}: {item.get('url', url)}\n"
                f"{compact(content, 3000)}"
            )

        if len(parts) == 1:
            parts.append(compact(str(response), 3000))

        return "\n\n".join(parts)


def skill_queries_from_job_text(job_text: str, limit: int) -> list[str]:
    text = job_text.lower()
    found = [term for term in KNOWN_SKILL_TERMS if term.lower() in text]

    priority_queries = [
        "production RAG systems LLM evaluation vector databases retrieval strategies best practices",
        "LLMOps MLOps model observability monitoring production generative AI systems",
        "responsible AI security privacy governance enterprise generative AI healthcare",
        "Python AI ML pipelines automated testing CI/CD REST API cloud services best practices",
        "prompt optimization evaluation frameworks generative AI applications",
        "Docker Kubernetes orchestration for machine learning AI services production",
    ]

    term_queries = [
        f"{term} best practices production AI systems interview senior AI developer"
        for term in found
    ]

    return unique_preserving_order([*priority_queries, *term_queries])[:limit]


def skill_queries_from_interviewer_profile(profile_text: str, role_hint: str, limit: int) -> list[str]:
    text = f"{profile_text}\n{role_hint}".lower()
    found = [term for term in KNOWN_SKILL_TERMS if term.lower() in text]

    queries = [
        f"{term} senior AI developer interview topics enterprise software engineering"
        for term in found[:6]
    ]

    if not queries:
        queries = ["senior AI developer interview collaboration technical leadership enterprise AI"]

    return unique_preserving_order(queries)[:limit]