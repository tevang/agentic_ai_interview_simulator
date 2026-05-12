#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from interview_sim.bootstrap import Bootstrapper
from interview_sim.config import load_config
from interview_sim.orchestrator import InterviewOrchestrator
from interview_sim.vector_store import MemoryStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agentic interview simulator for Senior AI Developer practice.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--force-reingest", action="store_true", help="Re-ingest PDFs and refresh cached web research")
    parser.add_argument("--bootstrap-only", action="store_true", help="Ingest sources and exit without starting the interview")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    console = Console()
    config = load_config(args.config)

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required. Put it in .env or export it in the shell.")
    if config.web_research.enabled and not os.environ.get("TAVILY_API_KEY"):
        console.print("[yellow]TAVILY_API_KEY is missing; web research will be skipped, but PDF-based simulation will still run.[/yellow]")

    Path(config.project.storage_dir).mkdir(parents=True, exist_ok=True)
    memory = MemoryStore(config.project.storage_dir, config.models.embedding_model)

    Bootstrapper(config, memory, console).run(force_reingest=args.force_reingest)
    if args.bootstrap_only:
        return

    InterviewOrchestrator(config, memory, console).run()


if __name__ == "__main__":
    main()