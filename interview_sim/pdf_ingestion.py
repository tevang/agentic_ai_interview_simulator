from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from .text_utils import file_hash


@dataclass(frozen=True)
class LoadedPdf:
    path: str
    text: str
    file_hash: str
    page_count: int


def extract_pdf_text(path: str | Path) -> LoadedPdf:
    pdf_path = Path(path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        pages.append(f"\n\n--- PAGE {i} ---\n{page_text.strip()}")

    text = "".join(pages).strip()
    return LoadedPdf(
        path=str(pdf_path),
        text=text,
        file_hash=file_hash(pdf_path),
        page_count=len(reader.pages),
    )


def load_many_pdfs(paths: list[str]) -> list[LoadedPdf]:
    return [extract_pdf_text(p) for p in paths]