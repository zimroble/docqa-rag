"""Document loading and chunking.

Chunks never span page boundaries, so every chunk maps to exactly one page and
answers can cite the page they came from.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

from pypdf import PdfReader

_WS = re.compile(r"\s+")
# Hyphenated line breaks from PDF text extraction: "concur-\nrency" -> "concurrency"
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


@dataclass(frozen=True)
class Page:
    number: int  # 1-based
    text: str


@dataclass(frozen=True)
class Chunk:
    page: int
    index: int  # position within the document
    text: str


def clean(text: str) -> str:
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    return _WS.sub(" ", text).strip()


def read_pdf(data: bytes) -> list[Page]:
    reader = PdfReader(io.BytesIO(data))
    return [Page(i + 1, clean(p.extract_text() or "")) for i, p in enumerate(reader.pages)]


def read_text(data: bytes) -> list[Page]:
    """Plain text / Markdown: form-feed characters act as page breaks."""
    raw = data.decode("utf-8", errors="replace")
    return [Page(i + 1, clean(part)) for i, part in enumerate(raw.split("\f"))]


def load(filename: str, data: bytes) -> list[Page]:
    name = filename.lower()
    if name.endswith(".pdf") or data[:5] == b"%PDF-":
        return read_pdf(data)
    if name.endswith((".txt", ".md", ".markdown")):
        return read_text(data)
    raise ValueError("unsupported file type: upload a .pdf, .txt or .md file")


def chunk_pages(pages: list[Page], size: int = 180, overlap: int = 40) -> list[Chunk]:
    """Split each page into windows of `size` words with `overlap` words of overlap."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    step = size - overlap
    chunks: list[Chunk] = []
    for page in pages:
        words = page.text.split()
        if not words:
            continue
        start = 0
        while True:
            window = words[start:start + size]
            chunks.append(Chunk(page.number, len(chunks), " ".join(window)))
            if start + size >= len(words):
                break
            start += step
    return chunks
