"""Grounded answer generation with numbered citations."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from .retrieval import Hit

SYSTEM_PROMPT = """You answer questions using ONLY the numbered sources provided.
Rules:
- Cite every factual sentence with the source numbers it relies on, like [1] or [2][3].
- If the sources do not contain the answer, reply exactly: "I couldn't find that in the uploaded documents."
- Do not use outside knowledge. Be concise: at most a few short paragraphs."""

_CITE = re.compile(r"\[(\d+)\]")
NOT_FOUND = "I couldn't find that in the uploaded documents."


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class AnthropicLLM:
    def __init__(self, api_key: str, model: str, max_tokens: int = 700) -> None:
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in msg.content if block.type == "text")


@dataclass
class Citation:
    n: int
    document_id: int
    filename: str
    page: int
    snippet: str


@dataclass
class Answer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    mode: str = "generative"


def build_prompt(question: str, hits: list[Hit]) -> str:
    sources = "\n\n".join(
        f"[{i}] ({h.filename}, page {h.page})\n{h.text}" for i, h in enumerate(hits, 1)
    )
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def extract_citations(text: str, hits: list[Hit]) -> list[Citation]:
    """Map [n] markers in the answer to sources, ignoring numbers that don't exist."""
    seen: list[int] = []
    for m in _CITE.finditer(text):
        n = int(m.group(1))
        if 1 <= n <= len(hits) and n not in seen:
            seen.append(n)
    return [_citation(n, hits[n - 1]) for n in seen]


def strip_invalid_markers(text: str, n_sources: int) -> str:
    return _CITE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_sources else "", text)


def _citation(n: int, h: Hit) -> Citation:
    snippet = h.text if len(h.text) <= 300 else h.text[:297].rsplit(" ", 1)[0] + "..."
    return Citation(n, h.document_id, h.filename, h.page, snippet)


def answer_question(question: str, hits: list[Hit], llm: LLM | None) -> Answer:
    if not hits:
        return Answer(NOT_FOUND, [], "generative" if llm else "extractive")
    if llm is None:
        return Answer(
            "No LLM API key is configured, so here are the most relevant passages.",
            [_citation(i, h) for i, h in enumerate(hits, 1)],
            mode="extractive",
        )
    text = llm.complete(SYSTEM_PROMPT, build_prompt(question, hits)).strip()
    text = strip_invalid_markers(text, len(hits))
    return Answer(text, extract_citations(text, hits))
