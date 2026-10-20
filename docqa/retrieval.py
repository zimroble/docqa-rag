"""Hybrid retrieval: Postgres full-text search + pgvector similarity, fused with
Reciprocal Rank Fusion (Cormack et al., 2009)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .db import Database
from .embeddings import Embedder

Mode = Literal["keyword", "vector", "hybrid"]

_TOKEN = re.compile(r"[A-Za-z0-9]+")


@dataclass
class Hit:
    chunk_id: int
    document_id: int
    filename: str
    page: int
    text: str
    score: float


def to_or_query(question: str) -> str:
    """Turn free text into an OR tsquery ("a | b | c"); stop words are dropped by Postgres.

    Using OR (instead of websearch_to_tsquery's AND) means a passage does not
    need to contain every word of a natural-language question to be found;
    ts_rank_cd still prefers passages that match more terms, closer together.
    """
    terms = _TOKEN.findall(question.lower())
    return " | ".join(dict.fromkeys(terms))


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank(d))."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class Retriever:
    def __init__(self, db: Database, embedder: Embedder, candidates: int = 30, rrf_k: int = 60):
        self.db = db
        self.embedder = embedder
        self.candidates = candidates
        self.rrf_k = rrf_k

    def _filter(self, document_ids: list[int] | None) -> tuple[str, list]:
        if not document_ids:
            return "", []
        return " AND c.document_id = ANY(%s)", [document_ids]

    def keyword_ids(self, question: str, limit: int, document_ids: list[int] | None = None) -> list[int]:
        query = to_or_query(question)
        if not query:
            return []
        where, params = self._filter(document_ids)
        with self.db.connection() as conn:
            rows = conn.execute(
                f"""SELECT c.id FROM chunks c, to_tsquery('english', %s) q
                    WHERE c.tsv @@ q {where}
                    ORDER BY ts_rank_cd(c.tsv, q, 32) DESC, c.id
                    LIMIT %s""",
                [query, *params, limit],
            ).fetchall()
        return [r["id"] for r in rows]

    def vector_ids(self, question: str, limit: int, document_ids: list[int] | None = None) -> list[int]:
        vec = self.embedder.embed([question])[0]
        where, params = self._filter(document_ids)
        with self.db.connection() as conn:
            # Exact search is fine at this scale; for millions of chunks the
            # HNSW index takes over automatically.
            rows = conn.execute(
                f"""SELECT c.id FROM chunks c WHERE TRUE {where}
                    ORDER BY c.embedding <=> %s, c.id LIMIT %s""",
                [*params, np.asarray(vec), limit],
            ).fetchall()
        return [r["id"] for r in rows]

    def search(self, question: str, k: int = 5, mode: Mode = "hybrid",
               document_ids: list[int] | None = None) -> list[Hit]:
        if mode == "keyword":
            ranked = [(i, 1.0 / (self.rrf_k + r)) for r, i in
                      enumerate(self.keyword_ids(question, k, document_ids), 1)]
        elif mode == "vector":
            ranked = [(i, 1.0 / (self.rrf_k + r)) for r, i in
                      enumerate(self.vector_ids(question, k, document_ids), 1)]
        else:
            ranked = rrf_fuse(
                [self.keyword_ids(question, self.candidates, document_ids),
                 self.vector_ids(question, self.candidates, document_ids)],
                k=self.rrf_k,
            )
        ranked = ranked[:k]
        return self._hydrate(ranked)

    def _hydrate(self, ranked: list[tuple[int, float]]) -> list[Hit]:
        if not ranked:
            return []
        ids = [i for i, _ in ranked]
        with self.db.connection() as conn:
            rows = conn.execute(
                """SELECT c.id, c.document_id, d.filename, c.page, c.text
                   FROM chunks c JOIN documents d ON d.id = c.document_id
                   WHERE c.id = ANY(%s)""",
                [ids],
            ).fetchall()
        by_id = {r["id"]: r for r in rows}
        return [
            Hit(i, by_id[i]["document_id"], by_id[i]["filename"], by_id[i]["page"], by_id[i]["text"], s)
            for i, s in ranked if i in by_id
        ]
