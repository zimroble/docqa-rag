"""Ingestion pipeline: parse -> chunk -> embed -> store."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from . import chunking
from .db import Database
from .embeddings import Embedder


@dataclass
class IngestResult:
    document_id: int
    filename: str
    pages: int
    chunks: int
    duplicate: bool = False


class EmptyDocumentError(ValueError):
    pass


def ingest(db: Database, embedder: Embedder, filename: str, data: bytes,
           chunk_words: int = 180, overlap: int = 40) -> IngestResult:
    digest = hashlib.sha256(data).hexdigest()
    with db.connection() as conn:
        existing = conn.execute(
            "SELECT id, filename, pages, chunks FROM documents WHERE sha256 = %s", [digest]
        ).fetchone()
    if existing:
        return IngestResult(existing["id"], existing["filename"], existing["pages"],
                            existing["chunks"], duplicate=True)

    pages = chunking.load(filename, data)
    chunks = chunking.chunk_pages(pages, chunk_words, overlap)
    if not chunks:
        raise EmptyDocumentError("no extractable text found (is this a scanned PDF?)")
    vectors = embedder.embed([c.text for c in chunks])

    with db.connection() as conn, conn.transaction():
        doc_id = conn.execute(
            "INSERT INTO documents (filename, sha256, pages, chunks) VALUES (%s, %s, %s, %s) RETURNING id",
            [filename, digest, len(pages), len(chunks)],
        ).fetchone()["id"]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (document_id, page, idx, text, embedding) VALUES (%s, %s, %s, %s, %s)",
                [(doc_id, c.page, c.index, c.text, np.asarray(v)) for c, v in zip(chunks, vectors, strict=True)],
            )
    return IngestResult(doc_id, filename, len(pages), len(chunks))
