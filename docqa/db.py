"""PostgreSQL + pgvector storage."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    filename    TEXT NOT NULL,
    sha256      TEXT NOT NULL UNIQUE,
    pages       INT  NOT NULL,
    chunks      INT  NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page        INT    NOT NULL,
    idx         INT    NOT NULL,
    text        TEXT   NOT NULL,
    tsv         TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    embedding   VECTOR({dim}) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id);
"""


class Database:
    def __init__(self, url: str, dim: int) -> None:
        self.dim = dim
        # Make sure the extension exists before any pooled connection
        # registers the vector type.
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.pool = ConnectionPool(
            url, min_size=1, max_size=10, open=True,
            kwargs={"row_factory": dict_row}, configure=register_vector,
        )
        with self.connection() as conn:
            conn.execute(SCHEMA.format(dim=dim))

    @contextmanager
    def connection(self) -> Iterator[psycopg.Connection]:
        with self.pool.connection() as conn:
            yield conn

    def close(self) -> None:
        self.pool.close()

    def reset(self) -> None:
        """Delete all documents (used by tests and the eval harness)."""
        with self.connection() as conn:
            conn.execute("TRUNCATE documents, chunks RESTART IDENTITY")
