# DocQA

[![CI](https://github.com/zimroble/docqa-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/zimroble/docqa-rag/actions/workflows/ci.yml)

Upload PDFs, ask questions in plain English, and get answers that **cite the
exact page** they came from. Built with FastAPI and PostgreSQL + pgvector,
using hybrid (keyword + vector) retrieval and an LLM for grounded answers.

![DocQA screenshot](docs/screenshot.png)

## How it works

```
upload ──▶ pypdf text per page ──▶ 180-word chunks (40 overlap, never cross pages)
                                        │
                                        ├─▶ tsvector (Postgres full-text, GIN index)
                                        └─▶ 256-d embedding (pgvector, HNSW index)

question ──▶ full-text top-30  ┐
         └─▶ vector top-30     ┴─▶ Reciprocal Rank Fusion ─▶ top-5 chunks
                                                              │
                         numbered sources [1]..[5] + page ◀───┘
                                     │
                                     ▼
              LLM API: "answer only from sources, cite like [2]"
                                     │
              validated citations ─▶ {answer, citations: [{file, page, snippet}]}
```

- **Page-level citations.** Chunks never span a page break, so every chunk maps
  to exactly one page. Citation markers the model returns are checked against
  the sources it was given, and any number that doesn't exist is stripped.
- **Hybrid retrieval.** Keyword search catches exact terms (`wal_buffers`,
  `GIN`), and vector search catches paraphrases ("rarely used indexes"). The two
  rankings are fused with RRF (k = 60), so no score calibration is needed.
- **Runs without a GPU or an API key.** Embeddings come from
  [WordLlama](https://github.com/dleemiller/WordLlama), a 256-d static
  embedding model that ships inside its pip wheel and embeds the whole
  54-page eval corpus in about a second on CPU. Without an LLM key the app
  falls back to *extractive mode* and returns the cited passages.
- **Duplicate detection** by SHA-256, cascading deletes, upload size limits,
  and filtering to selected documents.

## Retrieval evaluation

[`eval/`](eval) holds a reproducible benchmark: 5 PDFs rendered from chapters
of the PostgreSQL 16 docs (54 pages, 219 chunks) and **60 hand-written
questions**. Each question has an *evidence sentence*, and the gold page is
whichever page contains it, which the harness checks automatically. Questions
are paraphrased rather than copied from the text, so the benchmark rewards
retrieval of meaning, not string matching.

| retriever | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
|---|---:|---:|---:|---:|
| keyword (Postgres FTS) | 56.7% | 75.0% | 90.0% | 0.679 |
| vector (pgvector) | 65.0% | 81.7% | 88.3% | 0.736 |
| **hybrid (RRF)** | 56.7% | **86.7%** | **91.7%** | 0.708 |

Hybrid retrieval latency: p50 4.2 ms, p95 5.8 ms (local Postgres, 60 queries).
Full report with the missed questions: [`eval/results.md`](eval/results.md).

Hybrid gives the best recall at 3 and 5, which matters most here because the
LLM sees the top 5. Vector-only search is best at rank 1. A cross-encoder
reranker or a stronger embedding model are the obvious next experiments, and
the harness makes it easy to measure them.

```bash
python eval/run_eval.py --out eval/results.md   # wipes the target DB first
```

## Quick start (Docker)

```bash
cp .env.example .env            # optionally add DOCQA_LLM_API_KEY
docker compose up --build
open http://localhost:8000
```

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Postgres 16 with pgvector, e.g.:
docker run -d -p 5432:5432 -e POSTGRES_USER=docqa -e POSTGRES_PASSWORD=docqa \
  -e POSTGRES_DB=docqa pgvector/pgvector:pg16

uvicorn docqa.main:app --reload
```

Tests (30) need a Postgres database to run against:

```bash
DOCQA_TEST_DATABASE_URL=postgresql://docqa:docqa@localhost:5432/docqa_test pytest
```

## API

| method | path | |
|---|---|---|
| `POST` | `/api/documents` | multipart upload (`file`): PDF, TXT or MD → `{document_id, pages, chunks, duplicate}` |
| `GET` | `/api/documents` | list documents |
| `DELETE` | `/api/documents/{id}` | delete a document and its chunks |
| `POST` | `/api/search` | `{question, k?, mode?: keyword\|vector\|hybrid, document_ids?}` → ranked chunks |
| `POST` | `/api/ask` | `{question, k?, document_ids?}` → `{answer, citations, mode, timing_ms}` |
| `GET` | `/api/health` | DB check and whether an LLM is configured |

Interactive docs are at `/docs` (OpenAPI).

## Configuration

All settings are environment variables with the `DOCQA_` prefix (see
[`docqa/config.py`](docqa/config.py)): `DATABASE_URL`, `LLM_API_KEY`,
`LLM_MODEL`, `CHUNK_WORDS`, `CHUNK_OVERLAP`, `TOP_K`, `CANDIDATES`, `RRF_K`,
`MAX_UPLOAD_MB`.

## Deploying

The image is a plain `uvicorn` container that listens on `$PORT`, so it runs on
any container host. It needs a Postgres database with the `vector`
extension available: Neon, Supabase, AWS RDS, Google Cloud SQL and Render
Postgres all support pgvector. See [DEPLOY.md](DEPLOY.md) for Google Cloud Run
and Render walkthroughs.

## Project layout

```
docqa/
  main.py        FastAPI app + routes
  chunking.py    PDF/text loading, cleaning, page-bounded chunking
  embeddings.py  Embedder protocol + WordLlama implementation
  db.py          schema (tsvector + vector columns, GIN + HNSW indexes)
  ingest.py      parse → chunk → embed → store (one transaction)
  retrieval.py   keyword / vector / hybrid (RRF) search
  answer.py      prompt building, LLM call, citation validation
  static/        single-page UI (vanilla JS, light/dark)
eval/            corpus, questions, harness, results
tests/           unit + API tests (pytest, real Postgres)
```

## License

MIT. The eval corpus is derived from the PostgreSQL documentation under the
PostgreSQL License (see `eval/corpus/LICENSE`).
