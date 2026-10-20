"""FastAPI application."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from . import __version__
from .answer import LLM, AnthropicLLM, answer_question
from .config import Settings, get_settings
from .db import Database
from .embeddings import Embedder, get_embedder
from .ingest import EmptyDocumentError, ingest
from .retrieval import Mode, Retriever

STATIC = Path(__file__).parent / "static"


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)
    mode: Mode = "hybrid"
    document_ids: list[int] | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)
    document_ids: list[int] | None = None


def create_app(settings: Settings | None = None, embedder: Embedder | None = None,
               llm: LLM | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        emb = embedder or get_embedder()
        db = Database(settings.database_url, emb.dim)
        app.state.db = db
        app.state.embedder = emb
        app.state.retriever = Retriever(db, emb, settings.candidates, settings.rrf_k)
        app.state.llm = llm if llm is not None else (
            AnthropicLLM(settings.llm_api_key, settings.llm_model, settings.llm_max_tokens)
            if settings.llm_api_key else None
        )
        yield
        db.close()

    app = FastAPI(title="DocQA", version=__version__, lifespan=lifespan)

    @app.get("/api/health")
    def health(request: Request):
        with request.app.state.db.connection() as conn:
            conn.execute("SELECT 1")
        return {"status": "ok", "llm": request.app.state.llm is not None}

    @app.post("/api/documents", status_code=201)
    async def upload(request: Request, file: UploadFile):
        data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"file is larger than {settings.max_upload_mb} MB")
        if not data:
            raise HTTPException(400, "empty file")
        st = request.app.state
        try:
            result = await run_in_threadpool(
                ingest, st.db, st.embedder, file.filename or "upload", data,
                settings.chunk_words, settings.chunk_overlap,
            )
        except EmptyDocumentError as e:
            raise HTTPException(422, str(e)) from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:  # malformed PDFs raise a zoo of pypdf errors
            raise HTTPException(400, f"could not read file: {e}") from e
        return asdict(result)

    @app.get("/api/documents")
    def list_documents(request: Request):
        with request.app.state.db.connection() as conn:
            return conn.execute(
                "SELECT id, filename, pages, chunks, created_at FROM documents ORDER BY id"
            ).fetchall()

    @app.delete("/api/documents/{document_id}", status_code=204)
    def delete_document(request: Request, document_id: int):
        with request.app.state.db.connection() as conn:
            cur = conn.execute("DELETE FROM documents WHERE id = %s", [document_id])
            if cur.rowcount == 0:
                raise HTTPException(404, "document not found")

    @app.post("/api/search")
    def search(request: Request, body: SearchRequest):
        hits = request.app.state.retriever.search(body.question, body.k, body.mode, body.document_ids)
        return [asdict(h) for h in hits]

    @app.post("/api/ask")
    def ask(request: Request, body: AskRequest):
        t0 = time.perf_counter()
        st = request.app.state
        hits = st.retriever.search(body.question, body.k, "hybrid", body.document_ids)
        t1 = time.perf_counter()
        result = answer_question(body.question, hits, st.llm)
        t2 = time.perf_counter()
        return {
            **asdict(result),
            "timing_ms": {"retrieval": round((t1 - t0) * 1000, 1), "generation": round((t2 - t1) * 1000, 1)},
        }

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
