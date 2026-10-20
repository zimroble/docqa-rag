#!/usr/bin/env python3
"""Retrieval evaluation.

Indexes the PDFs in eval/corpus/ into a scratch database, runs every question
in eval/questions.jsonl through keyword, vector and hybrid retrieval, and
reports Hit@k and MRR at the page level. A question counts as a hit at rank r
if the r-th retrieved chunk comes from the gold document and page, meaning the
page that contains the question's evidence sentence.

    DOCQA_DATABASE_URL=postgresql://... python eval/run_eval.py [--out eval/results.md]

WARNING: this wipes the documents in the target database.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from docqa import chunking  # noqa: E402
from docqa.config import get_settings  # noqa: E402
from docqa.db import Database  # noqa: E402
from docqa.embeddings import get_embedder  # noqa: E402
from docqa.ingest import ingest  # noqa: E402
from docqa.retrieval import Retriever  # noqa: E402

EVAL = ROOT / "eval"
KS = (1, 3, 5)


def gold_pages(questions):
    cache = {}
    for q in questions:
        if q["doc"] not in cache:
            cache[q["doc"]] = chunking.load(q["doc"], (EVAL / "corpus" / q["doc"]).read_bytes())
        pages = [p.number for p in cache[q["doc"]] if q["evidence"] in p.text]
        if len(pages) != 1:
            raise SystemExit(f"evidence must appear on exactly one page: {q['question']!r} -> {pages}")
        q["page"] = pages[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="also write a Markdown report here")
    args = ap.parse_args()

    settings = get_settings()
    questions = [json.loads(line) for line in (EVAL / "questions.jsonl").read_text().splitlines() if line.strip()]
    gold_pages(questions)

    embedder = get_embedder()
    db = Database(settings.database_url, embedder.dim)
    db.reset()
    t0 = time.perf_counter()
    n_chunks = n_pages = 0
    for pdf in sorted((EVAL / "corpus").glob("*.pdf")):
        r = ingest(db, embedder, pdf.name, pdf.read_bytes(), settings.chunk_words, settings.chunk_overlap)
        n_chunks += r.chunks
        n_pages += r.pages
    ingest_s = time.perf_counter() - t0
    retriever = Retriever(db, embedder, settings.candidates, settings.rrf_k)

    rows = []
    misses = []
    latencies = []
    for mode in ("keyword", "vector", "hybrid"):
        hits = {k: 0 for k in KS}
        rr = 0.0
        for q in questions:
            t = time.perf_counter()
            results = retriever.search(q["question"], k=max(KS), mode=mode)
            if mode == "hybrid":
                latencies.append((time.perf_counter() - t) * 1000)
            rank = next((i for i, h in enumerate(results, 1)
                         if h.filename == q["doc"] and h.page == q["page"]), None)
            if rank:
                rr += 1 / rank
                for k in KS:
                    hits[k] += rank <= k
            elif mode == "hybrid":
                misses.append(q["question"])
        n = len(questions)
        rows.append((mode, *(hits[k] / n for k in KS), rr / n))

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(0.95 * (len(latencies) - 1))]

    lines = [
        f"Corpus: {len(list((EVAL / 'corpus').glob('*.pdf')))} PDFs, {n_pages} pages, {n_chunks} chunks "
        f"(indexed in {ingest_s:.1f}s). Questions: {len(questions)}.",
        "",
        "| retriever | Hit@1 | Hit@3 | Hit@5 | MRR@5 |",
        "|---|---:|---:|---:|---:|",
        *[f"| {m} | {h1:.1%} | {h3:.1%} | {h5:.1%} | {mrr:.3f} |" for m, h1, h3, h5, mrr in rows],
        "",
        f"Hybrid retrieval latency over {len(latencies)} queries: p50 {p50:.1f} ms, p95 {p95:.1f} ms.",
    ]
    if misses:
        lines += ["", "Hybrid misses (gold page not in top 5):", *[f"- {m}" for m in misses]]
    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text("# Retrieval evaluation\n\n" + report + "\n")
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
