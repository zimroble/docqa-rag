# Retrieval evaluation

Corpus: 5 PDFs, 54 pages, 219 chunks (indexed in 1.0s). Questions: 60.

| retriever | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
|---|---:|---:|---:|---:|
| keyword | 56.7% | 75.0% | 90.0% | 0.679 |
| vector | 65.0% | 81.7% | 88.3% | 0.736 |
| hybrid | 56.7% | 86.7% | 91.7% | 0.708 |

Hybrid retrieval latency over 60 queries: p50 4.2 ms, p95 5.8 ms.

Hybrid misses (gold page not in top 5):
- How do I label title words differently from body words so they rank higher?
- Why should indexes that are rarely used be dropped?
- Which index type can find the ten places closest to a point?
- How can I make a case-insensitive lookup on a column use an index?
- Does the autovacuum daemon ever run VACUUM FULL?
