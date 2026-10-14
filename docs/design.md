# System Design

## Use case
TenderLens retrieves evidence from tender notices, specifications, and amendments to help procurement staff answer clause-level questions.

## Retrieval flow
`Question -> BM25 candidates + dense embedding candidates -> weighted RRF -> cited evidence snippets`

BM25 is useful for exact terms (tender IDs, INR amounts, clause references). Dense retrieval is intended to find semantically related passages with different wording. RRF combines ranks without assuming raw BM25 and cosine scores are calibrated to the same scale.

## Data model
Each chunk stores `chunk_id`, `text`, `source`, `page`, and `chunk_index`. A chunk ID is deterministic for a source basename, page number, and chunk index; for multi-corpus production use, include a stable document hash to prevent basename collisions.

## Evaluation plan
Use a human-labelled golden set, grouped by tender/project. Report Recall@K, MRR, nDCG@K, query latency, and critical-clause miss rate. Compare BM25-only, dense-only, and hybrid configurations on the same split. The current script reports hybrid retrieval metrics only; the next experiment should add ablations and bootstrap confidence intervals.

## Risks and guardrails
- Conflicting addenda: display source and date; don't silently overwrite original text.
- Prompt injection in documents: retrieved text is evidence, never executable instructions.
- False confidence: no generated legal conclusion; present passages for human review.
- Data confidentiality: keep indexes and documents local, exclude them from Git.
- PDF extraction: scanned PDFs need OCR, which is outside this starter's scope.

## Scaling path
For larger corpora, replace the in-memory BM25 index with a persistent sparse index and store dense vectors in ChromaDB/Qdrant. Add incremental ingestion, content hashes, source access controls, amendment graphs, query tracing, and an API service. Benchmark ingestion throughput, p95 retrieval latency, and memory usage on a representative corpus before scaling claims.
