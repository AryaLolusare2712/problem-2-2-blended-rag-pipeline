# TenderLens — Hybrid RAG for Public Procurement

TenderLens helps procurement teams find evidence across long tender notices, technical specifications, eligibility rules, addenda, and corrigenda.

## Why this problem matters
Tender information is difficult to retrieve reliably because:
- exact identifiers, dates, clause numbers, and monetary thresholds need keyword matching;
- equivalent concepts may use different wording and benefit from semantic retrieval;
- corrigenda can amend earlier requirements, so source metadata and citations matter;
- unsupported answers can create compliance risk.

TenderLens combines BM25 lexical search with dense sentence embeddings, merges rankings with Reciprocal Rank Fusion (RRF), and returns cited source passages. It is a retrieval-first reference implementation: it does not make legal decisions or silently resolve conflicting versions.

## Architecture
1. Ingest PDF, TXT, and Markdown files.
2. Extract text and preserve file/page metadata.
3. Split text into overlapping chunks.
4. Build a BM25 index and a sentence-transformer vector index.
5. Retrieve candidates from both indexes and fuse ranks using weighted RRF.
6. Return evidence snippets with source, page, chunk ID, and score.
7. Evaluate retrieval against a small, human-editable labelled query set.

Dense embeddings use `sentence-transformers/all-MiniLM-L6-v2`; the first run may download the model. No paid API key is required. An extractive answer is returned by default; an LLM answer generator can be added behind the `AnswerGenerator` interface without changing retrieval.

## Quick start (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip install -e .
python -m tenderlens.cli ingest --data data/sample_tenders --index-dir .tenderlens_index
python -m tenderlens.cli query "What is the bid security amount and deadline?" --index-dir .tenderlens_index --top-k 5
python -m tenderlens.cli evaluate --index-dir .tenderlens_index --dataset evaluation/golden_queries.json --top-k 5
```

## Add real documents
Put tender PDFs, corrigenda, TXT, or Markdown files in a directory, then run `ingest` on that directory. Only ingest documents you are permitted to process. Keep the original files and retain source/version metadata. Rebuild the index after changing the corpus.

## Example
```powershell
python -m tenderlens.cli query "What is the minimum annual turnover?" --index-dir .tenderlens_index --top-k 5
```

## Evaluation
The included golden query set is a small smoke-test fixture, not a benchmark of public procurement at scale. `evaluate` reports Recall@K, MRR, and nDCG@K using labelled relevant chunk IDs. For a meaningful evaluation:
- create a representative corpus of real tender documents and amendments;
- label relevant chunks independently, ideally with two reviewers;
- split by tender/project rather than randomly splitting chunks to prevent leakage;
- compare BM25-only, dense-only, and hybrid retrieval at the same K;
- report confidence intervals and latency, and track missed critical clauses.

## Security and guardrails
- Treat retrieved document text as untrusted data, never as instructions to the system.
- Answers should be based on retrieved evidence and include source citations.
- Do not infer that a clause is current merely because it appears in a document; verify amendment dates and version relationships.
- If evidence is missing or conflicting, return the evidence and flag the conflict instead of guessing.
- Avoid placing confidential tender material in public logs or repositories.
- This tool supports human review; it does not replace procurement/legal approval.

## Limitations
- Dense retrieval quality depends on the embedding model and corpus.
- PDF extraction quality varies, especially for scanned/image-only PDFs. OCR is not included in this starter.
- The sample data and evaluation set are illustrative. No production accuracy claim is made.
- Current-version resolution across amendments is surfaced as metadata but requires explicit amendment linking and review.

## Project structure
```text
src/tenderlens/
  chunking.py       text extraction and overlapping chunking
  index.py          persistent hybrid index
  retrieval.py      BM25 + dense retrieval + RRF
  evaluation.py     Recall@K, MRR, nDCG@K
  cli.py            ingest/query/evaluate commands
tests/
evaluation/
data/sample_tenders/
docs/design.md
```
