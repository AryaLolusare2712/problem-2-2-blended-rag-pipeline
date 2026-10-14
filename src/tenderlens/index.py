"""Persistence helpers for chunk corpus and retriever rebuild."""
from __future__ import annotations
import json
from pathlib import Path
from .chunking import load_documents, make_chunks
from .retrieval import HybridRetriever

def ingest(data_dir: str | Path, index_dir: str | Path, chunk_size: int = 180, overlap: int = 35) -> dict:
    docs = load_documents(data_dir)
    chunks = make_chunks(docs, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError(f"No readable .pdf, .txt, or .md documents found in {data_dir}")
    out = Path(index_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "chunks": chunks}
    (out / "chunks.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # Build now to validate model availability and corpus encoding. Store vectors as numpy.
    retriever = HybridRetriever(chunks)
    import numpy as np
    np.save(out / "embeddings.npy", np.asarray(retriever.embeddings, dtype=np.float32))
    (out / "index_meta.json").write_text(json.dumps({
        "schema_version": 1, "documents_loaded": len({c["source"] for c in chunks}),
        "chunks": len(chunks), "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
    }, indent=2), encoding="utf-8")
    return {"documents_loaded": len({c["source"] for c in chunks}), "chunks_created": len(chunks), "index_dir": str(out)}

def load_retriever(index_dir: str | Path) -> HybridRetriever:
    import numpy as np
    root = Path(index_dir)
    payload = json.loads((root / "chunks.json").read_text(encoding="utf-8"))
    retriever = HybridRetriever.__new__(HybridRetriever)
    from .retrieval import tokenize
    from rank_bm25 import BM25Okapi
    chunks = payload["chunks"]
    retriever.chunks = chunks
    retriever.by_id = {c["chunk_id"]: c for c in chunks}
    retriever.ids = [c["chunk_id"] for c in chunks]
    retriever.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    from sentence_transformers import SentenceTransformer
    retriever.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    retriever.embeddings = np.load(root / "embeddings.npy")
    return retriever
