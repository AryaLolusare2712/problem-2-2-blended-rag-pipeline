"""Hybrid lexical + dense retrieval with weighted Reciprocal Rank Fusion."""
from __future__ import annotations
import re
from collections import defaultdict
from typing import Sequence

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+(?:[./_-][a-zA-Z0-9]+)*", text.lower())

def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], weights: Sequence[float] | None = None, c: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked ID lists. Higher weights increase a retriever's contribution."""
    if c <= 0:
        raise ValueError("c must be positive")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match number of rankings")
    scores = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] += float(weight) / (c + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))

class HybridRetriever:
    def __init__(self, chunks: list[dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not chunks:
            raise ValueError("Cannot build a retriever with zero chunks")
        self.chunks = chunks
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self.ids = [c["chunk_id"] for c in chunks]
        # Heavy optional dependencies are loaded only when a retriever is built.
        import numpy as np
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True)

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int = 20, weights: tuple[float, float] = (1.0, 1.0)) -> list[dict]:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if top_k < 1 or candidate_k < 1:
            raise ValueError("top_k and candidate_k must be positive")
        import numpy as np
        lexical_scores = self.bm25.get_scores(tokenize(query))
        lexical_order = np.argsort(-lexical_scores)[:candidate_k]
        qvec = self.model.encode([query], normalize_embeddings=True)[0]
        dense_scores = np.asarray(self.embeddings) @ qvec
        dense_order = np.argsort(-dense_scores)[:candidate_k]
        lexical_ids = [self.ids[int(i)] for i in lexical_order]
        dense_ids = [self.ids[int(i)] for i in dense_order]
        fused = reciprocal_rank_fusion([lexical_ids, dense_ids], weights=weights)
        lexical_map = {self.ids[int(i)]: float(lexical_scores[int(i)]) for i in lexical_order}
        dense_map = {self.ids[int(i)]: float(dense_scores[int(i)]) for i in dense_order}
        output = []
        for chunk_id, rrf_score in fused[:top_k]:
            item = dict(self.by_id[chunk_id])
            item["rrf_score"] = round(rrf_score, 6)
            item["bm25_score"] = round(lexical_map.get(chunk_id, 0.0), 6)
            item["dense_score"] = round(dense_map.get(chunk_id, 0.0), 6)
            output.append(item)
        return output
