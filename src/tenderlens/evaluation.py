"""Retrieval metrics over a manually labelled golden query set."""
from __future__ import annotations
import math

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)

def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank, item in enumerate(retrieved[:k], start=1) if item in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0

def evaluate_queries(retriever, dataset: list[dict], k: int = 5) -> dict:
    rows = []
    for case in dataset:
        hits = retriever.retrieve(case["query"], top_k=k)
        ids = [h["chunk_id"] for h in hits]
        relevant = set(case["relevant_chunk_ids"])
        rows.append({
            "query_id": case.get("query_id", case["query"][:32]),
            "recall_at_k": recall_at_k(ids, relevant, k),
            "mrr": reciprocal_rank(ids, relevant),
            "ndcg_at_k": ndcg_at_k(ids, relevant, k),
            "retrieved_chunk_ids": ids,
        })
    n = len(rows) or 1
    return {
        "query_count": len(rows), "k": k,
        "mean_recall_at_k": round(sum(r["recall_at_k"] for r in rows) / n, 4),
        "mrr": round(sum(r["mrr"] for r in rows) / n, 4),
        "mean_ndcg_at_k": round(sum(r["ndcg_at_k"] for r in rows) / n, 4),
        "results": rows,
        "warning": "Metrics are only meaningful when relevant_chunk_ids match the indexed corpus and are human-labelled."
    }
