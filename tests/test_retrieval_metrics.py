from tenderlens.retrieval import reciprocal_rank_fusion
from tenderlens.evaluation import recall_at_k, reciprocal_rank, ndcg_at_k

def test_rrf_prefers_item_ranked_by_both_retrievers():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    assert fused[0][0] in {"a", "b"}
    assert dict(fused)["a"] == dict(fused)["b"]

def test_rrf_weights_and_validation():
    fused = reciprocal_rank_fusion([["a"], ["b"]], weights=[2, 1])
    assert fused[0][0] == "a"

def test_retrieval_metrics():
    retrieved = ["x", "a", "b"]
    relevant = {"a", "b"}
    assert recall_at_k(retrieved, relevant, 3) == 1.0
    assert reciprocal_rank(retrieved, relevant) == 0.5
    assert 0 < ndcg_at_k(retrieved, relevant, 3) <= 1
