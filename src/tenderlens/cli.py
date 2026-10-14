"""Command-line interface for TenderLens."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .index import ingest, load_retriever
from .evaluation import evaluate_queries

def main():
    parser = argparse.ArgumentParser(prog="tenderlens", description="Hybrid RAG for public procurement documents")
    sub = parser.add_subparsers(dest="command", required=True)
    p_ingest = sub.add_parser("ingest", help="Index PDF/TXT/Markdown documents")
    p_ingest.add_argument("--data", required=True)
    p_ingest.add_argument("--index-dir", default=".tenderlens_index")
    p_ingest.add_argument("--chunk-size", type=int, default=180)
    p_ingest.add_argument("--overlap", type=int, default=35)
    p_query = sub.add_parser("query", help="Retrieve cited evidence for a question")
    p_query.add_argument("query")
    p_query.add_argument("--index-dir", default=".tenderlens_index")
    p_query.add_argument("--top-k", type=int, default=5)
    p_eval = sub.add_parser("evaluate", help="Evaluate retrieval against labelled queries")
    p_eval.add_argument("--index-dir", default=".tenderlens_index")
    p_eval.add_argument("--dataset", required=True)
    p_eval.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.command == "ingest":
        print(json.dumps(ingest(args.data, args.index_dir, args.chunk_size, args.overlap), indent=2))
    elif args.command == "query":
        retriever = load_retriever(args.index_dir)
        hits = retriever.retrieve(args.query, top_k=args.top_k)
        if not hits:
            print("No evidence found.")
            return
        print("Evidence retrieved (review source versions before acting):\n")
        for i, hit in enumerate(hits, 1):
            page = f', page {hit["page"]}' if hit.get("page") else ""
            print(f'[{i}] {hit["source"]}{page} | chunk={hit["chunk_id"]} | RRF={hit["rrf_score"]}')
            print(hit["text"] + "\n")
    elif args.command == "evaluate":
        retriever = load_retriever(args.index_dir)
        dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
        print(json.dumps(evaluate_queries(retriever, dataset, args.top_k), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
