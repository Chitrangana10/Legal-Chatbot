"""Measure how well retrieval actually ranks the right section.

"Nothing is broken" and "the answers are good" are different claims. The
validator proves the first; this proves the second, and gives a number to
compare future changes against instead of judging by eyeball.

Reported per configuration:
  Hit@1 / Hit@3 / Hit@5  how often an expected section lands in the top N
  MRR                    1/rank of the first correct hit, averaged - rewards
                         ranking the right section higher, not just present

By default it scores three configurations side by side - FAISS alone, BM25
alone, and the two fused with RRF - so the value of hybrid search is measured
rather than assumed.

Usage:
    python backend/scripts/evaluate_retrieval.py
    python backend/scripts/evaluate_retrieval.py --show-failures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer

from backend.app.config import get_settings
from backend.app.services.hybrid_search import HybridSearcher, canonical_key
from backend.app.services.query_expansion import expand_query
from backend.app.services.vector_store import VectorStore

QUESTIONS_PATH = PROJECT_ROOT / "backend/data/eval/retrieval_questions.json"
TOP_K = 5
# Measured over the 35-question set: a deeper pool did not improve Hit@1 and
# cost several times the latency, so reranking stays on the default depth.
RERANK_CANDIDATE_CHUNKS = 10


def _ranked_keys(results: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Section keys in rank order, keeping only the first appearance of each."""
    keys: List[Tuple[str, str]] = []
    for result in results:
        metadata = result.get("metadata", result)
        key = canonical_key(metadata.get("act", ""), metadata.get("section_number", ""))
        if key not in keys:
            keys.append(key)
    return keys


def _first_hit_rank(ranked: Sequence[Tuple[str, str]], expected: Sequence[Tuple[str, str]]) -> int:
    for position, key in enumerate(ranked, start=1):
        if key in expected:
            return position
    return 0


def score(runner, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    hits = {1: 0, 3: 0, 5: 0}
    reciprocal_total = 0.0
    failures: List[Dict[str, Any]] = []

    for item in questions:
        expected = [canonical_key(e["act"], e["section_number"]) for e in item["expected"]]
        ranked = _ranked_keys(runner(item["question"]))
        rank = _first_hit_rank(ranked, expected)

        if rank:
            reciprocal_total += 1.0 / rank
            for cutoff in hits:
                if rank <= cutoff:
                    hits[cutoff] += 1
        if not rank or rank > 3:
            failures.append({"question": item["question"], "rank": rank, "expected": expected, "got": ranked[:5]})

    total = len(questions)
    return {
        "hit@1": hits[1] / total,
        "hit@3": hits[3] / total,
        "hit@5": hits[5] / total,
        "mrr": reciprocal_total / total,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show-failures", action="store_true", help="list questions whose answer missed the top 3")
    parser.add_argument("--rerank", action="store_true", help="also score hybrid + cross-encoder reranking")
    args = parser.parse_args()

    settings = get_settings()
    store = VectorStore.load(PROJECT_ROOT / settings.faiss_index_dir)
    chunks = [store.metadatas[position] for position in sorted(store.metadatas)]
    model = SentenceTransformer(settings.embedding_model_name)
    searcher = HybridSearcher(vector_store=store, sections=chunks, embedding_model=model)

    payload = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = payload["questions"]

    def faiss_only(question: str):
        vector = model.encode([question], convert_to_numpy=True, normalize_embeddings=False)
        return store.search(vector, top_k=20)

    configurations = {
        "FAISS only (meaning)": faiss_only,
        "BM25 only (keywords)": lambda q: searcher._bm25_search(q, top_k=20),
        "Hybrid RRF (current)": lambda q: searcher.search(q, top_k=TOP_K),
    }

    if args.rerank:
        from backend.app.services.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(settings.reranker_model_name or None)

        def hybrid_reranked(question: str):
            # Mirrors RAGEngine.answer(): expand the wording for search only,
            # then let the cross-encoder - not RRF - decide the final order.
            search_query = expand_query(question)
            pool = searcher.search(search_query, top_k=200, candidate_k=RERANK_CANDIDATE_CHUNKS)
            return reranker.rerank(search_query, pool, top_k=TOP_K)

        configurations["Hybrid + cross-encoder"] = hybrid_reranked

    print(f"Retrieval evaluation over {len(questions)} questions ({QUESTIONS_PATH.relative_to(PROJECT_ROOT)})\n")
    print(f"{'configuration':24s} {'Hit@1':>7s} {'Hit@3':>7s} {'Hit@5':>7s} {'MRR':>7s}")
    print("-" * 58)

    results = {}
    for name, runner in configurations.items():
        outcome = score(runner, questions)
        results[name] = outcome
        print(
            f"{name:24s} {outcome['hit@1']:6.0%} {outcome['hit@3']:7.0%} "
            f"{outcome['hit@5']:7.0%} {outcome['mrr']:7.3f}"
        )

    # Report failures for whatever the app actually runs, so the list reflects
    # production rather than a configuration nobody serves.
    production = "Hybrid + cross-encoder" if args.rerank else "Hybrid RRF (current)"
    best = results[production]
    print(f"\n{production} missed the top 3 on {len(best['failures'])} of {len(questions)} questions.")
    if args.show_failures:
        for failure in best["failures"]:
            position = failure["rank"] or "not in top 5"
            print(f"\n  Q: {failure['question']}")
            print(f"     expected {failure['expected']}  ->  rank {position}")
            for got in failure["got"]:
                print(f"       got: {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
