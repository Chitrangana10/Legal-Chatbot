"""Reorder retrieved sections by reading each one against the question.

Retrieval and reranking answer different questions. FAISS turns a section into
one fixed vector before it has ever seen the query, so "unregistered vehicle"
lands near every section about registration - the requirement (s.39), where to
register (s.40), cancellation (s.55) - and the section that actually answers
"what happens if" (the penalty, s.192) has no way to stand out.

A cross-encoder reads the question and one section TOGETHER and scores that
pair, so it can tell a penalty from a requirement. It is far too slow to run
over the whole corpus, so it only reorders the handful of candidates retrieval
already found - which also means it can never rescue a section retrieval
missed entirely.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from backend.app.services.hybrid_search import build_text_for_embedding

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# The pair is truncated to the model's window, so the section title and the
# start of its text - the part that identifies it - must survive.
MAX_PAIR_LENGTH = 512


class CrossEncoderReranker:
    """Re-score (question, section) pairs with a cross-encoder model."""

    def __init__(self, model_name: Optional[str] = None, model: Optional[Any] = None) -> None:
        if model is not None:
            self.model = model
            return
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name or DEFAULT_RERANKER_MODEL, max_length=MAX_PAIR_LENGTH)

    def rerank(self, query: str, results: Sequence[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Return the best top_k results, reordered by relevance to `query`."""
        if not results:
            return []

        pairs = [(query, build_text_for_embedding(result["metadata"])) for result in results]
        scores = self.model.predict(pairs)

        reranked = []
        for result, score in zip(results, scores):
            enriched = dict(result)
            enriched["retrieval_score"] = result.get("score")
            enriched["rerank_score"] = float(score)
            enriched["score"] = float(score)
            reranked.append(enriched)

        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return reranked[:top_k]
