"""Additive hybrid retrieval combining FAISS semantic search with BM25 keyword search."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from backend.app.services.vector_store import VectorStore


def build_text_for_embedding(section: Dict[str, Any]) -> str:
    """Create the embedding text for one statute section."""
    return f"{section['act']} Section {section['section_number']}: {section['section_title']}\n{section['text']}"


def tokenize(text: str) -> List[str]:
    """Tokenize text for BM25 using punctuation-insensitive word matching."""
    return re.findall(r"\w+", text.lower())


def build_bm25_index(sections: Sequence[Dict[str, Any]]) -> BM25Okapi:
    """Build a BM25Okapi index over the same text used for embedding."""
    corpus = [tokenize(build_text_for_embedding(section)) for section in sections]
    return BM25Okapi(corpus)


class HybridSearcher:
    """Combine semantic FAISS retrieval with BM25 keyword retrieval using RRF."""

    def __init__(
        self,
        vector_store: VectorStore,
        sections: Sequence[Dict[str, Any]],
        embedding_model: Optional[SentenceTransformer] = None,
        bm25_index: Optional[BM25Okapi] = None,
    ) -> None:
        self.vector_store = vector_store
        self.sections = list(sections)
        self.embedding_model = embedding_model or SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.bm25_index = bm25_index or build_bm25_index(self.sections)
        self.section_texts = [build_text_for_embedding(section) for section in self.sections]
        self.section_keys = [self._section_key(section) for section in self.sections]

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Return top_k results fused from FAISS and BM25 rankings."""
        query_vector = self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=False)
        faiss_results = self.vector_store.search(query_vector, top_k=10)
        bm25_results = self._bm25_search(query, top_k=10)

        rrf_scores: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for rank, result in enumerate(faiss_results, start=1):
            metadata = result.get("metadata", {})
            key = self._section_key(metadata)
            rrf_scores.setdefault(
                key,
                {
                    "metadata": metadata,
                    "faiss_rank": 1000,
                    "bm25_rank": 1000,
                    "faiss_score": None,
                    "bm25_score": None,
                    "combined_score": 0.0,
                },
            )
            rrf_scores[key]["faiss_rank"] = rank
            rrf_scores[key]["faiss_score"] = result.get("score")
            rrf_scores[key]["combined_score"] += 1.0 / (60 + rank)

        for rank, result in enumerate(bm25_results, start=1):
            metadata = result.get("metadata", {})
            key = self._section_key(metadata)
            rrf_scores.setdefault(
                key,
                {
                    "metadata": metadata,
                    "faiss_rank": 1000,
                    "bm25_rank": 1000,
                    "faiss_score": None,
                    "bm25_score": None,
                    "combined_score": 0.0,
                },
            )
            rrf_scores[key]["bm25_rank"] = rank
            rrf_scores[key]["bm25_score"] = result.get("score")
            rrf_scores[key]["combined_score"] += 1.0 / (60 + rank)

        ranked_results = sorted(
            rrf_scores.values(),
            key=lambda item: item["combined_score"],
            reverse=True,
        )

        return [
            {
                "metadata": item["metadata"],
                "score": item["combined_score"],
                "combined_score": item["combined_score"],
                "faiss_rank": item["faiss_rank"],
                "bm25_rank": item["bm25_rank"],
                "faiss_score": item["faiss_score"],
                "bm25_score": item["bm25_score"],
            }
            for item in ranked_results[:top_k]
        ]

    def _bm25_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []
        scores = self.bm25_index.get_scores(tokenized_query)
        ranked_positions = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
        return [
            {
                "metadata": self.sections[position],
                "score": float(scores[position]),
            }
            for position in ranked_positions
        ]

    @staticmethod
    def _section_key(section: Dict[str, Any]) -> Tuple[str, str]:
        return str(section.get("act", "")), str(section.get("section_number", ""))
