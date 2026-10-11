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


# Words dropped before BM25 sees them. Without this, Constitution article 69
# ("Oath of the Vice-President") was the TOP keyword hit for "What are my
# rights if I am arrested?" - it matches no meaningful word in that question,
# but the oath repeats "I" four times, and being a short chunk, BM25's length
# normalisation then boosted it further. rank_bm25 also floors what would be a
# negative idf at a small positive value, so a word in almost every document
# still adds score instead of being ignored.
#
# Deliberately limited to grammar words. Anything that changes what a section
# MEANS stays in: "not", "no", "shall", "may", "must", "under", "against",
# "without", "before", "after", "within" are all load-bearing in statute text.
STOPWORDS = frozenset(
    """
    a an the and or but if then than so as of to in into for on at by with from
    that this these those there here it its
    i me my myself we us our ours you your yours he him his she her hers they them their theirs
    what which who whom whose when where why how
    is am are was were be been being do does did doing have has had having
    """.split()
)


def tokenize(text: str) -> List[str]:
    """Tokenize text for BM25 using punctuation-insensitive word matching."""
    return [word for word in re.findall(r"\w+", text.lower()) if word not in STOPWORDS]


def canonical_key(act: Any, section_number: Any) -> Tuple[str, str]:
    """Normalize an (act, section_number) pair into a stable fusion identity.

    The FAISS index and the raw corpus disagree on surface form: the index drops
    the leading "The" from some act names ("Indian Penal Code, 1860" vs
    "The Indian Penal Code, 1860") and keeps a trailing period on CrPC section
    numbers ("247." vs "247"). Left unnormalized, those sections never fuse
    across the two retrievers. Stripping a leading article, surrounding
    punctuation and case makes the key independent of which source produced it.
    """
    act_norm = re.sub(r"^the\s+", "", str(act).strip(), flags=re.IGNORECASE)
    act_norm = re.sub(r"\s+", " ", act_norm).strip().lower()
    section_norm = str(section_number).strip().strip(".").strip().lower()
    return act_norm, section_norm


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

    def search(self, query: str, top_k: int = 3, candidate_k: int = 10) -> List[Dict[str, Any]]:
        """Return top_k results fused from FAISS and BM25 rankings.

        `candidate_k` is how many CHUNKS each engine contributes before fusion.
        Several chunks can belong to one section, so the fused list is always
        shorter than 2 * candidate_k. A reranker needs this raised well above
        top_k, since it can only reorder what retrieval already found.
        """
        query_vector = self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=False)
        faiss_results = self.vector_store.search(query_vector, top_k=candidate_k)
        bm25_results = self._bm25_search(query, top_k=candidate_k)

        rrf_scores: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for rank, result in enumerate(faiss_results, start=1):
            self._apply_rank(rrf_scores, result, rank, source="faiss")
        for rank, result in enumerate(bm25_results, start=1):
            self._apply_rank(rrf_scores, result, rank, source="bm25")

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

    def _apply_rank(
        self,
        rrf_scores: Dict[Tuple[str, str], Dict[str, Any]],
        result: Dict[str, Any],
        rank: int,
        source: str,
    ) -> None:
        """Fold one engine's ranked hit into the running per-section RRF score.

        A section may appear more than once here if it was split into chunks
        (see chunking.py) and more than one chunk matched. When that happens,
        keep the metadata from whichever chunk ranked best overall - not just
        the first one seen - so the citation shown to the user is the most
        relevant excerpt, not an arbitrary one.
        """
        metadata = result.get("metadata", {})
        key = self._section_key(metadata)
        entry = rrf_scores.setdefault(
            key,
            {
                "metadata": metadata,
                "best_rank": rank,
                "faiss_rank": 1000,
                "bm25_rank": 1000,
                "faiss_score": None,
                "bm25_score": None,
                "combined_score": 0.0,
            },
        )
        entry[f"{source}_rank"] = rank
        entry[f"{source}_score"] = result.get("score")
        entry["combined_score"] += 1.0 / (60 + rank)
        if rank < entry["best_rank"]:
            entry["best_rank"] = rank
            entry["metadata"] = metadata

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
        return canonical_key(section.get("act", ""), section.get("section_number", ""))
