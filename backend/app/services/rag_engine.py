"""Orchestrate vector retrieval and Gemini-grounded legal answer generation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer

from backend.app.config import get_settings
from backend.app.services import llm_service
from backend.app.services.hybrid_search import HybridSearcher, canonical_key
from backend.app.services.query_expansion import expand_query
from backend.app.services.raw_corpus import load_raw_sections
from backend.app.services.vector_store import VectorStore


ANSWER_TOP_K = 5
# How many chunks each retriever contributes before fusion. Measured over the
# 35-question evaluation set: widening this did not improve Hit@1 and cost
# several times the reranking latency.
RERANK_CANDIDATE_CHUNKS = 10
# Fusion collapses chunks back to sections, so this is only an upper bound -
# the fused list is what actually reaches the reranker (about 13 sections).
MAX_RERANK_CANDIDATES = 200


def check_index_matches_corpus(vector_store: VectorStore, sections: List[Dict[str, Any]]) -> None:
    """Fail loudly if the saved FAISS index and the current raw corpus disagree.

    Without this, a stale index (built from an older copy of the data - bug W2)
    loads silently and /health reports index_loaded: true even though some
    sections can never be found by semantic search. Comparing at the SECTION
    level (not per-chunk) keeps this check stable even if the chunker's output
    changes between the index build and this run.
    """
    index_keys = {
        canonical_key(metadata.get("act", ""), metadata.get("section_number", ""))
        for metadata in vector_store.metadatas.values()
    }
    raw_keys = {canonical_key(section.get("act", ""), section.get("section_number", "")) for section in sections}

    missing_from_index = raw_keys - index_keys
    stale_in_index = index_keys - raw_keys
    if missing_from_index or stale_in_index:
        raise RuntimeError(
            "FAISS index is out of sync with backend/data/raw: "
            f"{len(missing_from_index)} section(s) in the raw corpus are missing from the index, "
            f"{len(stale_in_index)} section(s) in the index no longer exist in the raw corpus. "
            "Rebuild it with `python backend/scripts/build_combined_index.py`."
        )


class RAGEngine:
    """Minimal retrieval-augmented generation engine for legal section answers."""

    def __init__(
        self,
        index_dir: Optional[Path] = None,
        embedding_model: Optional[SentenceTransformer] = None,
        vector_store: Optional[VectorStore] = None,
    ) -> None:
        """Load the embedding model, vector store, and hybrid searcher needed for answering questions."""
        settings = get_settings()
        project_root = Path(__file__).resolve().parents[3]
        self.embedding_model = embedding_model or SentenceTransformer(settings.embedding_model_name)
        self.vector_store = vector_store or VectorStore.load(index_dir or project_root / settings.faiss_index_dir)
        raw_dir = project_root / settings.raw_data_dir
        sections, loaded_files, skipped_files = load_raw_sections(raw_dir)
        print("Loaded raw section files:")
        for path in loaded_files:
            print(f"- {path.name}")
        for path in skipped_files:
            print(f"Skipped non-section file: {path.name}")
        print(f"Total combined section count: {len(sections)}")

        check_index_matches_corpus(self.vector_store, sections)

        # BM25 has to search the very same chunks the FAISS index was built over
        # (bug W3) - two retrievers ranking different-sized pieces of text cannot
        # be fused meaningfully. Rather than re-run the chunker and hope it agrees,
        # read the chunks back out of the index itself, where the build step
        # already stored them. That makes drift impossible and skips ~23 seconds
        # of tokenizer work on every startup.
        chunks = [self.vector_store.metadatas[position] for position in sorted(self.vector_store.metadatas)]
        print(f"Loaded {len(chunks)} chunks from the index for search.")

        self.hybrid_searcher = HybridSearcher(
            vector_store=self.vector_store,
            sections=chunks,
            embedding_model=self.embedding_model,
        )
        print("Hybrid search (FAISS + BM25) loaded successfully.")
        self.reranker = self._load_reranker(settings) if settings.enable_reranker else None

    @staticmethod
    def _load_reranker(settings) -> Optional[Any]:
        """Load the cross-encoder, or carry on without it.

        Reranking improves the ordering of results that retrieval already
        found, so losing it degrades answer quality but does not make the
        engine unable to answer. Failing startup over it would take the whole
        API down for what is an enhancement.
        """
        try:
            from backend.app.services.reranker import CrossEncoderReranker

            reranker = CrossEncoderReranker(settings.reranker_model_name or None)
            print("Cross-encoder reranker loaded successfully.")
            return reranker
        except Exception as exc:  # noqa: BLE001 - any load failure means "run without it"
            print(f"Reranker unavailable ({type(exc).__name__}: {exc}); continuing with RRF ranking only.")
            return None

    def answer(self, question: str) -> Dict[str, Any]:
        """Answer a legal question using the top retrieved statute sections."""
        # Only the SEARCH uses the expanded wording. The prompt keeps the
        # user's own question, so the answer addresses what they actually
        # asked rather than the statute phrasing bolted on to find it.
        search_query = expand_query(question)

        if self.reranker is None:
            retrieved = self.hybrid_searcher.search(search_query, top_k=ANSWER_TOP_K)
        else:
            # Hand the reranker the whole fused list, so it - not RRF - decides
            # which sections the answer is built from.
            candidates = self.hybrid_searcher.search(
                search_query, top_k=MAX_RERANK_CANDIDATES, candidate_k=RERANK_CANDIDATE_CHUNKS
            )
            retrieved = self.reranker.rerank(search_query, candidates, top_k=ANSWER_TOP_K)
        prompt = build_grounded_prompt(question, retrieved)
        answer = llm_service.generate(prompt)

        degraded = is_llm_error_response(answer)
        notice = ""
        if degraded:
            notice = describe_llm_failure(answer)
            answer = build_fallback_answer(question, retrieved)

        sources = [
            {
                "act": result["metadata"]["act"],
                "section_number": result["metadata"]["section_number"],
                "section_title": result["metadata"]["section_title"],
            }
            for result in retrieved
        ]
        return {"answer": answer, "sources": sources, "degraded": degraded, "notice": notice}


def is_llm_error_response(answer: Any) -> bool:
    """Return True when the LLM provider returned an error or quota-related response."""
    if answer is None:
        return True

    normalized = str(answer).lower()
    return (
        not normalized
        or normalized.startswith("gemini error:")
        or "quota" in normalized
        or "rate limit" in normalized
        or "retry_delay" in normalized
        or "exceeded" in normalized
    )


def describe_llm_failure(raw_answer: Any) -> str:
    """Say plainly why no generated answer is being shown.

    The user needs to tell "the law does not cover this" apart from "the
    answer service is down" - those call for completely different next steps,
    and a silent fallback looks like the first when it is really the second.
    """
    text = str(raw_answer or "").lower()
    if "quota" in text or "exceeded" in text or "429" in text:
        return "The Gemini API quota has been exceeded, so no written answer could be generated."
    if "rate limit" in text or "retry_delay" in text:
        return "Gemini is rate limiting requests right now, so no written answer could be generated."
    if not text:
        return "Gemini returned an empty response, so no written answer could be generated."
    return "Gemini could not be reached, so no written answer could be generated."


def build_fallback_answer(question: str, retrieved_contexts: List[Dict[str, Any]]) -> str:
    """Provide a grounded fallback answer from retrieved sections when the LLM endpoint is unavailable."""
    if not retrieved_contexts:
        return "No sections in the available Acts matched this question."

    lines = [
        "These are the most relevant sections found for your question. "
        "They are retrieved directly from the Acts and are not a written answer:",
        "",
    ]
    for result in retrieved_contexts:
        metadata = result["metadata"]
        lines.append(f"- {metadata['act']}, Section {metadata['section_number']}: {metadata['section_title']}")
    return "\n".join(lines)


def build_grounded_prompt(question: str, retrieved_contexts: List[Dict[str, Any]]) -> str:
    """Build a source-restricted legal prompt from retrieved statute sections."""
    source_blocks = []
    for result in retrieved_contexts:
        metadata = result["metadata"]
        source_blocks.append(
            "\n".join(
                [
                    f"Act: {metadata['act']}",
                    f"Section: {metadata['section_number']}",
                    f"Title: {metadata['section_title']}",
                    f"Text: {metadata['text']}",
                ]
            )
        )

    sources = "\n\n---\n\n".join(source_blocks) if source_blocks else "No sources retrieved."
    return f"""You are a legal retrieval assistant.
Answer the user's question using ONLY the facts stated in the provided legal sections below.
You MAY compare, contrast, and synthesize information across multiple provided sections to answer questions that ask about differences, relationships, or combinations between them — as long as every specific factual claim you make is still directly grounded in and cited to a specific section provided below.
Only say "I cannot find this in the available sources." if the provided sections, taken together, genuinely contain no information relevant to answering the question — not merely because no single section states the full answer on its own.
Cite the exact Act and Section number for every legal claim.
Do not rely on outside legal knowledge. Do not guess or invent any citation, section number, or fact not present in the provided sections.

Question:
{question}

Available legal sections:
{sources}

Answer:"""

