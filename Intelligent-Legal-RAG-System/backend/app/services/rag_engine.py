"""Orchestrate vector retrieval and Gemini-grounded legal answer generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer

from backend.app.config import get_settings
from backend.app.services import llm_service
from backend.app.services.hybrid_search import HybridSearcher
from backend.app.services.vector_store import VectorStore


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
        ipc_sections_path = project_root / "backend/data/raw/ipc_full.json"
        crpc_sections_path = project_root / "backend/data/raw/code_of_criminal_procedure_1973.json"
        ipc_sections = json.loads(ipc_sections_path.read_text(encoding="utf-8"))
        crpc_sections = json.loads(crpc_sections_path.read_text(encoding="utf-8"))
        sections = ipc_sections + crpc_sections
        self.hybrid_searcher = HybridSearcher(
            vector_store=self.vector_store,
            sections=sections,
            embedding_model=self.embedding_model,
        )
        print("Hybrid search (FAISS + BM25) loaded successfully.")

    def answer(self, question: str) -> Dict[str, Any]:
        """Answer a legal question using the top retrieved statute sections."""
        retrieved = self.hybrid_searcher.search(question, top_k=5)
        prompt = build_grounded_prompt(question, retrieved)
        answer = llm_service.generate(prompt)

        if is_llm_error_response(answer):
            answer = build_fallback_answer(question, retrieved)

        sources = [
            {
                "section_number": result["metadata"]["section_number"],
                "section_title": result["metadata"]["section_title"],
            }
            for result in retrieved
        ]
        return {"answer": answer, "sources": sources}


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


def build_fallback_answer(question: str, retrieved_contexts: List[Dict[str, Any]]) -> str:
    """Provide a grounded fallback answer from retrieved sections when the LLM endpoint is unavailable."""
    if not retrieved_contexts:
        return "I cannot find this in the available sources."

    section_links = []
    for result in retrieved_contexts:
        metadata = result["metadata"]
        section_links.append(f"Section {metadata['section_number']} ({metadata['section_title']})")

    return (
        f"Based on the available legal sections, the most relevant sources are {', '.join(section_links)}. "
        f"Please review the retrieved sections above for the full statutory text."
    )


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


def answer_legal_query(query: str, jurisdiction: Optional[str] = None, top_k: int = 3) -> Dict[str, Any]:
    """Answer a legal query using the minimal RAG engine."""
    return RAGEngine().answer(query)

