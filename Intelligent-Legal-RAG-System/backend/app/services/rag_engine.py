"""Orchestrate vector retrieval and Gemini-grounded legal answer generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer

from backend.app.config import get_settings
from backend.app.services import llm_service
from backend.app.services.vector_store import VectorStore


class RAGEngine:
    """Minimal retrieval-augmented generation engine for legal section answers."""

    def __init__(
        self,
        index_dir: Optional[Path] = None,
        embedding_model: Optional[SentenceTransformer] = None,
        vector_store: Optional[VectorStore] = None,
    ) -> None:
        """Load the embedding model and vector store needed for answering questions."""
        settings = get_settings()
        project_root = Path(__file__).resolve().parents[3]
        self.embedding_model = embedding_model or SentenceTransformer(settings.embedding_model_name)
        self.vector_store = vector_store or VectorStore.load(index_dir or project_root / settings.faiss_index_dir)

    def answer(self, question: str) -> Dict[str, Any]:
        """Answer a legal question using the top retrieved statute sections."""
        query_vector = self.embedding_model.encode(question, convert_to_numpy=True, normalize_embeddings=False)
        retrieved = self.vector_store.search(query_vector, top_k=3)
        prompt = build_grounded_prompt(question, retrieved)
        answer = llm_service.generate(prompt)

        sources = [
            {
                "section_number": result["metadata"]["section_number"],
                "section_title": result["metadata"]["section_title"],
            }
            for result in retrieved
        ]
        return {"answer": answer, "sources": sources}


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

Answer the user's question ONLY using the provided legal sections.
Cite the exact Act and Section number for every legal claim.
If the provided sections do not actually answer the question, say exactly:
"I cannot find this in the available sources."

Do not rely on outside legal knowledge. Do not guess.

Question:
{question}

Available legal sections:
{sources}

Answer:"""


def answer_legal_query(query: str, jurisdiction: Optional[str] = None, top_k: int = 3) -> Dict[str, Any]:
    """Answer a legal query using the minimal RAG engine."""
    return RAGEngine().answer(query)

