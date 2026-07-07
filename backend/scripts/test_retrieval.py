"""Run a minimal retrieval smoke test against the bundled FAISS index."""

from __future__ import annotations

import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings
from backend.app.services.vector_store import VectorStore


def main() -> None:
    """Embed a hardcoded legal question and print the top retrieved IPC sections."""
    settings = get_settings()
    question = "what is the punishment for murder"
    index_dir = PROJECT_ROOT / settings.faiss_index_dir

    model = SentenceTransformer(settings.embedding_model_name)
    query_vector = model.encode(question, convert_to_numpy=True, normalize_embeddings=False)

    store = VectorStore.load(index_dir)
    results = store.search(query_vector, top_k=3)

    print(f"Question: {question}")
    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(
            f"{rank}. IPC Section {metadata['section_number']} - {metadata['section_title']} "
            f"(score={result['score']:.4f})"
        )
        print(f"   {metadata['text'][:300]}...")


if __name__ == "__main__":
    main()

