"""Build a combined FAISS index and BM25 data for all raw statute sections."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer

from backend.app.config import get_settings
from backend.app.services.hybrid_search import build_bm25_index
from backend.app.services.raw_corpus import load_raw_sections
from backend.app.services.vector_store import VectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_text_for_embedding(section: Dict[str, Any]) -> str:
    """Create the embedding text for one statute section."""
    return f"{section['act']} Section {section['section_number']}: {section['section_title']}\n{section['text']}"


def main() -> None:
    settings = get_settings()
    raw_dir = PROJECT_ROOT / settings.raw_data_dir
    index_dir = PROJECT_ROOT / settings.faiss_index_dir

    sections, loaded_files, skipped_files = load_raw_sections(raw_dir)

    print("Loaded raw section files:")
    for path in loaded_files:
        print(f"- {path.name}")

    for path in skipped_files:
        print(f"Skipped non-section file: {path.name}")

    print(f"Total combined section count: {len(sections)}")

    texts = [build_text_for_embedding(section) for section in sections]
    model = SentenceTransformer(settings.embedding_model_name)
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)

    build_bm25_index(sections)
    print(f"Built BM25 index over {len(sections)} combined sections.")

    store = VectorStore(dimension=int(vectors.shape[1]))
    store.add(vectors, sections)
    store.save(index_dir)

    print(f"Saved {len(sections)} combined sections to {index_dir}")


if __name__ == "__main__":
    main()
