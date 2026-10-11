"""Build a combined FAISS index and BM25 data for all raw statute sections.

Every section is split into embed-sized chunks first (see
backend/app/services/chunking.py, bug W3) so a long section's later
sub-clauses are never silently truncated by the embedding model's 256-token
limit. FAISS then holds one vector per CHUNK, not per section - a section that
didn't need splitting still comes through as exactly one chunk, so short
sections are unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer

from backend.app.config import get_settings
from backend.app.services.chunking import chunk_sections
from backend.app.services.hybrid_search import build_text_for_embedding
from backend.app.services.raw_corpus import load_raw_sections
from backend.app.services.vector_store import VectorStore


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

    chunks = chunk_sections(sections)
    split_count = sum(1 for c in chunks if c["chunk_count"] > 1)
    print(f"Split into {len(chunks)} chunks ({split_count} chunks came from sections too long to embed whole).")

    texts = [build_text_for_embedding(chunk) for chunk in chunks]
    model = SentenceTransformer(settings.embedding_model_name)
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)

    store = VectorStore(dimension=int(vectors.shape[1]))
    store.add(vectors, chunks)
    store.save(index_dir)

    print(f"Saved {len(chunks)} chunks to {index_dir}")


if __name__ == "__main__":
    main()
