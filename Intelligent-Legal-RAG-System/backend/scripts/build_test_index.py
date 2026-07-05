"""Build a small FAISS test index from bundled IPC sample sections."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings
from backend.app.services.vector_store import VectorStore


def load_sample_sections(path: Path) -> List[Dict[str, Any]]:
    """Load sample IPC sections from JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def build_text_for_embedding(section: Dict[str, Any]) -> str:
    """Create the embedding text for one statute section."""
    return f"{section['act']} Section {section['section_number']}: {section['section_title']}\n{section['text']}"


def main() -> None:
    """Embed sample IPC sections and save a FAISS index."""
    settings = get_settings()
    sample_path = PROJECT_ROOT / settings.sample_ipc_path
    index_dir = PROJECT_ROOT / settings.faiss_index_dir

    sections = load_sample_sections(sample_path)
    texts = [build_text_for_embedding(section) for section in sections]

    model = SentenceTransformer(settings.embedding_model_name)
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)

    store = VectorStore(dimension=int(vectors.shape[1]))
    store.add(vectors, sections)
    store.save(index_dir)

    print(f"Indexed {len(sections)} IPC sections into {index_dir}")


if __name__ == "__main__":
    main()

