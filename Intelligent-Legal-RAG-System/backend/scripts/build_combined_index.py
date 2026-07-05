"""Build a combined FAISS index and BM25 data for IPC and CrPC sections."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer

from backend.app.config import get_settings
from backend.app.services.hybrid_search import build_bm25_index
from backend.app.services.vector_store import VectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_sections(path: Path) -> List[Dict[str, Any]]:
    """Load statute sections from a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def build_text_for_embedding(section: Dict[str, Any]) -> str:
    """Create the embedding text for one statute section."""
    return f"{section['act']} Section {section['section_number']}: {section['section_title']}\n{section['text']}"


def main() -> None:
    settings = get_settings()
    ipc_path = PROJECT_ROOT / "backend/data/raw/ipc_full.json"
    crpc_path = PROJECT_ROOT / "backend/data/raw/code_of_criminal_procedure_1973.json"
    index_dir = PROJECT_ROOT / settings.faiss_index_dir

    ipc_sections = load_sections(ipc_path)
    crpc_sections = load_sections(crpc_path)
    sections = ipc_sections + crpc_sections

    texts = [build_text_for_embedding(section) for section in sections]
    model = SentenceTransformer(settings.embedding_model_name)
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)

    bm25_index = build_bm25_index(sections)
    print(f"Built BM25 index over {len(sections)} combined sections.")

    store = VectorStore(dimension=int(vectors.shape[1]))
    store.add(vectors, sections)
    store.save(index_dir)

    print(f"Saved {len(sections)} combined sections to {index_dir}")


if __name__ == "__main__":
    main()
