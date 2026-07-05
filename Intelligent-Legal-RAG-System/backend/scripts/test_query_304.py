import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings
from backend.app.services.vector_store import VectorStore


def build_text_for_embedding(section: dict) -> str:
    return f"{section['act']} Section {section['section_number']}: {section['section_title']}\n{section['text']}"


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = vec_a.astype(np.float32)
    b = vec_b.astype(np.float32)
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm))


def main() -> None:
    settings = get_settings()
    index_dir = PROJECT_ROOT / settings.faiss_index_dir
    vector_store = VectorStore.load(index_dir)

    model = SentenceTransformer(settings.embedding_model_name)
    query = "what is section 304"
    query_vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=False)

    results = vector_store.search(query_vector, top_k=10)

    print(f"Query: {query}")
    print(f"FAISS index vectors: {vector_store.index.ntotal}")
    print("Top 10 FAISS results:")
    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        section_number = metadata.get("section_number", "")
        section_title = metadata.get("section_title", "")
        score = result.get("score")
        print(f"{index}. section_number={section_number} section_title={section_title} score={score}")

    ipc_full_path = PROJECT_ROOT / "backend/data/raw/ipc_full.json"
    ipc_sections = json.loads(ipc_full_path.read_text(encoding="utf-8"))
    section_304_entry = next((item for item in ipc_sections if str(item.get("section_number", "")) == "304"), None)

    if section_304_entry is None:
        print("SECTION 304 entry not found in ipc_full.json")
        return

    section_304_text = build_text_for_embedding(section_304_entry)
    section_304_vector = model.encode([section_304_text], convert_to_numpy=True, normalize_embeddings=False)
    direct_similarity = cosine_similarity(section_304_vector[0], query_vector[0])

    print("Section 304 embedding text:")
    print(section_304_text)
    print("Direct cosine similarity with query:", direct_similarity)

    top_scores = [result.get("score") for result in results]
    print("Top-10 FAISS scores range:", min(top_scores), "to", max(top_scores))
    if direct_similarity >= min(top_scores) - 1e-9:
        print("Direct similarity is within the top-10 score range")
    else:
        print("Direct similarity is below the top-10 score range")

    section_304_in_top10 = [r for r in results if str(r.get("metadata", {}).get("section_number", "")) == "304"]
    if section_304_in_top10:
        print("SECTION 304 found in top-10 at rank:", results.index(section_304_in_top10[0]) + 1)
    else:
        print("SECTION 304 not found in top-10")


if __name__ == "__main__":
    main()
