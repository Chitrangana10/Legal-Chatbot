import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings
from backend.app.services.hybrid_search import HybridSearcher, tokenize
from backend.app.services.vector_store import VectorStore


def main() -> None:
    settings = get_settings()
    index_dir = PROJECT_ROOT / settings.faiss_index_dir
    vector_store = VectorStore.load(index_dir)
    sections = json.loads((PROJECT_ROOT / "backend/data/raw/ipc_full.json").read_text(encoding="utf-8"))

    hybrid_searcher = HybridSearcher(vector_store=vector_store, sections=sections)
    query = "what is section 304"

    print(f"Query: {query}")

    tokenized_query = tokenize(query)
    bm25_scores = hybrid_searcher.bm25_index.get_scores(tokenized_query)
    bm25_ranked_positions = sorted(range(len(bm25_scores)), key=lambda idx: bm25_scores[idx], reverse=True)[:5]
    print("RAW BM25-only top 5 results:")
    for index, position in enumerate(bm25_ranked_positions, start=1):
        metadata = sections[position]
        print(
            f"{index}. section_number={metadata.get('section_number')} "
            f"section_title={metadata.get('section_title')} "
            f"bm25_score={bm25_scores[position]:.6f}"
        )

    query_vector = hybrid_searcher.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=False)
    faiss_results = vector_store.search(query_vector, top_k=5)
    print("RAW FAISS-only top 5 results:")
    for index, result in enumerate(faiss_results, start=1):
        metadata = result.get("metadata", {})
        print(
            f"{index}. section_number={metadata.get('section_number')} "
            f"section_title={metadata.get('section_title')} "
            f"similarity_score={result.get('score')}"
        )

    results = hybrid_searcher.search(query, top_k=5)
    print("Top 5 hybrid results:")
    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        print(
            f"{index}. section_number={metadata.get('section_number')} "
            f"section_title={metadata.get('section_title')} "
            f"combined_score={result.get('combined_score')}"
        )

    section_304 = next((item for item in sections if str(item.get("section_number", "")) == "304"), None)
    if section_304:
        print("Section 304 present in results:", any(str(item.get("metadata", {}).get("section_number", "")) == "304" for item in results))
    else:
        print("Section 304 entry not found in ipc_full.json")


if __name__ == "__main__":
    main()
