"""The search side reads its chunks back out of the FAISS index.

RAGEngine no longer re-runs the chunker at startup; it takes the chunks the
build step already stored in the index. That is what keeps BM25 searching the
exact pieces FAISS was built over, so the order and content of the stored
metadata have to survive a save/load round trip intact.
"""

from pathlib import Path

import numpy as np

from backend.app.services.vector_store import VectorStore


def _chunk(number: str, chunk_index: int, chunk_count: int) -> dict:
    return {
        "act": "The Motor Vehicles Act, 1988",
        "section_number": number,
        "section_title": "Wearing of protective headgear.",
        "text": f"body of section {number} part {chunk_index}",
        "source_type": "statute",
        "source_url": "https://indiankanoon.org/doc/785258/",
        "scraped_at": "2026-09-18",
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
    }


def _chunks_as_rag_engine_reads_them(store: VectorStore) -> list:
    return [store.metadatas[position] for position in sorted(store.metadatas)]


def test_stored_chunks_round_trip_in_index_order(tmp_path: Path) -> None:
    chunks = [_chunk("129", 0, 3), _chunk("129", 1, 3), _chunk("129", 2, 3), _chunk("130", 0, 1)]
    store = VectorStore(dimension=4)
    store.add(np.eye(4, dtype="float32"), chunks)
    store.save(tmp_path)

    reloaded = VectorStore.load(tmp_path)
    recovered = _chunks_as_rag_engine_reads_them(reloaded)

    assert len(recovered) == reloaded.index.ntotal == 4
    assert [c["chunk_index"] for c in recovered] == [0, 1, 2, 0]
    assert [c["section_number"] for c in recovered] == ["129", "129", "129", "130"]
    assert recovered[1]["text"] == "body of section 129 part 1"


def test_every_indexed_vector_has_a_chunk_behind_it(tmp_path: Path) -> None:
    """A vector with no metadata would be an unciteable search hit."""
    chunks = [_chunk("129", 0, 2), _chunk("129", 1, 2)]
    store = VectorStore(dimension=4)
    store.add(np.eye(2, 4, dtype="float32"), chunks)
    store.save(tmp_path)

    reloaded = VectorStore.load(tmp_path)

    assert len(_chunks_as_rag_engine_reads_them(reloaded)) == reloaded.index.ntotal
    assert all(c.get("text") for c in reloaded.metadatas.values())
