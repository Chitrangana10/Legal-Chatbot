"""Manage FAISS vector indexes and metadata for legal document retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import faiss
import numpy as np


class VectorStore:
    """Persist and query a FAISS IndexFlatIP store with parallel JSON metadata."""

    index_filename = "index.faiss"
    metadata_filename = "metadata.json"

    def __init__(self, dimension: int) -> None:
        """Create an empty inner-product FAISS index for vectors of the given dimension."""
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadatas: Dict[int, Dict[str, Any]] = {}

    def add(self, vectors: Sequence[Sequence[float]], metadatas: Sequence[Dict[str, Any]]) -> None:
        """Normalize and add vectors with metadata records keyed by FAISS position."""
        vector_array = self._as_2d_float32(vectors)
        if vector_array.shape[0] != len(metadatas):
            raise ValueError("Number of vectors must match number of metadata records.")
        if vector_array.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors with dimension {self.dimension}.")

        faiss.normalize_L2(vector_array)
        start_position = self.index.ntotal
        self.index.add(vector_array)

        for offset, metadata in enumerate(metadatas):
            self.metadatas[start_position + offset] = dict(metadata)

    def search(self, query_vector: Sequence[float] | Sequence[Sequence[float]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the index with a normalized query vector and return scored metadata matches."""
        if self.index.ntotal == 0:
            return []

        query_array = self._as_2d_float32(query_vector)
        if query_array.shape[0] != 1:
            raise ValueError("Search expects a single query vector.")
        if query_array.shape[1] != self.dimension:
            raise ValueError(f"Expected query vector with dimension {self.dimension}.")

        faiss.normalize_L2(query_array)
        scores, positions = self.index.search(query_array, min(top_k, self.index.ntotal))

        results: List[Dict[str, Any]] = []
        for score, position in zip(scores[0], positions[0]):
            if position == -1:
                continue
            metadata = self.metadatas.get(int(position), {})
            results.append(
                {
                    "score": float(score),
                    "position": int(position),
                    "metadata": metadata,
                }
            )
        return results

    def save(self, path: Path | str) -> None:
        """Save the FAISS index and metadata JSON into the target directory."""
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(output_dir / self.index_filename))

        metadata_payload = {
            "dimension": self.dimension,
            "metadatas": {str(key): value for key, value in self.metadatas.items()},
        }
        (output_dir / self.metadata_filename).write_text(
            json.dumps(metadata_payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str) -> "VectorStore":
        """Load a FAISS index and metadata JSON from the target directory."""
        input_dir = Path(path)
        index = faiss.read_index(str(input_dir / cls.index_filename))
        metadata_payload = json.loads((input_dir / cls.metadata_filename).read_text(encoding="utf-8"))

        store = cls(dimension=int(metadata_payload["dimension"]))
        store.index = index
        store.metadatas = {int(key): value for key, value in metadata_payload["metadatas"].items()}
        return store

    @staticmethod
    def _as_2d_float32(vectors: Sequence[float] | Sequence[Sequence[float]]) -> np.ndarray:
        """Convert a vector or batch of vectors into a two-dimensional float32 array."""
        array = np.asarray(vectors, dtype="float32")
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError("Vectors must be one- or two-dimensional.")
        return np.ascontiguousarray(array)

