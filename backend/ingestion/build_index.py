"""Build and persist the FAISS index from processed legal document chunks."""

from pathlib import Path
from typing import Any, Dict, Iterable


def build_faiss_index(embedded_chunks: Iterable[Dict[str, Any]]) -> Any:
    """Build a FAISS index from embedded legal chunks."""
    raise NotImplementedError


def build_index_pipeline(processed_dir: Path, index_dir: Path) -> None:
    """Run the processed-data to vector-index build pipeline."""
    raise NotImplementedError

