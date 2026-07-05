"""Create embeddings for legal document chunks using sentence transformer models."""

from typing import Any, Dict, Iterable, List


def load_embedding_model(model_name: str) -> Any:
    """Load the configured embedding model."""
    raise NotImplementedError


def embed_chunks(chunks: Iterable[Dict[str, Any]], model: Any) -> List[Dict[str, Any]]:
    """Attach vector embeddings to legal retrieval chunks."""
    raise NotImplementedError

