"""Split cleaned legal documents into citation-preserving retrieval chunks."""

from typing import Any, Dict, List


def chunk_document(document: Dict[str, Any], chunk_size: int = 800, overlap: int = 120) -> List[Dict[str, Any]]:
    """Split one cleaned document into overlapping retrieval chunks."""
    raise NotImplementedError


def assign_chunk_ids(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assign stable identifiers to legal document chunks."""
    raise NotImplementedError

