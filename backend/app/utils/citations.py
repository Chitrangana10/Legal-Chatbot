"""Format and validate citations for grounded legal responses."""

from typing import Any, Dict, List


def extract_citations(retrieved_contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract citation payloads from retrieved legal contexts."""
    raise NotImplementedError


def format_citation(citation: Dict[str, Any]) -> str:
    """Format a citation for display in an answer or UI."""
    raise NotImplementedError

