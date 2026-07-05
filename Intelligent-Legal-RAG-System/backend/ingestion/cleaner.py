"""Clean raw legal documents into normalized text and metadata records."""

from pathlib import Path
from typing import Any, Dict, Iterable


def clean_document(raw_path: Path) -> Dict[str, Any]:
    """Clean a raw legal document and return normalized content with metadata."""
    raise NotImplementedError


def clean_corpus(raw_dir: Path, output_dir: Path) -> Iterable[Path]:
    """Clean every supported document in a raw corpus directory."""
    raise NotImplementedError

