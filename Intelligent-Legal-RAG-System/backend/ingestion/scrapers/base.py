"""Define common interfaces for legal data source scrapers."""

from pathlib import Path
from typing import Any, Dict, Iterable


def fetch_source(source_url: str) -> Iterable[Dict[str, Any]]:
    """Fetch legal records from a configured source URL."""
    raise NotImplementedError


def write_raw_records(records: Iterable[Dict[str, Any]], output_dir: Path) -> Iterable[Path]:
    """Write fetched legal records to raw data files."""
    raise NotImplementedError

