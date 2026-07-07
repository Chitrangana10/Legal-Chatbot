"""Helpers for loading raw statute section JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


REQUIRED_SECTION_KEYS = {"act", "section_number", "section_title", "text"}


def is_section_record(record: Any) -> bool:
    """Return True when a JSON item looks like a statute section record."""
    return isinstance(record, dict) and REQUIRED_SECTION_KEYS.issubset(record)


def load_raw_sections(raw_dir: Path) -> Tuple[List[Dict[str, Any]], List[Path], List[Path]]:
    """Load and combine every valid JSON file under the raw statute directory."""
    sections: List[Dict[str, Any]] = []
    loaded_files: List[Path] = []
    skipped_files: List[Path] = []

    for path in sorted(raw_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            skipped_files.append(path)
            continue

        if not isinstance(data, list) or not all(is_section_record(item) for item in data):
            skipped_files.append(path)
            continue

        sections.extend(data)
        loaded_files.append(path)

    return sections, loaded_files, skipped_files