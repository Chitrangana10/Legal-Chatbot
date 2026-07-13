"""Check exactly where Section 2 of Consumer Protection Act got cut off."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
raw_dir = PROJECT_ROOT / "backend/data/raw"

for file in raw_dir.glob("*.json"):
    if "consumer" in file.name.lower():
        sections = json.loads(file.read_text(encoding="utf-8"))
        for s in sections:
            if s["section_number"] == "2":
                print(f"Section 2 text length: {len(s['text'])} characters")
                print(f"Last 300 characters (where it ends):")
                print(s["text"][-300:])