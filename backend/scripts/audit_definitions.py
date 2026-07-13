"""Audit all scraped Acts for suspiciously short/truncated sections, especially Definitions."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
raw_dir = PROJECT_ROOT / "backend/data/raw"

for file in sorted(raw_dir.glob("*.json")):
    if file.name == "sample_ipc.json":
        continue
    sections = json.loads(file.read_text(encoding="utf-8"))
    print(f"\n=== {file.name} ({len(sections)} sections) ===")
    suspicious = [
        s for s in sections
        if ("definition" in s.get("section_title", "").lower() and len(s.get("text", "")) < 500)
    ]
    for s in suspicious:
        print(f"  SUSPICIOUS: Section {s['section_number']} '{s['section_title']}' - only {len(s['text'])} chars")