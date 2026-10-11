"""Re-scrape statutes from their recorded sources into backend/data/raw.

Every act's URL lives in backend/ingestion/sources.py, so re-scraping needs no
edits here. Files are written only after a scrape succeeds and passes a sanity
check, so a failed or truncated download can never overwrite good data.

Usage:
    python backend/scripts/scrape_acts.py                     # all acts
    python backend/scripts/scrape_acts.py --only ipc_full     # just one
    python backend/scripts/scrape_acts.py --dry-run           # scrape, write nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings
from backend.ingestion.scrapers.indian_kanoon import scrape_act
from backend.ingestion.sources import STATUTE_SOURCES, StatuteSource, find_source

# A real act always yields far more than this. Anything less means the page
# failed to render or its markup changed, and must not overwrite a good file.
MINIMUM_PLAUSIBLE_SECTIONS = 50


def check_scrape(records: List[Dict[str, Any]], source: StatuteSource) -> List[str]:
    """Return the reasons this scrape must not be written, if any."""
    problems: List[str] = []
    if len(records) < MINIMUM_PLAUSIBLE_SECTIONS:
        problems.append(f"only {len(records)} sections (expected at least {MINIMUM_PLAUSIBLE_SECTIONS})")

    numbers = [str(r.get("section_number", "")) for r in records]
    duplicates = {n for n in numbers if numbers.count(n) > 1}
    if duplicates:
        problems.append(f"{len(duplicates)} duplicated section number(s), e.g. {sorted(duplicates)[:5]}")

    if any(not str(r.get("text", "")).strip() for r in records):
        problems.append("at least one section has empty text")
    if any(not r.get("source_url") for r in records):
        problems.append("at least one section is missing its source_url")
    return problems


def scrape_one(source: StatuteSource, raw_dir: Path, dry_run: bool) -> bool:
    """Scrape a single act. Returns True when the file is left in a good state."""
    destination = raw_dir / source.filename
    previous = len(json.loads(destination.read_text(encoding="utf-8"))) if destination.exists() else 0

    try:
        records = scrape_act(source.url, source.act_name, timeout=240)
    except Exception as exc:
        print(f"  FAILED  {source.filename}: {exc}")
        return False

    problems = check_scrape(records, source)
    if problems:
        print(f"  REJECTED {source.filename}: {'; '.join(problems)}")
        print("           existing file left untouched")
        return False

    if dry_run:
        print(f"  would write {source.filename}: {previous} -> {len(records)} sections")
        return True

    destination.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {source.filename:45s} {previous:5d} -> {len(records):5d} sections   {source.url}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="scrape a single act by file stem or act name")
    parser.add_argument("--dry-run", action="store_true", help="scrape and check, but write nothing")
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / get_settings().raw_data_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    sources = (find_source(args.only),) if args.only else STATUTE_SOURCES

    print(f"Scraping {len(sources)} act(s) into {raw_dir}")
    failures = [source.filename for source in sources if not scrape_one(source, raw_dir, args.dry_run)]

    if failures:
        print(f"\n{len(failures)} act(s) did not update: {', '.join(failures)}")
        return 1
    print(f"\nAll {len(sources)} act(s) scraped successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
