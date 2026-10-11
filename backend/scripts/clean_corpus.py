"""Cleaning pass over the 8 raw statute files.

This applies ONLY changes that are provably safe - it never guesses. Anything
uncertain (the Schedule-vs-body boundary) is written to the report for the
step-4 scraper rework, not touched here.

What it does, per file:
  Rule 1  Remove exact-duplicate records - same section id AND same title AND
          identical text (after whitespace is normalised). The first copy is
          kept. The title check stops a real section that shares boilerplate
          "[Omitted by Act X]" text with its neighbour from being deleted.
  Rule 2  Remove garbled fragments only - text that is empty, a bare date
          fragment ("12-1991."), or an untitled record whose text starts with a
          stray date prefix. These are scrape corruption, not source content.
          Bare "[Omitted]" / "[Repealed]" stubs ARE kept - the source shows them,
          so removing them would make us less faithful, not more.
  Rule 3  Normalise text - strip the invisible soft-hyphen character, collapse
          runs of whitespace, trim ends. No wording is changed.

Every removed or modified record is written to _cleaning_report.json so the pass
is fully auditable and reversible. The cleaned files are written back in place;
git holds the previous version.

Usage:
    python backend/scripts/clean_corpus.py            # apply and overwrite
    python backend/scripts/clean_corpus.py --dry-run  # report only, write nothing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.hybrid_search import canonical_key

RAW_DIR = PROJECT_ROOT / "backend" / "data" / "raw"
REPORT_PATH = RAW_DIR / "_cleaning_report.json"

SOFT_HYPHEN = "­"
DATE_ONLY = re.compile(r"^\s*\d{1,4}[-/]\d{1,4}(?:[-/]\d{1,4})?\.?\s*$")
DATE_PREFIX = re.compile(r"^\s*\d{1,2}[-/]\d{4}\.")  # "12-1991. ..." broken CrPC fragments
# Faithful-to-source stubs - kept, never treated as junk.
SOURCE_STUBS = {"omitted", "repealed", "deleted"}


def normalise(text: str) -> str:
    """Rule 3: drop soft hyphens, collapse whitespace, trim."""
    return re.sub(r"\s+", " ", text.replace(SOFT_HYPHEN, "")).strip()


def letters_only(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def is_junk(record: Dict[str, Any]) -> Tuple[bool, str]:
    """Rule 2: return (True, reason) only for garbled scrape fragments."""
    text = normalise(str(record.get("text", "")))
    title = normalise(str(record.get("section_title", "")))

    if letters_only(text) in SOURCE_STUBS or "omitted by" in text.lower():
        return False, ""  # faithful stub - always keep
    if not text and not title:
        return True, "both text and title are empty"
    if DATE_ONLY.match(text):
        return True, "text is only a date fragment"
    if not title and DATE_PREFIX.match(text):
        return True, "untitled record starting with a stray date fragment"
    # A tiny text next to a long title means the scraper mis-split a heading -
    # the content is still there, so keep it and let step 4 fix the field split.
    return False, ""


def clean_file(path: Path) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))

    removed_junk: List[Dict[str, Any]] = []
    removed_dupes: List[Dict[str, Any]] = []
    normalised_count = 0
    kept: List[Dict[str, Any]] = []
    seen_fingerprints: set[Tuple[str, str, str]] = set()

    for position, record in enumerate(records):
        junk, reason = is_junk(record)
        if junk:
            removed_junk.append({"position": position, "reason": reason, "record": record})
            continue

        # Rule 3 - normalise the two text fields, count if anything changed.
        new_record = dict(record)
        changed = False
        for field in ("section_title", "text"):
            original = str(record.get(field, ""))
            cleaned = normalise(original)
            if cleaned != original:
                new_record[field] = cleaned
                changed = True
        if changed:
            normalised_count += 1

        # Rule 1 - exact-duplicate detection on normalised key + title + text.
        fingerprint = (
            *canonical_key(new_record.get("act", ""), new_record.get("section_number", "")),
            normalise(str(new_record.get("section_title", ""))).lower(),
            normalise(str(new_record.get("text", ""))),
        )
        if fingerprint in seen_fingerprints:
            removed_dupes.append({"position": position, "record": record})
            continue
        seen_fingerprints.add(fingerprint)
        kept.append(new_record)

    return {
        "file": path.name,
        "before": len(records),
        "after": len(kept),
        "removed_junk": removed_junk,
        "removed_dupes": removed_dupes,
        "normalised": normalised_count,
        "cleaned_records": kept,
    }


def flag_schedule_candidates(cleaned_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Report-only: records that look like Schedule paragraphs mis-labelled as
    body sections. NOT modified - the body/Schedule split needs the source HTML
    structure and is handled in step 4."""
    flags: List[Dict[str, Any]] = []
    for position, record in enumerate(cleaned_records):
        text = str(record.get("text", ""))
        title = str(record.get("section_title", ""))
        blob = f"{title} {text}".lower()
        looks_schedule = (
            "in this schedule" in blob
            or "in this part," in blob
            or re.search(r"\blist [ivx]+ *-", blob)
            or re.match(r"^\s*rule \d", text.lower())
        )
        if looks_schedule:
            flags.append(
                {
                    "position": position,
                    "act": record.get("act"),
                    "section_number": record.get("section_number"),
                    "section_title": title[:80],
                }
            )
    return flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = parser.parse_args()

    files = sorted(RAW_DIR.glob("*.json"))
    files = [f for f in files if not f.name.startswith("_")]
    if not files:
        print("No raw JSON files found.")
        return

    report: Dict[str, Any] = {"mode": "dry-run" if args.dry_run else "applied", "files": []}
    grand = {"before": 0, "after": 0, "junk": 0, "dupes": 0, "normalised": 0, "schedule_flags": 0}

    print(f"{'file':<40} {'before':>7} {'after':>7} {'junk':>6} {'dupes':>7} {'norm':>6} {'sched?':>7}")
    print("-" * 84)

    for path in files:
        result = clean_file(path)
        schedule_flags = flag_schedule_candidates(result["cleaned_records"])

        n_junk = len(result["removed_junk"])
        n_dupes = len(result["removed_dupes"])
        print(
            f"{result['file']:<40} {result['before']:>7} {result['after']:>7} "
            f"{n_junk:>6} {n_dupes:>7} {result['normalised']:>6} {len(schedule_flags):>7}"
        )

        grand["before"] += result["before"]
        grand["after"] += result["after"]
        grand["junk"] += n_junk
        grand["dupes"] += n_dupes
        grand["normalised"] += result["normalised"]
        grand["schedule_flags"] += len(schedule_flags)

        report["files"].append(
            {
                "file": result["file"],
                "before": result["before"],
                "after": result["after"],
                "normalised": result["normalised"],
                "removed_junk": result["removed_junk"],
                "removed_dupes": result["removed_dupes"],
                "schedule_candidates_for_step4": schedule_flags,
            }
        )

        if not args.dry_run:
            path.write_text(
                json.dumps(result["cleaned_records"], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    print("-" * 84)
    print(
        f"{'TOTAL':<40} {grand['before']:>7} {grand['after']:>7} "
        f"{grand['junk']:>6} {grand['dupes']:>7} {grand['normalised']:>6} {grand['schedule_flags']:>7}"
    )

    if not args.dry_run:
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nAudit trail written to {REPORT_PATH.relative_to(PROJECT_ROOT)}")
        print("Cleaned files written in place. Previous versions are in git.")
    else:
        print("\nDry run - no files changed.")
    print(
        f"\nNote: {grand['schedule_flags']} records look like Schedule paragraphs mis-labelled "
        "as body sections.\nThey were NOT changed - the body/Schedule split needs the source "
        "structure and is done in step 4."
    )


if __name__ == "__main__":
    main()
