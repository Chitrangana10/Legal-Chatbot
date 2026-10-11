"""Corpus health scoreboard.

Run this BEFORE and AFTER any data-cleaning or scraper change so every fix is
measurable instead of guessed. It only reads files - it never writes anything.

For each of the 8 raw statute files it reports:
  - record count and how many DISTINCT sections that really is
  - exact-duplicate records (same act + section + text)
  - key-duplicate records (same act + section number, DIFFERENT text -> W13)
  - junk records (empty / "Omitted" / stray dates -> scrape noise)
  - encoding-corruption records (the U+FFFD "replacement" char, e.g. "Murder.")
  - records whose embedding text is longer than the model's 256-token limit,
    so their tail is silently dropped before it is ever indexed (W3)
  - gaps: numbered sections missing below the highest section number seen

Then it compares the raw corpus against the committed FAISS index (W2):
  - raw sections that are absent from the index (unreachable by semantic search)
  - index sections that no longer exist in the raw files (stale vectors)

Usage:
    python backend/scripts/validate_corpus.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.hybrid_search import build_text_for_embedding, canonical_key

RAW_DIR = PROJECT_ROOT / "backend" / "data" / "raw"
INDEX_METADATA = PROJECT_ROOT / "backend" / "data" / "faiss_index" / "metadata.json"

# all-MiniLM-L6-v2 truncates its input at 256 word-pieces. Text past that point
# is dropped with no error, so it never influences the section's vector.
EMBED_TOKEN_LIMIT = 256

# Records whose text is really just "this provision was removed" - real, but they
# carry no searchable content and should not count as live sections.
JUNK_TEXT_PATTERN = re.compile(
    r"^\s*[\[\(]?\s*(omitted|repealed|rep\.? by|deleted|\*+)\s*[\]\)]?\s*\.?\s*$",
    re.IGNORECASE,
)
# A record whose entire text is a date fragment like "12-1991." - a broken scrape.
DATE_ONLY_PATTERN = re.compile(r"^\s*[\d]{1,4}[-/][\d]{1,4}([-/][\d]{1,4})?\.?\s*$")

# The leading number of a section id: "302" from "302", "115V" -> "115", "34A" -> "34".
LEADING_INT_PATTERN = re.compile(r"^(\d+)")


def _load_tokenizer():
    """Return the real embedding tokenizer, or None if it cannot be loaded.

    Counting with the actual tokenizer is exact. If transformers or the model
    files are unavailable (offline CI, for example) we fall back to a word count,
    which is close enough to see the scale of the truncation problem.
    """
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as exc:  # noqa: BLE001 - any failure means "no tokenizer"
        print(f"  (note: real tokenizer unavailable - {type(exc).__name__}; using word-count estimate)")
        return None


def _token_count(text: str, tokenizer) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
    # Rough estimate: statute English averages ~1.3 word-pieces per word.
    return int(len(text.split()) * 1.3)


def _is_junk(record: Dict[str, Any]) -> bool:
    """A garbled scrape fragment. Bare "[Omitted]" / "[Repealed]" stubs are NOT
    junk - the source shows them, so they are faithful and are kept."""
    text = str(record.get("text", "")).strip()
    title = str(record.get("section_title", "")).strip()
    letters = re.sub(r"[^a-z]", "", text.lower())
    if letters in {"omitted", "repealed", "deleted"} or "omitted by" in text.lower():
        return False
    if not text and not title:
        return True
    if DATE_ONLY_PATTERN.match(text):
        return True
    if not title and re.match(r"^\s*\d{1,2}[-/]\d{4}\.", text):
        return True
    return False


def _is_corrupt(record: Dict[str, Any]) -> bool:
    blob = f"{record.get('section_title', '')} {record.get('text', '')}"
    return "�" in blob or "Â" in blob or "â" in blob


def _leading_int(section_number: Any) -> int:
    match = LEADING_INT_PATTERN.match(str(section_number).strip())
    return int(match.group(1)) if match else -1


def analyse_file(path: Path, tokenizer) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))

    acts = sorted({str(r.get("act", "")) for r in records})
    keys = [canonical_key(r.get("act", ""), r.get("section_number", "")) for r in records]
    distinct_keys = set(keys)

    # Exact duplicates: same section id AND same title AND same text copied twice.
    # The title check keeps real neighbours (35C vs 35CC) that share boilerplate
    # "[Omitted by Act X]" text from being counted as duplicates.
    seen_exact: set[Tuple[str, str, str, str]] = set()
    exact_dupes = 0
    for r in records:
        fingerprint = (
            *canonical_key(r.get("act", ""), r.get("section_number", "")),
            str(r.get("section_title", "")).strip().lower(),
            str(r.get("text", "")).strip(),
        )
        if fingerprint in seen_exact:
            exact_dupes += 1
        else:
            seen_exact.add(fingerprint)

    # Key duplicates: same section id, but the text differs -> ambiguous identity.
    key_counts: Dict[Tuple[str, str], int] = {}
    for key in keys:
        key_counts[key] = key_counts.get(key, 0) + 1
    key_dupe_records = sum(count - 1 for count in key_counts.values() if count > 1)

    junk = sum(1 for r in records if _is_junk(r))
    corrupt = sum(1 for r in records if _is_corrupt(r))

    truncated = 0
    longest_tokens = 0
    for r in records:
        tokens = _token_count(build_text_for_embedding(r), tokenizer)
        longest_tokens = max(longest_tokens, tokens)
        if tokens > EMBED_TOKEN_LIMIT:
            truncated += 1

    # Gaps: which whole-number sections are missing below the highest one present.
    present_ints = {n for n in (_leading_int(r.get("section_number")) for r in records) if n > 0}
    highest = max(present_ints) if present_ints else 0
    missing_ints = sorted(set(range(1, highest + 1)) - present_ints)

    return {
        "file": path.name,
        "acts": acts,
        "records": len(records),
        "distinct_sections": len(distinct_keys),
        "exact_dupes": exact_dupes,
        "key_dupe_records": key_dupe_records,
        "junk": junk,
        "corrupt": corrupt,
        "truncated": truncated,
        "longest_tokens": longest_tokens,
        "highest_section": highest,
        "missing_count": len(missing_ints),
        "missing_sample": missing_ints[:20],
        "keys": distinct_keys,
    }


def compare_with_index(raw_keys: set[Tuple[str, str]]) -> Dict[str, Any]:
    if not INDEX_METADATA.exists():
        return {"available": False}

    payload = json.loads(INDEX_METADATA.read_text(encoding="utf-8"))
    metadatas = payload.get("metadatas", {})
    index_keys = {
        canonical_key(item.get("act", ""), item.get("section_number", ""))
        for item in metadatas.values()
    }
    return {
        "available": True,
        "index_vectors": len(metadatas),
        "index_keys": len(index_keys),
        "raw_not_in_index": len(raw_keys - index_keys),
        "index_not_in_raw": len(index_keys - raw_keys),
        "shared": len(raw_keys & index_keys),
    }


def main() -> None:
    print(f"Corpus validation  -  reading {RAW_DIR}\n")
    tokenizer = _load_tokenizer()
    print()

    files = [f for f in sorted(RAW_DIR.glob("*.json")) if not f.name.startswith("_")]
    if not files:
        print("No raw JSON files found.")
        return

    results = [analyse_file(path, tokenizer) for path in files]

    header = (
        f"{'file':<40} {'recs':>6} {'distinct':>9} {'exact_dup':>10} "
        f"{'key_dup':>8} {'junk':>6} {'corrupt':>8} {'trunc>256':>10} {'gaps':>6}"
    )
    print(header)
    print("-" * len(header))
    totals = {k: 0 for k in ("records", "distinct_sections", "exact_dupes", "key_dupe_records", "junk", "corrupt", "truncated", "missing_count")}
    for r in results:
        print(
            f"{r['file']:<40} {r['records']:>6} {r['distinct_sections']:>9} "
            f"{r['exact_dupes']:>10} {r['key_dupe_records']:>8} {r['junk']:>6} "
            f"{r['corrupt']:>8} {r['truncated']:>10} {r['missing_count']:>6}"
        )
        for k in totals:
            totals[k] += r[k]
    print("-" * len(header))
    print(
        f"{'TOTAL':<40} {totals['records']:>6} {totals['distinct_sections']:>9} "
        f"{totals['exact_dupes']:>10} {totals['key_dupe_records']:>8} {totals['junk']:>6} "
        f"{totals['corrupt']:>8} {totals['truncated']:>10} {totals['missing_count']:>6}"
    )

    print("\nPer-file detail:")
    for r in results:
        print(f"\n  {r['file']}")
        print(f"    act name(s) inside : {r['acts']}")
        print(f"    highest section    : {r['highest_section']}   (longest embedding text: {r['longest_tokens']} tokens)")
        if r["missing_sample"]:
            more = "" if r["missing_count"] <= 20 else f" ... (+{r['missing_count'] - 20} more)"
            print(f"    missing below top  : {r['missing_sample']}{more}")

    raw_keys: set[Tuple[str, str]] = set()
    for r in results:
        raw_keys |= r["keys"]

    print("\nRaw corpus vs committed FAISS index (W2):")
    cmp = compare_with_index(raw_keys)
    if not cmp["available"]:
        print("  index metadata not found - skipped")
    else:
        print(f"    index vectors            : {cmp['index_vectors']}")
        print(f"    distinct raw sections    : {len(raw_keys)}")
        print(f"    shared (can be retrieved): {cmp['shared']}")
        print(f"    raw sections NOT in index: {cmp['raw_not_in_index']}   <- unreachable by semantic search")
        print(f"    index keys NOT in raw    : {cmp['index_not_in_raw']}   <- stale vectors from an old snapshot")

    print("\nDone. Nothing was modified.")


if __name__ == "__main__":
    main()
