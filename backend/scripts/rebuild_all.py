"""Rebuild the whole corpus and search index in one command.

Runs the four stages in the only order that is safe:

    1. scrape    pull every act from its recorded source          (scrape_acts.py)
    2. clean     apply the provably-safe cleaning rules           (clean_corpus.py)
    3. validate  print the health scoreboard, change nothing      (validate_corpus.py)
    4. build     chunk, embed, and write the FAISS index          (build_combined_index.py)

Order matters: embedding is the expensive step, so it runs last, on data that
has already been cleaned and checked. A stage that fails stops the run, so a
bad scrape never reaches the index.

Usage:
    python backend/scripts/rebuild_all.py                # the whole pipeline
    python backend/scripts/rebuild_all.py --skip-scrape  # reuse raw files as they are
    python backend/scripts/rebuild_all.py --only ipc_full
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent


def run_stage(name: str, command: List[str]) -> bool:
    print("\n" + "=" * 78)
    print(f"STAGE: {name}")
    print("=" * 78)
    started = time.time()
    completed = subprocess.run([sys.executable, *command], cwd=PROJECT_ROOT)
    elapsed = time.time() - started

    if completed.returncode != 0:
        print(f"\n{name} FAILED after {elapsed:.1f}s (exit code {completed.returncode}).")
        return False
    print(f"\n{name} finished in {elapsed:.1f}s.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-scrape", action="store_true", help="keep the existing raw files")
    parser.add_argument("--only", help="scrape just this act (implies a full rebuild of the index)")
    args = parser.parse_args()

    stages = []
    if not args.skip_scrape:
        scrape = [str(SCRIPTS / "scrape_acts.py")]
        if args.only:
            scrape += ["--only", args.only]
        stages.append(("1/4 scrape", scrape))
    stages += [
        ("2/4 clean", [str(SCRIPTS / "clean_corpus.py")]),
        ("3/4 validate", [str(SCRIPTS / "validate_corpus.py")]),
        ("4/4 build index", [str(SCRIPTS / "build_combined_index.py")]),
    ]

    started = time.time()
    for name, command in stages:
        if not run_stage(name, command):
            print("\nPipeline stopped. The index was NOT rebuilt, so the running app keeps using the last good one.")
            return 1

    print("\n" + "=" * 78)
    print(f"Rebuild complete in {time.time() - started:.1f}s. Restart the API to pick up the new index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
