"""The scrape stage must never overwrite a good file with a bad scrape.

A statute page that fails to render still returns HTTP 200 with a near-empty
body, so "the request succeeded" is not enough to justify a write.
"""

import pytest

from backend.ingestion.sources import STATUTE_SOURCES, find_source
from backend.scripts.scrape_acts import MINIMUM_PLAUSIBLE_SECTIONS, check_scrape


def _records(count: int, **overrides) -> list:
    records = []
    for index in range(count):
        record = {
            "act": "The Indian Evidence Act, 1872",
            "section_number": str(index + 1),
            "section_title": f"Section {index + 1}",
            "amendment_note": "",
            "text": "Some substantive statutory text for this section.",
            "source_type": "statute",
            "source_url": "https://indiankanoon.org/doc/1953529/",
            "scraped_at": "2026-09-19",
        }
        record.update(overrides)
        records.append(record)
    return records


def test_every_source_is_unique_and_well_formed() -> None:
    filenames = [s.filename for s in STATUTE_SOURCES]
    document_ids = [s.document_id for s in STATUTE_SOURCES]

    assert len(set(filenames)) == len(filenames)
    assert len(set(document_ids)) == len(document_ids)
    for source in STATUTE_SOURCES:
        assert source.filename.endswith(".json")
        assert source.document_id.isdigit()
        assert source.url == f"https://indiankanoon.org/doc/{source.document_id}/"


def test_sources_can_be_looked_up_by_stem_or_act_name() -> None:
    assert find_source("ipc_full").document_id == "1569253"
    assert find_source("ipc_full.json").document_id == "1569253"
    assert find_source("Constitution of India").filename == "constitution_of_india.json"

    with pytest.raises(KeyError):
        find_source("not_a_real_act")


def test_a_healthy_scrape_is_accepted() -> None:
    assert check_scrape(_records(200), find_source("ipc_full")) == []


def test_a_near_empty_scrape_is_rejected() -> None:
    """A page that failed to render returns a handful of stray nodes, not 0."""
    problems = check_scrape(_records(MINIMUM_PLAUSIBLE_SECTIONS - 1), find_source("ipc_full"))
    assert any("expected at least" in p for p in problems)


def test_duplicate_section_numbers_are_rejected() -> None:
    records = _records(200)
    records[5]["section_number"] = records[4]["section_number"]

    problems = check_scrape(records, find_source("ipc_full"))
    assert any("duplicated section number" in p for p in problems)


def test_empty_text_is_rejected() -> None:
    records = _records(200)
    records[7]["text"] = "   "

    problems = check_scrape(records, find_source("ipc_full"))
    assert any("empty text" in p for p in problems)


def test_missing_provenance_is_rejected() -> None:
    """Without source_url a record cannot be traced back to what it came from."""
    records = _records(200)
    records[3]["source_url"] = ""

    problems = check_scrape(records, find_source("ipc_full"))
    assert any("source_url" in p for p in problems)
