"""Regression tests for W1: FAISS and BM25 results must fuse on a shared identity.

The FAISS index and the raw corpus disagree on surface form for the two
most-queried Acts (IPC, CrPC). Without normalization their results never fuse.
"""

from backend.app.services.hybrid_search import canonical_key


def test_leading_article_is_ignored() -> None:
    """FAISS drops "The" from some act names; the raw corpus keeps it."""
    assert canonical_key("Indian Penal Code, 1860", "302") == canonical_key(
        "The Indian Penal Code, 1860", "302"
    )


def test_trailing_period_on_section_number_is_ignored() -> None:
    """CrPC section numbers carry a trailing period in the index only."""
    assert canonical_key("Code of Criminal Procedure, 1973", "247.") == canonical_key(
        "The Code of Criminal Procedure, 1973", "247"
    )


def test_distinct_sections_still_differ() -> None:
    """Normalization must not collapse genuinely different sections."""
    assert canonical_key("Indian Penal Code, 1860", "302") != canonical_key(
        "Indian Penal Code, 1860", "303"
    )
    assert canonical_key("Indian Penal Code, 1860", "1") != canonical_key(
        "Constitution of India", "1"
    )
