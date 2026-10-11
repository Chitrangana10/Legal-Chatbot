"""BM25 must not rank a section on grammar words alone.

Constitution article 69 ("Oath of the Vice-President") was the TOP keyword hit
for "What are my rights if I am arrested?" - it matches none of the meaningful
words in that question, but the oath repeats "I" four times. rank_bm25 floors
a would-be-negative idf at a small positive value, so a word that appears in
nearly every document still adds score rather than being ignored.
"""

from backend.app.services.hybrid_search import build_bm25_index, tokenize


def test_grammar_words_are_dropped() -> None:
    assert tokenize("What are my rights if I am arrested?") == ["rights", "arrested"]


def test_words_that_change_the_law_are_kept() -> None:
    """"shall not" and "shall" mean opposite things; "under section 41" names a
    specific provision. Dropping these would change what a section says."""
    kept = tokenize("shall not be detained under section 41 without an order before trial")
    for word in ("shall", "not", "under", "without", "before", "section", "41"):
        assert word in kept


def test_case_and_punctuation_are_still_ignored() -> None:
    assert tokenize("Murder, PUNISHMENT; rape.") == ["murder", "punishment", "rape"]


def test_a_query_of_only_grammar_words_yields_nothing_to_search() -> None:
    """HybridSearcher skips BM25 on an empty token list and leans on FAISS."""
    assert tokenize("what is the of and") == []


def _section(number: str, title: str, text: str) -> dict:
    return {"act": "Test Act", "section_number": number, "section_title": title, "text": text}


def test_the_oath_no_longer_outranks_arrest_law() -> None:
    """The real failure, in miniature. Needs filler sections because BM25 idf
    is meaningless on a two-document corpus - every term would score zero."""
    oath = _section(
        "69",
        "Oath or affirmation by the Vice-President",
        "I, A.B., do swear that I will bear true faith and allegiance, "
        "and that I will faithfully discharge my duty.",
    )
    arrest = _section(
        "50",
        "Person arrested to be informed of grounds of arrest",
        "Every police officer arresting any person shall forthwith communicate to him "
        "the grounds for such arrest and his rights to bail.",
    )
    filler = [
        _section("1", "Short title", "This Act may be called the Test Act."),
        _section("2", "Definitions", "In this Act, unless the context otherwise requires."),
        _section("3", "Stamp duty", "Every instrument shall bear a stamp of the prescribed value."),
        _section("4", "Registration of vehicles", "No vehicle shall be driven unless registered."),
        _section("5", "Audit of accounts", "The accounts shall be audited every financial year."),
        _section("6", "Power to make rules", "The Central Government may make rules by notification."),
    ]
    corpus = [oath, arrest, *filler]

    scores = build_bm25_index(corpus).get_scores(tokenize("What are my rights if I am arrested?"))

    assert scores[1] > scores[0], "the arrest section must outrank the oath"
    assert scores[0] == 0, "the oath shares no meaningful word with the question"
