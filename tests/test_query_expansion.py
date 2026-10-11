"""Everyday wording must be translated into the statute's wording before search.

"How do I file an FIR?" reduces to the two words "file" and "fir". Neither
appears in CrPC section 154 ("Information in cognisable cases"), so both
retrievers match "file" against appeals and applications and the right
section is never even a candidate - which no amount of reranking can fix.
"""

from backend.app.services.query_expansion import ALIASES, expand_query, find_aliases


def test_jargon_is_translated_into_statute_wording() -> None:
    expanded = expand_query("How do I file an FIR?")

    assert "How do I file an FIR?" in expanded, "the user's own words must survive"
    assert "cognisable" in expanded.lower()
    assert "police station" in expanded.lower()


def test_a_question_with_no_jargon_is_left_exactly_alone() -> None:
    """Most questions need no help, and must not be diluted with extra words."""
    question = "What is the punishment for murder?"

    assert expand_query(question) == question


def test_matching_is_word_bounded() -> None:
    """"fir" must not fire inside "first", "firm" or "confirm"."""
    for question in ("Can I confirm my firm's registration?", "What is the first schedule?"):
        assert find_aliases(question) == [], question


def test_the_longest_phrase_wins_over_its_own_substring() -> None:
    """"without a licence" already implies "driving licence"; adding both
    would repeat the same wording and skew the counts BM25 relies on."""
    matched = [phrase for phrase, _wording in find_aliases("fine for driving without a licence")]

    assert "without a licence" in matched
    assert "driving licence" not in matched


def test_several_unrelated_aliases_can_apply_together() -> None:
    matched = dict(find_aliases("is a helmet needed and what about drunk driving"))

    assert "helmet" in matched
    assert "drunk driving" in matched


def test_aliases_only_cover_wording_the_statute_does_not_use() -> None:
    """A word the law already uses needs no alias - it would only dilute the
    query. These are all present in the corpus verbatim."""
    for already_in_the_law in ("bail", "dowry", "theft", "murder", "arrest", "defamation"):
        assert already_in_the_law not in ALIASES
