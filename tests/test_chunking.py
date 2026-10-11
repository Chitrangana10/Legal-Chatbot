"""Regression tests for W3: long sections must not be silently truncated."""

from backend.app.services.chunking import chunk_section, chunk_sections


def _section(text: str, **overrides):
    base = {
        "act": "Indian Penal Code, 1860",
        "section_number": "375",
        "section_title": "Rape.",
        "text": text,
        "source_type": "statute",
    }
    base.update(overrides)
    return base


def test_short_section_is_returned_unchanged_as_a_single_chunk() -> None:
    section = _section("A short provision that easily fits under the token limit.")
    chunks = chunk_section(section)

    assert len(chunks) == 1
    assert chunks[0]["text"] == section["text"]
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["chunk_count"] == 1


def test_long_section_is_split_into_multiple_chunks() -> None:
    long_text = " ".join(f"word{i}" for i in range(2000))
    section = _section(long_text)

    chunks = chunk_section(section)

    assert len(chunks) > 1
    assert all(c["chunk_count"] == len(chunks) for c in chunks)
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_every_chunk_keeps_the_parent_citation() -> None:
    long_text = " ".join(f"word{i}" for i in range(2000))
    section = _section(long_text, act="The Income Tax Act, 1961", section_number="80C")

    chunks = chunk_section(section)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["act"] == "The Income Tax Act, 1961"
        assert chunk["section_number"] == "80C"
        assert chunk["section_title"] == "Rape."  # title carried through unchanged


def test_consecutive_chunks_overlap_so_no_words_are_lost() -> None:
    words = [f"word{i}" for i in range(2000)]
    section = _section(" ".join(words))

    chunks = chunk_section(section)

    # Every word that appears in the original text must appear in at least one chunk.
    seen = set()
    for chunk in chunks:
        seen.update(chunk["text"].split())
    assert seen == set(words)

    # Consecutive chunks share at least one word (the overlap).
    for first, second in zip(chunks, chunks[1:]):
        first_words = set(first["text"].split())
        second_words = set(second["text"].split())
        assert first_words & second_words, "consecutive chunks should overlap"


def test_no_chunk_exceeds_the_real_embedding_token_limit() -> None:
    from backend.app.services.chunking import MAX_EMBED_TOKENS, _default_tokenizer
    from backend.app.services.hybrid_search import build_text_for_embedding

    tokenizer = _default_tokenizer()
    if tokenizer is None:
        return  # offline environment - the word-count fallback is checked separately

    long_text = " ".join(
        "Provided that the Central Government may, by notification in the Official Gazette,"
        for _ in range(200)
    )
    section = _section(long_text, section_title="A moderately long heading for this test.")

    for chunk in chunk_section(section, tokenizer=tokenizer):
        token_count = len(
            tokenizer.encode(build_text_for_embedding(chunk), add_special_tokens=True, truncation=False)
        )
        assert token_count <= MAX_EMBED_TOKENS


def test_chunk_sections_flattens_a_list_of_sections() -> None:
    short = _section("Fits in one chunk.", section_number="1")
    long_text = " ".join(f"word{i}" for i in range(2000))
    long = _section(long_text, section_number="2")

    flattened = chunk_sections([short, long])

    assert flattened[0]["section_number"] == "1"
    assert sum(1 for c in flattened if c["section_number"] == "2") > 1
