"""Split long statute sections into overlapping, embed-sized chunks.

Why this exists (bug W3): the embedding model (all-MiniLM-L6-v2) truncates its
input at 256 word-pieces ("tokens"). A section longer than that silently loses
everything past the cut point before it is ever searchable - the vector
represents only the opening words. `chunk_section` splits a long section's text
into smaller pieces that each stay under that limit, with a small overlap
between consecutive pieces so an idea sitting on a cut boundary isn't lost by
both sides.

Every chunk is a full copy of the original section dict with `text` replaced by
just that piece, plus `chunk_index` / `chunk_count` - so a chunk always still
carries its parent `act` and `section_number`, and a citation built from it is
never missing which law it came from.

Legal text does NOT tokenize at a fixed rate: plain prose runs about 1.4 tokens
per word, but text full of numbers, brackets and citations - "[Substituted by
Act 4 of 1988, s. 2 (w.e.f. 1-4-1989)]" - can run over 4 tokens per word. A
fixed word-count chunk size calibrated on one section is wrong for another, so
chunking is done by counting REAL tokens with the model's own tokenizer, not by
guessing a word count. A word-count fallback is used only if the tokenizer
cannot be loaded (e.g. no network), and is deliberately conservative because it
cannot see the true density.

A section that already fits under the limit is returned unchanged as a single
chunk (chunk_index=0, chunk_count=1) - the ~1,950 sections that are already
fine are not touched.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

MAX_EMBED_TOKENS = 256  # sentence-transformers/all-MiniLM-L6-v2's hard limit
SAFETY_MARGIN_TOKENS = 8  # headroom for [CLS]/[SEP] and tokenizer rounding
MIN_CHUNK_TOKENS = 24  # never budget a chunk smaller than this, even if the header is huge

# Used only when the real tokenizer can't be loaded - picked conservatively
# against the worst legal-text density we measured (~4.5 tokens/word), not the
# ~1.4 average, since a wrong guess here would silently reintroduce W3.
DEFAULT_CHUNK_WORDS = 50
DEFAULT_OVERLAP_WORDS = 10

_tokenizer_cache: Dict[str, Any] = {}


def _default_tokenizer() -> Optional[Any]:
    """Lazily load and cache the real embedding tokenizer; None if unavailable."""
    if "tokenizer" not in _tokenizer_cache:
        try:
            from transformers import AutoTokenizer

            _tokenizer_cache["tokenizer"] = AutoTokenizer.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception:  # noqa: BLE001 - any failure means "fall back to word count"
            _tokenizer_cache["tokenizer"] = None
    return _tokenizer_cache["tokenizer"]


def _header_text(section: Dict[str, Any]) -> str:
    return f"{section.get('act', '')} Section {section.get('section_number', '')}: {section.get('section_title', '')}\n"


def chunk_section(
    section: Dict[str, Any],
    tokenizer: Optional[Any] = None,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> List[Dict[str, Any]]:
    """Split one section's `text` into chunks that each fit the embed budget.

    Uses the real tokenizer to size chunks accurately when one is available
    (pass one explicitly to reuse it across many calls - loading it per call
    would be slow); otherwise falls back to a conservative word count.
    """
    words = str(section.get("text", "")).split()
    if not words:
        whole = dict(section)
        whole["chunk_index"] = 0
        whole["chunk_count"] = 1
        return [whole]

    active_tokenizer = tokenizer if tokenizer is not None else _default_tokenizer()
    if active_tokenizer is not None:
        piece_texts = _pack_by_tokens(section, words, active_tokenizer, overlap_words)
    else:
        piece_texts = _pack_by_words(words, chunk_words, overlap_words)

    if len(piece_texts) == 1:
        whole = dict(section)
        whole["text"] = piece_texts[0]
        whole["chunk_index"] = 0
        whole["chunk_count"] = 1
        return [whole]

    chunks: List[Dict[str, Any]] = []
    for index, piece_text in enumerate(piece_texts):
        piece = dict(section)
        piece["text"] = piece_text
        piece["chunk_index"] = index
        piece["chunk_count"] = len(piece_texts)
        chunks.append(piece)
    return chunks


def _pack_by_tokens(section: Dict[str, Any], words: List[str], tokenizer: Any, overlap_words: int) -> List[str]:
    header_tokens = len(tokenizer.encode(_header_text(section), add_special_tokens=True))
    budget = max(MIN_CHUNK_TOKENS, MAX_EMBED_TOKENS - header_tokens - SAFETY_MARGIN_TOKENS)

    word_lengths = [len(tokenizer.encode(w, add_special_tokens=False)) or 1 for w in words]
    full_length = sum(word_lengths)
    if full_length <= budget:
        return [" ".join(words)]

    pieces: List[str] = []
    start = 0
    while start < len(words):
        end = start
        total = 0
        while end < len(words) and total + word_lengths[end] <= budget:
            total += word_lengths[end]
            end += 1
        if end == start:  # a single pathological word alone exceeds the budget
            end = start + 1
        pieces.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = max(end - overlap_words, start + 1)
    return pieces


def _pack_by_words(words: List[str], chunk_words: int, overlap_words: int) -> List[str]:
    if len(words) <= chunk_words:
        return [" ".join(words)]

    step = max(chunk_words - overlap_words, 1)
    pieces: List[str] = []
    start = 0
    while start < len(words):
        pieces.append(" ".join(words[start : start + chunk_words]))
        if start + chunk_words >= len(words):
            break
        start += step
    return pieces


def chunk_sections(
    sections: Sequence[Dict[str, Any]],
    tokenizer: Optional[Any] = None,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> List[Dict[str, Any]]:
    """Chunk every section in a list and flatten the result into one list."""
    active_tokenizer = tokenizer if tokenizer is not None else _default_tokenizer()
    flattened: List[Dict[str, Any]] = []
    for section in sections:
        flattened.extend(
            chunk_section(
                section,
                tokenizer=active_tokenizer,
                chunk_words=chunk_words,
                overlap_words=overlap_words,
            )
        )
    return flattened
