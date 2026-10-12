"""Translate everyday words into the wording the statute actually uses.

People do not ask questions in statute language. "How do I file an FIR?"
becomes the two words "file" and "fir" once grammar words are dropped, and
neither appears in CrPC section 154 - which is titled "Information in
cognisable cases" and never once says "FIR". Both retrievers then match
"file" against appeals and applications, and the question fails completely.

Nothing downstream can repair that: a reranker only reorders what retrieval
found, so if the right section is never a candidate it can never be shown.
Adding the statute's own phrasing to the query is what puts it in the pool.

Only phrases that genuinely differ from the law's wording belong here. A term
the statute already uses ("bail", "dowry", "theft") needs no help, and adding
it would only dilute the query.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# user phrasing -> the wording the relevant sections actually use
ALIASES: Dict[str, str] = {
    "fir": "information in cognisable cases reduced to writing officer in charge of a police station",
    "first information report": "information in cognisable cases officer in charge of a police station",
    "anticipatory bail": "direction for grant of bail to person apprehending arrest",
    "without a valid licence": "driving vehicles in contravention of section 3 or section 4",
    "without a valid license": "driving vehicles in contravention of section 3 or section 4",
    "without a licence": "driving vehicles in contravention of section 3 or section 4 necessity for driving licence",
    "without licence": "driving vehicles in contravention of section 3 or section 4 necessity for driving licence",
    "driving licence": "necessity for driving licence",
    "drivers licence": "necessity for driving licence",
    "helmet": "protective headgear",
    "drunk driving": "driving by a drunken person or under the influence of drugs",
    "drink and drive": "driving by a drunken person or under the influence of drugs",
    "hacking": "unauthorised access to a computer resource damage to computer",
    "phishing": "cheating by personation by using computer resource",
    "online fraud": "cheating by personation by using computer resource identity theft",
    "alimony": "order for maintenance of wives children and parents",
    "income tax return": "return of income",
    "tds": "deduction of tax at source",
}


def find_aliases(question: str) -> List[Tuple[str, str]]:
    """Return the (phrase, statute wording) pairs that appear in `question`."""
    lowered = question.lower()
    matched = [
        (phrase, wording)
        for phrase, wording in ALIASES.items()
        if re.search(rf"\b{re.escape(phrase)}\b", lowered)
    ]
    # "without a licence" already implies "driving licence"; keeping both would
    # repeat the same wording and skew the term counts BM25 relies on.
    longest_first = sorted(matched, key=lambda pair: len(pair[0]), reverse=True)
    kept: List[Tuple[str, str]] = []
    for phrase, wording in longest_first:
        if not any(phrase in seen for seen, _ in kept):
            kept.append((phrase, wording))
    return kept


def expand_query(question: str) -> str:
    """Append statute wording for any everyday phrases the question uses.

    Returns the question unchanged when nothing matches, so the overwhelming
    majority of queries are untouched and cannot be made worse.
    """
    matches = find_aliases(question)
    if not matches:
        return question
    return f"{question} {' '.join(wording for _phrase, wording in matches)}"
