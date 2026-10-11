"""Scrape Indian Kanoon statute pages into section dictionaries."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import requests
from bs4 import BeautifulSoup


SECTION_HEADING_PATTERN = re.compile(
    r"^(?P<number>\d+[A-Za-z]{0,4}(?:-[A-Za-z]{1,3})?)\.?\s+(?P<title>.*)$", re.S
)
# The section node's own anchor id ("section_376DA") is the authoritative
# section number. The <h3> heading is not: the source sometimes splits a
# multi-letter suffix across the heading text ("376DA." rendering as
# "376D" + "A. ..."), which silently merges distinct sections. Verified
# across all eight acts: every section node carries a unique, correctly
# numbered id whose number always prefixes its own heading.
SECTION_ID_PATTERN = re.compile(r"^section_(?P<number>[0-9][0-9A-Za-z\-]*)$")
# Used only to repair a heading whose trailing suffix letter was split off,
# and only when the repaired number restores ascending order (see
# _assign_section_number).
SPLIT_SUFFIX_PATTERN = re.compile(r"^(?P<letter>[A-Z])\.\s*(?P<title>\S.*)$", re.S)
# Indian bare acts conventionally write a heading as "Title. - Body..." - the
# first ". -" (or em/en dash) after the number is where the short title ends
# and the substantive text begins. Used to recover a clean title (and any body
# text riding along with it) instead of letting the title swallow everything.
TITLE_BODY_SPLIT_PATTERN = re.compile(r"^(?P<title>.*?[.:])\s*[-–—]\s*(?P<body>.+)$", re.S)
SECTION_2_URL = "https://indiankanoon.org/doc/48103131/"


ZERO_WIDTH_CHARACTERS = dict.fromkeys(map(ord, "​‌‍﻿­"))
# A heading is printed as "Title.— Body". Where the body lives outside the
# heading the dash is left dangling on the end of the title ("Punishment for
# murder.—"), which then shows up in every citation. It is a separator, not
# part of the title, so it comes off.
TRAILING_SEPARATOR_PATTERN = re.compile(r"[\s\-–—]+$")


def _normalize_text(text: str) -> str:
    return " ".join(text.translate(ZERO_WIDTH_CHARACTERS).split())


def _clean_title(title: str) -> str:
    return TRAILING_SEPARATOR_PATTERN.sub("", title).strip()


# An amended section is headed "[ Real Title. [Inserted by Act 25 of 2005 ...]".
# The citation is provenance, not part of the section's name, and it is the
# only place the source records it - so it moves to its own field rather than
# being discarded. Left in the title it shows in every citation AND rides in
# the header of every chunk's embedding text.
AMENDMENT_CITATION_PATTERN = re.compile(
    r"\[\s*(?:Substituted|Inserted|Omitted|Added|Renumbered|Repealed|Subs\.|Ins\.|Rep\.)", re.I
)


def _split_title_and_note(title: str) -> Tuple[str, str]:
    """Separate a heading into (section title, amendment note)."""
    cleaned = title.strip()
    note = ""

    match = AMENDMENT_CITATION_PATTERN.search(cleaned)
    if match:
        candidate = cleaned[: match.start()].lstrip("[").strip()
        candidate = _clean_title(candidate)
        # A title that is nothing BUT the marker (e.g. "[Omitted]") is the
        # section's actual name - splitting it would leave no title at all.
        if candidate:
            note = cleaned[match.start():].strip()
            cleaned = candidate

    # Strip only a bracket left dangling by the split. A balanced pair is the
    # source's own convention for amended wording ("[Omitted]") and stays.
    if cleaned.count("[") > cleaned.count("]"):
        cleaned = cleaned.lstrip("[").strip()
    if cleaned.count("]") > cleaned.count("["):
        cleaned = cleaned.rstrip("]").strip()
    return _clean_title(cleaned), note


def _extract_act_title(soup: BeautifulSoup, fallback: str) -> str:
    for heading in soup.find_all("h1"):
        text = _normalize_text(heading.get_text(" ", strip=True))
        if text and text != "Indian Kanoon - Search engine for Indian Law":
            return text
    return fallback


def _parse_section_heading(heading_text: str) -> tuple[str, str, str]:
    """Split a raw <h3> heading into (number, short title, any inline body text).

    Most sections render as just "N. Title." with the body living elsewhere in
    the page, so inline body is usually empty. Some sections render the WHOLE
    section - title and body - inside the heading itself; for those,
    TITLE_BODY_SPLIT_PATTERN recovers a clean title and returns the body
    separately instead of one giant "title" containing the whole section.
    """
    cleaned = _normalize_text(heading_text)
    match = SECTION_HEADING_PATTERN.match(cleaned)
    if not match:
        return cleaned, cleaned, ""

    section_number = match.group("number").rstrip(".")
    raw_title = match.group("title").strip()

    split = TITLE_BODY_SPLIT_PATTERN.match(raw_title)
    if split:
        return section_number, _clean_title(split.group("title")), split.group("body").strip()
    return section_number, _clean_title(raw_title), ""


def _number_sort_key(number: str) -> Tuple[int, str]:
    """Order section numbers the way a statute does: 271, 271A, 271AA, 271B.

    Plain string comparison on the letter suffix already gives that order,
    because a shorter suffix sorts before any longer suffix it prefixes.
    Returns None for anything that is not a section number at all.
    """
    cleaned = str(number).replace("-", "").strip()
    match = re.match(r"^(\d+)([A-Za-z]*)$", cleaned)
    if not match:
        return None
    return int(match.group(1)), match.group(2).upper()


def _segment_sections(soup: BeautifulSoup) -> List[Tuple[Any, str]]:
    """Pair every section node with the text it owns, in document order.

    A node's own get_text() is not the section's text. On these pages a
    section's body can spill out past its node (Constitution article 332A
    keeps only clause 1 inside it), and a following section can be nested
    INSIDE it (article 334A sits inside article 334's node, so 334's own
    get_text() would swallow all of 334A). Slicing the page's flat string
    stream from one section's start to the next section's start gives each
    one exactly its own text in both cases.
    """
    strings = list(soup.strings)
    position = {id(string): index for index, string in enumerate(strings)}
    nodes = soup.select("section.akn-section")

    def start_of(tag: Any):
        first = next(iter(tag.strings), None)
        return position.get(id(first)) if first is not None else None

    starts: List[Any] = [start_of(node) for node in nodes]
    # A Chapter or Part heading between two sections belongs to neither, so it
    # ends the section before it. No section node on any of these pages
    # contains an <h1> or <h2>, so they are safe to treat as hard boundaries.
    division_starts = sorted(
        start for start in (start_of(tag) for tag in soup.find_all(["h1", "h2"])) if start is not None
    )

    segments: List[Tuple[Any, str]] = []
    for index, node in enumerate(nodes):
        start = starts[index]
        if start is None:
            segments.append((node, ""))
            continue
        end = None
        for later in range(index + 1, len(nodes)):
            if starts[later] is not None and starts[later] > start:
                end = starts[later]
                break
        for division in division_starts:
            if division > start and (end is None or division < end):
                end = division
                break
        piece = strings[start:end] if end is not None else strings[start:]
        segments.append((node, _normalize_text(" ".join(str(s) for s in piece))))
    return segments


def _act_body_segments(segments: List[Tuple[Any, str]]) -> List[Tuple[Any, str]]:
    """Drop everything after the Act's last real section.

    Schedules are rendered with the same markup as sections and are not
    labelled as schedules on every page, so they cannot be recognised by
    name. They are recognisable by numbering: a statute's sections run
    strictly ascending (verified - zero backward jumps across all eight
    acts), and a schedule restarts the count (Income Tax 298 -> 19,
    Constitution 395 -> 12A). Everything from that restart on is schedule
    material, as is any id-less node trailing after the last real section.
    """
    limit = len(segments)
    previous_key = None
    last_real = None
    for index, (node, _text) in enumerate(segments):
        node_id = node.get("id")
        if not node_id:
            continue
        match = SECTION_ID_PATTERN.match(node_id)
        if match is None:
            continue
        key = _number_sort_key(match.group("number"))
        if key is None:
            continue
        if previous_key is not None and key < previous_key:
            limit = index
            break
        previous_key = key
        last_real = index

    if last_real is not None:
        limit = min(limit, last_real + 1)
    return segments[:limit]


def _assign_section_number(node: Any, heading_text: str, previous_key) -> Tuple[str, str, str]:
    """Return (number, title, inline_body) for one section node.

    Prefers the node's anchor id, which is authoritative. For a node with no
    id - a section the source page failed to tag at all - the heading is all
    there is, so the number is read from it and only accepted if it continues
    the Act's ascending order. If it does not, the heading may be one whose
    suffix letter was split off ("271AA. B.Penalty ..." for 271AAB); joining
    that letter back on is accepted only when doing so restores the order.
    """
    number, title, inline_body = _parse_section_heading(heading_text)

    node_id = node.get("id")
    if node_id:
        match = SECTION_ID_PATTERN.match(node_id)
        if match:
            number = match.group("number")
        return number, title, inline_body

    if previous_key is not None and _number_sort_key(number) is not None:
        if _number_sort_key(number) <= previous_key:
            split = SPLIT_SUFFIX_PATTERN.match(title)
            if split:
                joined = f"{number}{split.group('letter')}"
                if (_number_sort_key(joined) or (0, "")) > previous_key:
                    rest = split.group("title").strip()
                    body_split = TITLE_BODY_SPLIT_PATTERN.match(rest)
                    if body_split:
                        return joined, _clean_title(body_split.group("title")), body_split.group("body").strip()
                    return joined, _clean_title(rest), ""
    return number, title, inline_body


# A section inserted by a later amendment sometimes gets no DOM node of its
# own at all - the source page glues it, still wrapped in "[ ... ]", onto the
# end of whichever section happened to precede it (e.g. Motor Vehicles Act
# section 129 sits inside section 128's own node, with no id="section_129"
# anywhere on the page). BRACKET_HEADING_START recognizes where such an
# embedded section begins.
BRACKET_HEADING_START = re.compile(r"^\d+[A-Za-z]{0,2}\.\s+\S")
# A bracket whose content is nothing but repeal punctuation ("***", "- - -")
# means that section was formally omitted by amendment - there is no text to
# recover, and creating an empty record for it would be its own inaccuracy.
REPEAL_MARKER_PATTERN = re.compile(r"^[\s*\-–—]*$")
# Statutes also bracket a phrase that an amendment substituted INTO a
# sentence, and such a phrase can begin with a section number too - Income
# Tax section 246A contains "[271C. section 271-CA]", a cross-reference, not
# section 271C. Treating it as a section both invents a bogus record and
# deletes the reference from its parent. A real section always reads
# "<number>. <Title ending in a full stop> <body>"; a substituted phrase does
# not, so requiring that shape tells them apart.
EMBEDDED_NUMBER_PATTERN = re.compile(r"^\d+[A-Za-z]{0,3}\.\s+(?P<rest>\S.+)$", re.S)
PERIOD_TITLE_SPLIT_PATTERN = re.compile(r"^(?P<title>\S.*?[.:])\s*(?:[-–—]\s*)?(?P<body>\S.+)$", re.S)
MIN_EMBEDDED_BODY_CHARS = 40
MAX_EMBEDDED_TITLE_CHARS = 150
# Some sections separate title from body with a bare spaced dash and no full
# stop ("41A. Notice of appearance before police officer - (1) ..."). Without
# this the whole section ends up in section_title, and since the title is
# repeated in the header of every chunk it then eats the embedding token
# budget and shatters the section into dozens of tiny chunks.
DASH_TITLE_SPLIT_PATTERN = re.compile(r"^(?P<title>\S.*?)\s+[-–—]\s+(?P<body>\S.+)$", re.S)
# A section is sometimes printed with its number OUTSIDE the bracket holding
# the rest of it ("292. [ Evidence of officers of the Mint. (1) ...]" in the
# CrPC). The same shape occurs constantly by coincidence where a year
# precedes a citation ("...Act, 1978 [Vide M.P. Act 21 of 1995...]"), so a
# candidate counts only if its number falls strictly between the section it
# sits in and the section that follows - see _promote_numbered_brackets.
NUMBER_BEFORE_BRACKET = re.compile(r"(?<![\d.A-Za-z])(?P<number>\d+[A-Z]{0,3})\.\s*\[")


def _embedded_title_candidates(inner: str) -> List[Tuple[str, str]]:
    """Ways to read "[N. Title - Body]" as a (title, body) pair, best first.

    A section separates its title from its body either with a spaced dash or
    with a full stop, and some use both. Trying the dash first keeps a title
    like "Notice of appearance before police officer" from running on into
    the body just because the body happens to contain the first full stop.
    """
    match = EMBEDDED_NUMBER_PATTERN.match(inner)
    if match is None:
        return []

    rest = match.group("rest")
    candidates: List[Tuple[str, str]] = []
    for pattern in (DASH_TITLE_SPLIT_PATTERN, PERIOD_TITLE_SPLIT_PATTERN):
        found = pattern.match(rest)
        if found is not None:
            candidates.append((_clean_title(found.group("title")), found.group("body").strip()))
    return candidates


def _looks_like_embedded_section(inner: str) -> bool:
    """Does this bracket hold a whole section, or just a substituted phrase?"""
    return any(len(body) >= MIN_EMBEDDED_BODY_CHARS for _title, body in _embedded_title_candidates(inner))


def _promote_numbered_brackets(text: str, parent_number: Any, next_number: Any) -> str:
    """Rewrite "N. [ Title. Body ]" into "[N. Title. Body ]".

    That puts the number inside the bracket, where the ordinary embedded
    section machinery can already see it.
    """
    parent_key = _number_sort_key(parent_number)
    next_key = _number_sort_key(next_number) if next_number is not None else None
    if parent_key is None or next_key is None:
        return text

    result = text
    for match in reversed(list(NUMBER_BEFORE_BRACKET.finditer(text))):
        key = _number_sort_key(match.group("number"))
        if key is None or not parent_key < key < next_key:
            continue
        result = f"{result[:match.start()]}[{match.group('number')}. {result[match.end():]}"
    return result


def _next_section_bracket(text: str, start: int) -> int:
    """Index of the next bracket that opens a section heading, else len(text)."""
    for index in range(start, len(text)):
        if text[index] == "[" and BRACKET_HEADING_START.match(text[index + 1 :]):
            return index
    return len(text)


def _find_bracketed_subsections(text: str) -> List[Tuple[int, int, str]]:
    """Find top-level "[N. Title - Body]" blocks embedded in `text`.

    Matches by counting bracket depth rather than a single regex, because the
    embedded section's own body can itself contain further bracketed
    citations (e.g. "[Substituted by Act 4 of 1988]") that must not be
    mistaken for the block's closing bracket.

    A source bracket is sometimes never closed at all (CrPC sections 41A and
    292, Motor Vehicles 182A and 192 are each published that way). Dropping
    those loses a whole real section, so an unclosed block that opens with a
    section heading runs to wherever the next such heading begins.
    """
    spans: List[Tuple[int, int, str]] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "[":
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if text[j] == "[":
                    depth += 1
                elif text[j] == "]":
                    depth -= 1
                j += 1
            if depth == 0:
                inner = text[i + 1 : j - 1]
                if BRACKET_HEADING_START.match(inner):
                    spans.append((i, j, inner))
                i = j
                continue
            if BRACKET_HEADING_START.match(text[i + 1 :]):
                end = _next_section_bracket(text, i + 1)
                spans.append((i, end, text[i + 1 : end].strip()))
                i = end
                continue
        i += 1
    return spans


def split_embedded_amendments(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pull "[N. Title - Body]" sections embedded in another section's text
    out into their own records.

    Returns a new list: every input section with any embedded amendments
    removed from its `text`, plus one new record per amendment actually
    recovered. A bracket whose content reduces to a bare repeal marker is
    dropped rather than kept as an empty record - that means the section was
    formally omitted, not that something failed to scrape.
    """
    result: List[Dict[str, Any]] = []
    for index, section in enumerate(sections):
        following = sections[index + 1]["section_number"] if index + 1 < len(sections) else None
        text = _promote_numbered_brackets(
            str(section.get("text", "")), section.get("section_number"), following
        )
        spans = _find_bracketed_subsections(text)
        if not spans:
            result.append(section)
            continue

        cleaned_text = text
        changed = False
        extracted: List[Dict[str, Any]] = []
        for start, end, inner in reversed(spans):
            number, title, body = _parse_section_heading(inner)
            recovered_text = body or title
            if REPEAL_MARKER_PATTERN.match(re.sub(r"[.]", "", recovered_text)):
                cleaned_text = cleaned_text[:start] + cleaned_text[end:]
                changed = True
                continue
            if not _looks_like_embedded_section(inner):
                continue
            if not body:
                # The heading split needs a ". -" separator, which not every
                # section is printed with. Fall back to the looser split that
                # already proved this is a section, so the title does not
                # swallow the whole body - but only when it yields a title of
                # plausible length, since a section printed with no usable
                # separator at all is better left whole than cut arbitrarily.
                for candidate_title, candidate_body in _embedded_title_candidates(inner):
                    if len(candidate_title) <= MAX_EMBEDDED_TITLE_CHARS and len(candidate_body) >= MIN_EMBEDDED_BODY_CHARS:
                        title, recovered_text = candidate_title, candidate_body
                        break
            title, amendment_note = _split_title_and_note(title)
            new_section = {
                k: v
                for k, v in section.items()
                if k not in ("section_number", "section_title", "amendment_note", "text")
            }
            new_section["section_number"] = number
            new_section["section_title"] = title
            new_section["amendment_note"] = amendment_note
            new_section["text"] = recovered_text
            extracted.append(new_section)
            cleaned_text = cleaned_text[:start] + cleaned_text[end:]
            changed = True

        if not changed:
            # Nothing was pulled out, so keep the original text rather than the
            # rewritten copy _promote_numbered_brackets may have produced.
            result.append(section)
            continue

        updated = dict(section)
        updated["text"] = _normalize_text(cleaned_text)
        result.append(updated)
        result.extend(reversed(extracted))
    return result


def _direct_text_length(tag: Any) -> int:
    direct_text = _normalize_text(" ".join(string for string in tag.find_all(string=True, recursive=False) if string.strip()))
    return len(direct_text)


def _describe_tag(tag: Any) -> str:
    classes = tag.get("class")
    class_text = " ".join(classes) if classes else ""
    tag_id = tag.get("id")
    extras = []
    if tag_id:
        extras.append(f"id={tag_id}")
    if class_text:
        extras.append(f"class={class_text}")
    extra_text = f" ({', '.join(extras)})" if extras else ""
    return f"<{tag.name}>{extra_text}"


def _last_descendant(tag: Any):
    last_tag = None
    for descendant in tag.find_all(True):
        last_tag = descendant
    return last_tag


def _nearest_section_id(tag: Any) -> str:
    for ancestor in [tag, *tag.parents]:
        if getattr(ancestor, "name", None) == "section" and ancestor.get("id"):
            return ancestor.get("id")
    return ""


def _find_section_2_node(soup: BeautifulSoup):
    for section in soup.select("section.akn-section"):
        heading = section.find("h3")
        heading_text = _normalize_text(heading.get_text(" ", strip=True)) if heading else ""
        section_id = section.get("id", "")

        if re.search(r"\bsection\s*2\b", heading_text, re.IGNORECASE):
            return section
        if re.search(r"\b2\b", heading_text) and "section" in heading_text.lower():
            return section
        if re.search(r"section[-_]?2", section_id, re.IGNORECASE):
            return section

    return None


def _find_clause_46_node(section: Any):
    return section.find(id="section_2.46")


def _extract_section_text(section: Any, heading: Any, inline_body: str = "", full_text: str = None) -> str:
    """Return the section's body text, with the heading stripped off the front.

    Most sections put their body in sibling content after the <h3> heading, so
    subtracting the heading text from the full node text leaves the body. Some
    sections (see TITLE_BODY_SPLIT_PATTERN) render the ENTIRE section inside
    the heading itself, with nothing left over outside it - subtracting the
    heading there would leave an empty string and silently drop the section, so
    `inline_body` (recovered from the heading by the caller) is used instead.

    `full_text` overrides the node's own text with the section's segment (see
    _segment_sections), which is what the scraper passes - a node's own text
    is neither complete nor exclusive on these pages.
    """
    section_text = full_text if full_text is not None else _normalize_text(section.get_text(" ", strip=True))
    if heading is None:
        return section_text

    heading_text = _normalize_text(heading.get_text(" ", strip=True))
    if section_text.startswith(heading_text):
        remainder = section_text[len(heading_text):].lstrip(" \t\n\r-–—:;.,")
        # A section can have BOTH a real inline body inside the heading itself
        # (see docstring above) AND further sibling content after the heading
        # (e.g. a bracketed amendment glued on afterwards - see
        # split_embedded_amendments). Dropping either one would silently
        # discard real text, so when both are present they are concatenated.
        if remainder and inline_body:
            return f"{inline_body} {remainder}"
        if remainder:
            return remainder
        if inline_body:
            return inline_body

    return section_text


def diagnose_consumer_protection_act_section_2(timeout: int = 60) -> None:
    """Print the HTML structure around Consumer Protection Act Section 2."""
    response = requests.get(
        SECTION_2_URL,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=timeout,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    section = _find_section_2_node(soup)
    if section is None:
        print("Section 2 akn-section not found.")
        return

    heading = section.find("h3")
    content = section.select_one("span.akn-content")
    clause_46 = _find_clause_46_node(section)
    current_stop = _last_descendant(content) if content is not None else None
    real_end = _last_descendant(clause_46) if clause_46 is not None else None

    print("Section 2 diagnostic for Consumer Protection Act")
    print(f"Section node: {_describe_tag(section)}")
    if heading is not None:
        print(f"Heading: {_normalize_text(heading.get_text(' ', strip=True))}")
    print(f"Direct text length: {_direct_text_length(section)}")

    print("Nested tags that may resemble boundaries:")
    seen = set()
    for tag in section.find_all(True):
        if tag is section:
            continue
        classes = tuple(tag.get("class") or [])
        if not classes and tag.name not in {"section", "article", "div", "p", "h3", "h4", "span", "ol", "ul", "li", "table", "tr", "td", "thead", "tbody"}:
            continue

        key = (tag.name, classes, tag.get("id"))
        if key in seen:
            continue
        seen.add(key)
        print(f"- {_describe_tag(tag)}")

    if content is None:
        print("Current extraction node: span.akn-content not found.")
    else:
        print(f"Current extraction node: {_describe_tag(content)}")
        print(f"Current extraction direct text length: {_direct_text_length(content)}")
        print(f"Current extraction contains clause 46: {bool(clause_46 and content in clause_46.parents)}")
        if current_stop is not None:
            print(f"Current extraction last descendant: {_describe_tag(current_stop)}")
            print(f"Current extraction last descendant section id: {_nearest_section_id(current_stop)}")

    if clause_46 is None:
        print("Clause 46 boundary: not found in the section HTML.")
    else:
        print(f"Clause 46 boundary node: {_describe_tag(clause_46)}")
        print(f"Clause 46 text: {_normalize_text(clause_46.get_text(' ', strip=True))}")
        if real_end is not None:
            print(f"Clause 46 last descendant: {_describe_tag(real_end)}")
            print(f"Clause 46 last descendant section id: {_nearest_section_id(real_end)}")
        if content is not None:
            print(f"Clause 46 is inside current extraction node: {content in clause_46.parents}")

    if content is not None and clause_46 is not None:
        if content in clause_46.parents:
            print("Extraction stop vs real end: clause 46 is inside the current extraction node, so the stop point is at the end of span.akn-content.")
        else:
            print("Extraction stop vs real end: clause 46 lies outside span.akn-content, so the current extraction stops before the actual Section 2 content ends.")


def parse_act_html(html: str, act_name: str, source_url: str, scraped_at: str = "") -> List[Dict[str, Any]]:
    """Parse a statute page's HTML into the project section schema."""
    soup = BeautifulSoup(html, "html.parser")
    act_title = _extract_act_title(soup, act_name)
    scraped_at = scraped_at or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    sections: List[Dict[str, Any]] = []
    previous_key = None

    for node, segment_text in _act_body_segments(_segment_sections(soup)):
        heading = node.find("h3")
        if heading is None:
            continue

        heading_text = _normalize_text(heading.get_text(" ", strip=True))
        number, title, inline_body = _assign_section_number(node, heading_text, previous_key)
        key = _number_sort_key(number)

        if not node.get("id") and sections:
            # An id-less node that does not advance the numbering is not a new
            # section. It is either the source re-rendering the section above
            # it, or the rest of that section's body (Income Tax section 127
            # keeps only its heading, and its entire body follows in a node of
            # its own). Re-renders repeat the heading; a continuation does not.
            if key is None or previous_key is None or key <= previous_key:
                if key is not None and previous_key is not None and key == previous_key:
                    continue
                sections[-1]["text"] = _normalize_text(f"{sections[-1]['text']} {segment_text}")
                continue

        section_text = _extract_section_text(node, heading, inline_body=inline_body, full_text=segment_text)
        if not section_text:
            continue

        if key is not None:
            previous_key = key

        title, amendment_note = _split_title_and_note(title)
        sections.append(
            {
                "act": act_title,
                "section_number": number,
                "section_title": title,
                "amendment_note": amendment_note,
                "text": section_text,
                "source_type": "statute",
                "source_url": source_url,
                "scraped_at": scraped_at,
            }
        )

    return split_embedded_amendments(sections)


def scrape_act(source_url: str, act_name: str, timeout: int = 60) -> List[Dict[str, Any]]:
    """Scrape a statute from Indian Kanoon into the project section schema."""
    response = requests.get(
        source_url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_act_html(response.text, act_name, source_url)