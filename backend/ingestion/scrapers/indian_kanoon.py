"""Scrape Indian Kanoon statute pages into section dictionaries."""

from __future__ import annotations

import re
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup


SECTION_HEADING_PATTERN = re.compile(r"^(?P<number>\d+(?:-[A-Za-z]+)?[A-Za-z]?\.?)\s*(?P<title>.*)$")
SECTION_2_URL = "https://indiankanoon.org/doc/48103131/"


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _extract_act_title(soup: BeautifulSoup, fallback: str) -> str:
    for heading in soup.find_all("h1"):
        text = _normalize_text(heading.get_text(" ", strip=True))
        if text and text != "Indian Kanoon - Search engine for Indian Law":
            return text
    return fallback


def _parse_section_heading(heading_text: str) -> tuple[str, str]:
    cleaned = _normalize_text(heading_text)
    match = SECTION_HEADING_PATTERN.match(cleaned)
    if not match:
        return cleaned, cleaned

    section_number = match.group("number").rstrip(".")
    section_title = match.group("title").strip()
    return section_number, section_title


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


def _extract_section_text(section: Any, heading: Any) -> str:
    section_text = _normalize_text(section.get_text(" ", strip=True))
    if heading is None:
        return section_text

    heading_text = _normalize_text(heading.get_text(" ", strip=True))
    if section_text.startswith(heading_text):
        section_text = section_text[len(heading_text):].lstrip(" \t\n\r-–—:;.,")

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


def scrape_act(source_url: str, act_name: str, timeout: int = 60) -> List[Dict[str, Any]]:
    """Scrape a statute from Indian Kanoon into the project section schema."""
    response = requests.get(
        source_url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=timeout,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    act_title = _extract_act_title(soup, act_name)
    sections: List[Dict[str, Any]] = []

    for section in soup.select("section.akn-section"):
        heading = section.find("h3")
        if heading is None:
            continue

        section_number, section_title = _parse_section_heading(heading.get_text(" ", strip=True))
        section_text = _extract_section_text(section, heading)
        if not section_text:
            continue

        sections.append(
            {
                "act": act_title,
                "section_number": section_number,
                "section_title": section_title,
                "text": section_text,
                "source_type": "statute",
            }
        )

    return sections