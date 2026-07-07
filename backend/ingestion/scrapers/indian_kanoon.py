"""Scrape Indian Kanoon statute pages into section dictionaries."""

from __future__ import annotations

import re
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup


SECTION_HEADING_PATTERN = re.compile(r"^(?P<number>\d+(?:-[A-Za-z]+)?[A-Za-z]?\.?)\s*(?P<title>.*)$")


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
        content = section.select_one("span.akn-content")
        if heading is None or content is None:
            continue

        section_number, section_title = _parse_section_heading(heading.get_text(" ", strip=True))
        section_text = _normalize_text(content.get_text(" ", strip=True))
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