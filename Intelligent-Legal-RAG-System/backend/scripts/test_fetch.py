import json
import re
from pathlib import Path

import requests

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

headers = {"User-Agent": "Mozilla/5.0"}

try:
    indian_kanoon_response = requests.get(
        "https://indiankanoon.org/doc/1569253/",
        headers=headers,
        timeout=20,
    )
    print("URL 1: https://indiankanoon.org/doc/1569253/")
    print("status_code:", indian_kanoon_response.status_code)

    if BeautifulSoup is None:
        print("BeautifulSoup not available")
        raise SystemExit

    soup = BeautifulSoup(indian_kanoon_response.text, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    print("title:", title)

    sections = []
    section_nodes = soup.find_all("section", class_="akn-section")
    method_used = "akn-section"

    if not section_nodes:
        method_used = "h3/h4 fallback"
        heading_tags = soup.find_all(["h3", "h4"])
        for heading in heading_tags:
            heading_text = " ".join(heading.get_text(" ", strip=True).split())
            match = re.match(r"^\s*(\d{1,4}[A-Za-z]?)(?:\.\s*|\s+)(.+)$", heading_text)
            if not match:
                continue
            section_number = match.group(1)
            section_title = re.sub(r"[\s\W]+$", "", match.group(2)).strip()
            body_parts = []
            for sibling in heading.next_siblings:
                if getattr(sibling, "name", None) in {"h3", "h4"}:
                    break
                if getattr(sibling, "string", None):
                    body_parts.append(sibling.string)
                elif getattr(sibling, "get_text", None):
                    text = " ".join(sibling.get_text(" ", strip=True).split())
                    if text:
                        body_parts.append(text)
            body_text = " ".join(part for part in body_parts if part).strip()
            if section_number and section_title:
                sections.append((section_number, section_title, body_text))
    else:
        for section in section_nodes:
            heading = section.find(["h3", "h4", "h5"])
            if not heading:
                continue
            heading_text = " ".join(heading.get_text(" ", strip=True).split())
            match = re.match(r"^\s*(\d{1,4}[A-Za-z]?)(?:\.\s*|\s+)(.+)$", heading_text)
            if not match:
                continue
            section_number = match.group(1)
            section_title = re.sub(r"[\s\W]+$", "", match.group(2)).strip()
            full_text = " ".join(section.get_text(" ", strip=True).split())
            body_text = full_text
            if heading_text and full_text.startswith(heading_text):
                body_text = full_text[len(heading_text):].strip()
            else:
                body_text = full_text.replace(heading_text, "", 1).strip()
            body_text = " ".join(body_text.split())
            if section_number and section_title:
                sections.append((section_number, section_title, body_text))

    clean_sections = []
    for section_number, section_title, section_text in sections:
        clean_text = section_text.split("[[", 1)[0].strip()
        if len(clean_text) < 10:
            continue
        clean_sections.append(
            {
                "act": "Indian Penal Code, 1860",
                "section_number": section_number,
                "section_title": section_title,
                "text": clean_text,
                "source_type": "statute",
            }
        )

    output_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "ipc_full.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(clean_sections, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("method_used:", method_used)
    print("total_extracted_sections:", len(sections))
    print("sections_saved:", len(clean_sections))

    for entry in clean_sections:
        if entry["section_number"] == "303":
            print("SECTION 303 saved entry:")
            print(json.dumps(entry, indent=2, ensure_ascii=False))
            break
except Exception as e:
    print(f"Request failed for URL 1: {e}")
    print("---")

# --- India Code section — paused for now, resume once Indian Kanoon scraping is done ---
# session = requests.Session()
# session.get("https://www.indiacode.nic.in/", headers=headers, timeout=20)
# ... (rest of the block)
