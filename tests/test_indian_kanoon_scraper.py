"""Regression tests for the Indian Kanoon scraper's amendment-extraction fix.

Covers the Motor Vehicles Act bug: some sections inserted by a later
amendment get no DOM node of their own on the source page - they are glued,
still wrapped in "[N. Title - Body]", onto the end of whichever section
happened to precede them. Left unfixed, that section is invisible to search
and its text pollutes its neighbour's.
"""

from bs4 import BeautifulSoup

from backend.ingestion.scrapers.indian_kanoon import (
    _extract_section_text,
    _parse_section_heading,
    parse_act_html,
    split_embedded_amendments,
)


def _section(text: str, **overrides):
    base = {
        "act": "The Motor Vehicles Act, 1988",
        "section_number": "128",
        "section_title": "Safety measures for drivers and pillion riders.",
        "text": text,
        "source_type": "statute",
        "source_url": "https://indiankanoon.org/doc/785258/",
        "scraped_at": "2026-09-18",
    }
    base.update(overrides)
    return base


def test_embedded_section_is_split_into_its_own_record() -> None:
    parent = _section(
        "In addition to the safety measures mentioned in sub-section (1), the "
        "Central Government may prescribe other measures. "
        "[129. Wearing of protective headgear. - Every person shall wear a helmet: "
        "Provided that a Sikh wearing a turban is exempt.]"
    )

    result = split_embedded_amendments([parent])

    numbers = [s["section_number"] for s in result]
    assert numbers == ["128", "129"]

    recovered = result[1]
    assert recovered["section_title"] == "Wearing of protective headgear."
    assert recovered["text"].startswith("Every person shall wear a helmet")
    assert recovered["act"] == parent["act"]
    assert recovered["source_url"] == parent["source_url"]


def test_embedded_text_is_removed_from_the_parent() -> None:
    parent = _section(
        "Real section 128 content. [129. Wearing of protective headgear. - Every person "
        "driving or riding on a motorcycle shall wear protective headgear.]"
    )

    result = split_embedded_amendments([parent])

    assert "129" not in result[0]["text"]
    assert "protective headgear" not in result[0]["text"]
    assert "Real section 128 content." in result[0]["text"]


def test_multiple_embedded_sections_in_one_parent_are_all_recovered() -> None:
    parent = _section(
        "diplomatic officer means... "
        "[43. Temporary registration. - The owner of a motor vehicle may apply for the "
        "temporary registration of that vehicle.] "
        "[44. Production of vehicle at the time of registration. - The registering "
        "authority shall require the vehicle to be produced before it.]",
        section_number="42",
    )

    result = split_embedded_amendments([parent])

    numbers = [s["section_number"] for s in result]
    assert numbers == ["42", "43", "44"]
    assert result[1]["text"].startswith("The owner of a motor vehicle may apply")
    assert result[2]["text"].startswith("The registering authority shall require")


def test_bare_repeal_marker_is_dropped_not_kept_as_an_empty_section() -> None:
    parent = _section("Some real content here. [191. ***]")

    result = split_embedded_amendments([parent])

    assert len(result) == 1
    assert "191" not in result[0]["text"]


def test_ordinary_citation_bracket_is_left_untouched() -> None:
    """A bracket that doesn't start with a section number is not a section -
    e.g. "[Substituted by Act 4 of 1988]" or "[Explanation. - ...]"."""
    parent = _section(
        "Some content. [Explanation. - This clarifies section 56.] More content."
    )

    result = split_embedded_amendments([parent])

    assert len(result) == 1
    assert "[Explanation. - This clarifies section 56.]" in result[0]["text"]


def test_substituted_phrase_in_brackets_is_not_mistaken_for_a_section() -> None:
    """An amendment can substitute a PHRASE that itself starts with a section
    number - Income Tax section 246A contains "[271C. section 271-CA]", a
    cross-reference. Splitting it out would invent a bogus section 271C and
    delete the reference from section 246A."""
    parent = _section(
        "an order imposing a penalty under section [271C. section 271-CA] , section "
        "271-D or section 271-E; and any other order of that kind.",
        section_number="246A",
    )

    result = split_embedded_amendments([parent])

    assert len(result) == 1
    assert "[271C. section 271-CA]" in result[0]["text"]


def test_section_number_printed_outside_its_bracket_is_recovered() -> None:
    """CrPC section 292 is printed as "292. [ Evidence ... ]" - the number sits
    outside the bracket. It counts only because 292 falls between the section
    it sits in (291A) and the one that follows (293)."""
    parent = _section(
        "Identification report of a Magistrate shall be admissible. "
        "292. [ Evidence of officers of the Mint. (1) Any document purporting to be a "
        "report under the hand of an officer of any Mint shall be admissible.]",
        section_number="291A",
    )
    following = _section("Reports of certain Government scientific experts.", section_number="293")

    result = split_embedded_amendments([parent, following])

    numbers = [s["section_number"] for s in result]
    assert numbers == ["291A", "292", "293"]
    recovered = [s for s in result if s["section_number"] == "292"][0]
    assert recovered["section_title"] == "Evidence of officers of the Mint."
    assert recovered["text"].startswith("(1) Any document purporting")
    assert "Evidence of officers" not in result[0]["text"]


def test_a_year_before_a_citation_is_not_treated_as_a_section_number() -> None:
    """"...Act, 1978 [Vide M.P. Act 21 of 1995...]" has the same shape as a
    number printed outside its bracket. 1978 is not between 24 and 25, so it
    must be left alone."""
    parent = _section(
        "substituted by the M.P. Act of 1978 [Vide M.P. Act 21 of 1995, Section 3, "
        "with effect from the twenty-fourth day of January 1995.]",
        section_number="24",
    )
    following = _section("Executive Magistrates.", section_number="25")

    result = split_embedded_amendments([parent, following])

    assert [s["section_number"] for s in result] == ["24", "25"]
    assert "Vide M.P. Act 21 of 1995" in result[0]["text"]


def test_unclosed_section_bracket_runs_to_the_next_section_heading() -> None:
    """Motor Vehicles section 182's node holds "[182A. ..." with no closing
    bracket, followed by a properly closed "[182B. ...]". 182A must stop where
    182B begins rather than swallowing it."""
    parent = _section(
        "Offences relating to licences are punishable. "
        "[182A. Punishment for offences relating to construction. - Whoever, being a "
        "manufacturer of motor vehicles, sells a vehicle that is not compliant. "
        "[182B. Punishment for contravention of section 62A. - Whoever contravenes "
        "the provisions of section 62A shall be punishable with a fine.]",
        section_number="182",
    )

    result = split_embedded_amendments([parent])

    assert [s["section_number"] for s in result] == ["182", "182A", "182B"]
    assert "182B" not in result[1]["text"]
    assert result[2]["text"].startswith("Whoever contravenes the provisions of section 62A")


def test_title_separated_by_a_bare_dash_does_not_swallow_the_whole_section() -> None:
    """CrPC 41A is printed as "Title - (1) body" with no full stop before the
    dash. Left unsplit the entire section lands in section_title, and because
    the title is repeated in every chunk's header it then eats the embedding
    token budget and shatters the section into dozens of tiny chunks."""
    parent = _section(
        "A police officer may arrest without a warrant. "
        "[41A. Notice of appearance before police officer - (1) The police officer "
        "shall, in all cases where the arrest of a person is not required, issue a "
        "notice directing the person to appear before him.]",
        section_number="41",
    )

    result = split_embedded_amendments([parent])

    recovered = result[1]
    assert recovered["section_number"] == "41A"
    assert recovered["section_title"] == "Notice of appearance before police officer"
    assert recovered["text"].startswith("(1) The police officer shall")


def test_nested_bracket_inside_an_embedded_section_does_not_break_extraction() -> None:
    """The embedded section's own body may itself contain a bracketed
    citation - the split must find the TRUE closing bracket, not the first one."""
    parent = _section(
        "Preceding text. [62A. Prohibition of registration. - No authority shall "
        "register any vehicle [as defined in section 2] that contravenes this rule.]"
    )

    result = split_embedded_amendments([parent])

    assert len(result) == 2
    assert result[1]["section_number"] == "62A"
    assert result[1]["text"] == "No authority shall register any vehicle [as defined in section 2] that contravenes this rule."


def test_range_repeal_marker_is_not_mistaken_for_a_section() -> None:
    """"[140 to 144. ***]" is a block-repeal marker, not a single numbered
    section - BRACKET_HEADING_START must not match "140 to 144."."""
    parent = _section("Preceding text. [140 to 144. ***] Following text.")

    result = split_embedded_amendments([parent])

    assert len(result) == 1
    assert "[140 to 144. ***]" in result[0]["text"]


def test_section_with_no_brackets_is_returned_unchanged() -> None:
    parent = _section("Perfectly ordinary section text with no amendments.")

    result = split_embedded_amendments([parent])

    assert result == [parent]


def test_heading_inline_body_is_kept_when_sibling_content_also_follows() -> None:
    """A section can have its real body inside the <h3> heading itself (e.g.
    "Title. - Body...") AND a bracketed amendment as sibling content right
    after it (e.g. Motor Vehicles Act section 62, followed by 62A/62B/63).
    Both pieces of real text must survive - neither is junk to be dropped."""
    html = (
        '<section class="akn-section" id="section_62">'
        "<h3>62. Some heading. - The real body of section 62 itself.</h3>"
        "<span>[62A. Inserted section. - Body of 62A.]</span>"
        "</section>"
    )
    soup = BeautifulSoup(html, "html.parser")
    section = soup.find("section")
    heading = section.find("h3")
    _, _, inline_body = _parse_section_heading(heading.get_text(" ", strip=True))

    text = _extract_section_text(section, heading, inline_body=inline_body)

    assert "The real body of section 62 itself." in text
    assert "[62A. Inserted section. - Body of 62A.]" in text


def _page(body: str) -> str:
    return f"<h1>The Test Act, 1999</h1>{body}"


def _parse(body: str):
    return parse_act_html(_page(body), "The Test Act, 1999", "https://example.invalid/", "2026-09-18")


def test_section_number_comes_from_the_node_id_not_the_heading() -> None:
    """The source splits a multi-letter suffix in the heading text: 376DA
    renders as "376D" + "A. ...". Trusting the heading merges 376D, 376DA and
    376DB into one number; the id keeps them apart."""
    records = _parse(
        '<section class="akn-section" id="section_376D"><h3>376D. Gang rape</h3>'
        "<p>Where a woman is raped by one or more persons.</p></section>"
        '<section class="akn-section" id="section_376DA"><h3>376DA. Punishment for gang '
        "rape on woman under sixteen years of age.</h3><p>Where a woman under sixteen "
        "years of age is raped.</p></section>"
    )

    assert [r["section_number"] for r in records] == ["376D", "376DA"]
    assert records[1]["section_title"].startswith("Punishment for gang rape")


def test_schedule_content_after_the_numbering_restarts_is_excluded() -> None:
    """Schedules use the same markup as sections and are not always labelled.
    Income Tax runs 1..298 then restarts at 19 for the Second Schedule."""
    records = _parse(
        '<section class="akn-section" id="section_297"><h3>297. Repeals and savings.</h3>'
        "<p>The repealed Act shall continue to apply.</p></section>"
        '<section class="akn-section" id="section_298"><h3>298. Power to remove difficulties.</h3>'
        "<p>The Central Government may remove difficulties.</p></section>"
        '<section class="akn-section" id="section_19"><h3>19. Assistance by police.</h3>'
        "<p>Any officer authorised to attach property may apply for police help.</p></section>"
    )

    assert [r["section_number"] for r in records] == ["297", "298"]


def test_section_nested_inside_another_section_gets_its_own_record() -> None:
    """Constitution article 334A sits INSIDE article 334's node with no id of
    its own, so 334 must not swallow it."""
    records = _parse(
        '<section class="akn-section" id="section_334"><h3>334. Reservation of seats to cease.</h3>'
        "<p>Reservation shall cease after seventy years.</p>"
        '<section class="akn-section"><h3>334A. Reservation of seats for women take effect.</h3></section>'
        "<p>Seats reserved for women shall take effect after delimitation.</p>"
        "</section>"
        '<section class="akn-section" id="section_335"><h3>335. Claims to services and posts.</h3>'
        "<p>The claims of the Scheduled Castes shall be taken into consideration.</p></section>"
    )

    assert [r["section_number"] for r in records] == ["334", "334A", "335"]
    assert "women" not in records[0]["text"]
    assert "delimitation" in records[1]["text"]
    assert "delimitation" not in records[2]["text"]


def test_body_that_spills_outside_the_section_node_is_kept() -> None:
    """Constitution article 332A keeps only clause 1 inside its node; clauses
    2 and 3 follow it as siblings before the next article begins."""
    records = _parse(
        '<section class="akn-section" id="section_332A"><h3>332A. Reservation of seats for women.</h3>'
        "<p>1 Seats shall be reserved for women in every State.</p></section>"
        '<section class="akn-subsection"><p>2 One-third of the seats shall be so reserved.</p></section>'
        '<section class="akn-section" id="section_333"><h3>333. Representation of the community.</h3>'
        "<p>The Governor may nominate one member.</p></section>"
    )

    assert [r["section_number"] for r in records] == ["332A", "333"]
    assert "One-third of the seats" in records[0]["text"]
    assert "One-third of the seats" not in records[1]["text"]


def test_id_less_node_that_does_not_advance_numbering_is_a_continuation() -> None:
    """Income Tax section 127 keeps only its heading; its whole body follows
    in a node of its own headed "1." - which must not become section 1."""
    records = _parse(
        '<section class="akn-section" id="section_127"><h3>127. Powers to transfer cases.</h3></section>'
        '<section class="akn-section"><h3>1.</h3><p>) The Commissioner may transfer any '
        "case from one Assessing Officer to another.</p></section>"
        '<section class="akn-section" id="section_128"><h3>128. Functions of Inspectors.</h3>'
        "<p>Inspectors shall perform such functions as are assigned.</p></section>"
    )

    assert [r["section_number"] for r in records] == ["127", "128"]
    assert "Commissioner may transfer any case" in records[0]["text"]


def test_trailing_separator_dash_is_stripped_from_the_title() -> None:
    """IPC headings read "Punishment for murder.-" with the body elsewhere on
    the page. The dash separates title from body, so leaving it on the title
    puts it into every citation shown to the user."""
    _number, title, _body = _parse_section_heading("302. Punishment for murder.—")
    assert title == "Punishment for murder."

    records = _parse(
        '<section class="akn-section" id="section_302"><h3>302. Punishment for murder.—</h3>'
        "<p>Whoever commits murder shall be punished with death.</p></section>"
    )
    assert records[0]["section_title"] == "Punishment for murder."
    assert records[0]["text"] == "Whoever commits murder shall be punished with death."


def test_zero_width_characters_are_removed() -> None:
    """Invisible characters survive normal whitespace stripping and would end
    up inside words, where they break keyword matching."""
    records = _parse(
        '<section class="akn-section" id="section_5"><h3>5. Saving​ of certain laws.</h3>'
        "<p>Nothing​ in this Act shall affect any special law.</p></section>"
    )
    assert "​" not in records[0]["section_title"]
    assert "​" not in records[0]["text"]
    assert records[0]["section_title"] == "Saving of certain laws."


def test_amendment_citation_moves_out_of_the_title_into_its_own_field() -> None:
    """CrPC 24's heading is "[ Public Prosecutors. [Substituted by Act 45 of
    1978 ...]". Only the first part names the section; the citation is
    provenance the source records nowhere else, so it is kept, not dropped."""
    records = _parse(
        '<section class="akn-section" id="section_24"><h3>24. [ Public Prosecutors. '
        "[Substituted by Act 45 of 1978, Section 8, for Section 24, w.e.f. 18.12.1978.]</h3>"
        "<p>For every High Court the Central Government shall appoint a Public Prosecutor.</p>"
        "</section>"
    )

    assert records[0]["section_title"] == "Public Prosecutors."
    assert records[0]["amendment_note"].startswith("[Substituted by Act 45 of 1978")
    assert records[0]["text"].startswith("For every High Court")


def test_a_title_that_is_only_a_marker_is_left_alone() -> None:
    """"[Omitted]" IS the section's name - splitting the marker off would
    leave the section with no title at all."""
    records = _parse(
        '<section class="akn-section" id="section_280A"><h3>280A. [Omitted]</h3>'
        "<p>Omitted by the Finance Act, 1990 with effect from 1 April 1990.</p></section>"
    )

    assert records[0]["section_title"] == "[Omitted]"
    assert records[0]["amendment_note"] == ""


def test_parse_section_heading_still_splits_title_and_body() -> None:
    number, title, body = _parse_section_heading(
        "26. Maintenance of State Registers of Driving Licences. - Each State "
        "Government shall maintain a register."
    )
    assert number == "26"
    assert title == "Maintenance of State Registers of Driving Licences."
    assert body == "Each State Government shall maintain a register."
