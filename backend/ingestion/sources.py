"""Where every statute in the corpus is scraped from.

This is the single place the source URLs live. The first eight acts were
originally scraped without recording their origin, which later made them
impossible to re-scrape or verify; each document id below was recovered by
searching Indian Kanoon for the act's "Entire Act" page and confirming the
page's own <h1> matches `act_name`.

`act_name` is only a fallback - the scraper prefers the title printed on the
page itself - but it must stay correct, because it is what the corpus is keyed
by if a page ever fails to declare one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class StatuteSource:
    """One act: where it comes from and which file it lands in."""

    filename: str
    act_name: str
    document_id: str

    @property
    def url(self) -> str:
        return f"https://indiankanoon.org/doc/{self.document_id}/"


STATUTE_SOURCES: Tuple[StatuteSource, ...] = (
    StatuteSource("ipc_full.json", "The Indian Penal Code, 1860", "1569253"),
    StatuteSource("code_of_criminal_procedure_1973.json", "The Code of Criminal Procedure, 1973", "445276"),
    StatuteSource("constitution_of_india.json", "Constitution of India", "237570"),
    StatuteSource("the_income_tax_act_1961.json", "The Income Tax Act, 1961", "789969"),
    StatuteSource("the_indian_evidence_act_1872.json", "The Indian Evidence Act, 1872", "1953529"),
    StatuteSource("the_information_technology_act_2000.json", "The Information Technology Act, 2000", "1965344"),
    StatuteSource("consumer_protection_act_2019.json", "Consumer Protection Act, 2019", "48103131"),
    StatuteSource("the_motor_vehicles_act_1988.json", "The Motor Vehicles Act, 1988", "785258"),
)


def find_source(name: str) -> StatuteSource:
    """Look an act up by filename, by its stem, or by act name."""
    wanted = name.strip().lower().removesuffix(".json")
    for source in STATUTE_SOURCES:
        if wanted in (source.filename.lower().removesuffix(".json"), source.act_name.lower()):
            return source
    known = ", ".join(s.filename.removesuffix(".json") for s in STATUTE_SOURCES)
    raise KeyError(f"No statute source named {name!r}. Known sources: {known}")
