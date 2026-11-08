"""Claim extraction from fetched pages, followed by mandatory quote verification.

Two extractors:
  - RuleExtractor (no LLM): picks sentences that share keywords with the research
    question; sentences with numbers are preferred. The claim is literally what the
    source says, so it can be a FEIT once verified; relevance is judged later.
  - LLMExtractor: the model proposes claims with verbatim quotes; every quote is
    verified against the page and unverifiable quotes are dropped (and counted).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from kansenradar.domain import Evidence, EvidenceType, ResearchTopic, SourceQuality
from kansenradar.history.embeddings import tokens
from kansenradar.research.fetch import FetchedPage
from kansenradar.research.verify import verify_quote

log = logging.getLogger(__name__)

MAX_PAGE_CHARS = 5_000  # fewer tokens per call: fits free-tier limits (~1.700 tokens)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class ExtractionResult:
    evidence: list[Evidence] = field(default_factory=list)
    rejected_quotes: int = 0


def classify_source(url: str) -> SourceQuality:
    host = re.sub(r"^www\.", "", url.split("/")[2].lower()) if "//" in url else ""
    if host.endswith((".gov", ".europa.eu", "fgov.be", "vlaanderen.be", "overheid.nl", "cbs.nl",
                      "wodc.nl", "rechtspraak.nl", "advocatenorde.nl", "ordevanvlaamsebalies.be", "nbb.be",
                      "rijksoverheid.nl", "kvk.nl", "statbel.be", "belgium.be", "wallonie.be", "brussels")):
        return SourceQuality.OFFICIAL
    if any(k in host for k in ("reddit.", "forum", "trustpilot", "facebook.", "quora.")):
        return SourceQuality.COMMUNITY
    if any(k in host for k in ("nieuws", "news", "tijd.be", "standaard.be", "nos.nl", "fd.nl",
                               "nrc.nl", "vrt.be", "hln.be", "demorgen.be", "volkskrant.nl")):
        return SourceQuality.MEDIA
    if any(k in host for k in ("federatie", "unizo", "voka", "mkb.nl", "bouwunie", "confederatie",
                               "branche", "vereniging")):
        return SourceQuality.INDUSTRY
    return SourceQuality.COMPANY


def _fact(claim: str, quote: str, topic: ResearchTopic, page: FetchedPage) -> Evidence:
    return Evidence.create(
        claim=claim, type=EvidenceType.FACT, topic=topic.value, source_url=page.url,
        source_title=page.title or None, source_quality=classify_source(page.url), quote=quote,
        verified=True, retrieved_at=page.fetched_at, published_at=page.published_at,
    )


class RuleExtractor:
    name = "regels"

    def __init__(self, per_page: int = 3) -> None:
        self.per_page = per_page

    def extract(self, page: FetchedPage, topic: ResearchTopic, question: str) -> ExtractionResult:
        q_tokens = set(tokens(question))
        scored = []
        for sentence in _SENTENCE.split(page.text[:MAX_PAGE_CHARS * 3]):
            sentence = sentence.strip()
            if not 60 <= len(sentence) <= 400:
                continue
            # Headings, questions and fragments are not evidence: require a complete statement.
            if not sentence.endswith((".", "!")) or sentence.endswith("?"):
                continue
            overlap = len(q_tokens & set(tokens(sentence)))
            has_number = bool(re.search(r"\d", sentence))
            if overlap < 2 or (not has_number and overlap < 3):
                continue
            scored.append((overlap + (1.5 if re.search(r"\d", sentence) else 0), sentence))
        scored.sort(key=lambda t: t[0], reverse=True)
        result = ExtractionResult()
        for _, sentence in scored[: self.per_page]:
            check = verify_quote(sentence, page.text)  # sanity check of our own extraction
            if not check.found:
                result.rejected_quotes += 1
                continue
            result.evidence.append(_fact(f"Bron vermeldt: “{sentence}”", sentence, topic, page))
        return result


class _LLMClaim(BaseModel):
    claim: str = Field(description="De bewering in eigen woorden, in het Nederlands.")
    quote: str = Field(description="LETTERLIJK citaat uit de tekst dat de bewering staaft (geen parafrase).")
    relevant: bool = Field(description="Beantwoordt dit (deels) de onderzoeksvraag?")


class _LLMClaims(BaseModel):
    claims: list[_LLMClaim]


EXTRACT_INSTRUCTIONS = """\
Taak: haal uit de brontekst alleen beweringen die de onderzoeksvraag helpen beantwoorden.
- Elk citaat moet LETTERLIJK in de tekst staan (kopieer, vertaal niet, parafraseer niet).
- Citaten worden automatisch gecontroleerd; niet-letterlijke citaten worden verworpen.
- Geen beweringen uit eigen kennis. Niets relevants gevonden: geef een lege lijst.
- Maximaal 4 beweringen."""


class LLMExtractor:
    name = "llm"

    def __init__(self, llm, fallback: RuleExtractor | None = None) -> None:
        self.llm = llm
        self.fallback = fallback or RuleExtractor()

    def extract(self, page: FetchedPage, topic: ResearchTopic, question: str) -> ExtractionResult:
        from kansenradar.llm import Tier

        # Skip pages that do not even mention the subject: saves a (slow, rate-limited) LLM call.
        if len(set(tokens(question)) & set(tokens(page.text[:MAX_PAGE_CHARS]))) < 2:
            return ExtractionResult()
        prompt = (f"Onderzoeksvraag ({topic.value}): {question}\n\nBron: {page.title} ({page.url})\n"
                  f"Brontekst:\n{page.text[:MAX_PAGE_CHARS]}")
        try:
            out = self.llm.structured(task="extract_claims", tier=Tier.EXTRACTION, prompt=prompt,
                                      schema=_LLMClaims, system_extra=EXTRACT_INSTRUCTIONS,
                                      max_tokens=2_000)
        except Exception as exc:
            log.warning("LLM-extractie mislukt (%s); regels gebruikt.", exc)
            return self.fallback.extract(page, topic, question)
        result = ExtractionResult()
        for c in out.claims:
            if not c.relevant:
                continue
            check = verify_quote(c.quote, page.text)
            if not check.found:
                result.rejected_quotes += 1
                log.info("Citaat verworpen (%s): %s", check.reason, c.quote[:80])
                continue
            result.evidence.append(_fact(c.claim, c.quote, topic, page))
        return result
