"""Competitor discovery and analysis.

Candidates come from targeted web searches and from company sources found during
market research. Media, forums, review sites and government sites are sources,
not competitors. Prices are only taken from the competitor's own pages and are
recorded as FEIT with quote. Without web search the saturation is 'onbekend':
no information is not evidence of an empty market.
"""

from __future__ import annotations

import logging
import re
import statistics
from datetime import UTC, datetime
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from kansenradar.domain import (
    CompetitionAnalysis,
    Competitor,
    Evidence,
    EvidenceType,
    Idea,
    ResearchTopic,
    SourceQuality,
)
from kansenradar.competition.prices import extract_prices
from kansenradar.history.embeddings import tokens
from kansenradar.research.extract import classify_source
from kansenradar.research.fetch import FetchSkipped, PageFetcher
from kansenradar.research.researcher import key_terms, short_target
from kansenradar.research.search import NoSearch, SearchProvider
from kansenradar.research.verify import verify_quote

log = logging.getLogger(__name__)

QUERIES = (
    "{kern} software {target}",
    "alternatief {kern} {target}",
    "{kern} app prijs",
    "beste {kern} tool België",
    "{kern} tool Nederland",
)
NOT_COMPETITORS = (
    "wikipedia.", "youtube.", "linkedin.", "facebook.", "instagram.", "x.com", "twitter.",
    "reddit.", "capterra.", "g2.com", "trustpilot.", "getapp.", "softwareadvice.", "medium.com",
    "google.", "apple.com", "play.google", "amazon.", "bol.com", "indeed.", "glassdoor.",
)
# Pages that are articles about a market, not a competitor's own offering.
ARTICLE_URL = re.compile(r"/(blog|blogs|nieuws|news|artikel|artikelen|article|articles|actueel|kennis|"
                         r"insights|report|rapport|trends|magazine|whitepaper|vacature|events?)(/|$)|/20\d\d/", re.I)
ARTICLE_TITLE = re.compile(r"\b20\d\d\b|\?|\bbeste\b|\btop ?\d|\boverzicht\b|\btrends?\b|"
                           r"\bgroeikansen\b|\bvoorspelling|\breport\b|\brapport\b|\bhoe\b|\bwaarom\b|"
                           r"\btips\b|\bgids\b|\bvergelijking\b", re.I)
GENERIC_TITLES = {"title", "home", "homepage", "welkom", "index"}
PRICE_PATHS = ("/prijzen", "/pricing", "/tarieven", "/prijs")
MAX_CANDIDATES = 8
MIN_OVERLAP = 2


def _domain(url: str) -> str:
    host = urlsplit(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _is_candidate(url: str) -> bool:
    d = _domain(url)
    if not d or any(x in d for x in NOT_COMPETITORS) or ARTICLE_URL.search(urlsplit(url).path):
        return False
    return classify_source(url) is SourceQuality.COMPANY


class _LLMCompetitor(BaseModel):
    index: int
    kind: str = Field(description="direct | indirect | geen")
    description: str = Field(description="Wat biedt dit bedrijf aan, in 1 zin (op basis van de tekst).")
    strengths: list[str]
    weaknesses: list[str]


class _LLMCompetitors(BaseModel):
    competitors: list[_LLMCompetitor]
    gaps: list[str] = Field(description="Mogelijke gaten in de markt die het idee kan benutten (hypotheses).")


class _LLMKnown(BaseModel):
    name: str
    domain: str
    description: str
    kind: str = Field(description="direct | indirect")


class _LLMKnownList(BaseModel):
    competitors: list[_LLMKnown]


CLASSIFY_INSTRUCTIONS = """\
Taak: beoordeel per kandidaat of het een concurrent is van het idee, uitsluitend op basis van \
de gegeven paginatekst. direct = lost hetzelfde probleem op voor dezelfde doelgroep; \
indirect = alternatief dat klanten nu gebruiken (bv. generieke tool); geen = niet relevant. \
Wees kritisch en concreet. Gaten in de markt zijn hypotheses."""

KNOWN_INSTRUCTIONS = """\
Taak: noem bestaande concurrenten van dit idee in België, Nederland of internationaal, \
UITSLUITEND als je zeker bent dat ze bestaan. Liever een korte of lege lijst dan gokken. \
Deze lijst wordt als te verifiëren hypothese behandeld."""


class CompetitorAnalyzer:
    def __init__(self, search: SearchProvider | None = None, fetcher: PageFetcher | None = None,
                 llm=None) -> None:
        self.search = search or NoSearch()
        self.fetcher = fetcher or PageFetcher()
        self.llm = llm

    # --- discovery ----------------------------------------------------------
    def _candidate_urls(self, idea: Idea, prior: list[Evidence]) -> list[str]:
        fmt = {"kern": key_terms(idea), "target": short_target(idea)}
        urls: list[str] = []
        for q in QUERIES:
            urls += [h.url for h in self.search.search(q.format(**fmt), max_results=5)]
        urls += [e.source_url for e in prior
                 if e.source_url and e.source_quality is SourceQuality.COMPANY]
        seen, out = set(), []
        for u in urls:
            d = _domain(u)
            if _is_candidate(u) and d not in seen:
                seen.add(d)
                out.append(u)
        return out[:MAX_CANDIDATES]

    def _inspect(self, url: str, idea: Idea) -> tuple[Competitor, str] | None:
        try:
            page = self.fetcher.fetch(url)
        except FetchSkipped as exc:
            log.info("Kandidaat overgeslagen %s: %s", url, exc)
            return None
        if ARTICLE_URL.search(urlsplit(page.url).path) or ARTICLE_TITLE.search(page.title or ""):
            log.info("Geen concurrent (artikel of overzicht): %s", page.url)
            return None
        parts = urlsplit(page.url)
        prices = extract_prices(page.text, page.url)
        if not prices:
            for path in PRICE_PATHS:
                try:
                    pp = self.fetcher.fetch(f"{parts.scheme}://{parts.netloc}{path}")
                except FetchSkipped:
                    continue
                prices = extract_prices(pp.text, pp.url)
                if prices:
                    break
        overlap = len(set(tokens(f"{key_terms(idea, 5)} {idea.target_customer}"))
                      & set(tokens(page.title + " " + page.text[:5000])))
        name = page.title.split("|")[0].split(" - ")[0].strip()
        if not name or name.lower() in GENERIC_TITLES:
            name = _domain(page.url)
        name = name[:80]
        # Only prices with a clear period are evidence; loose amounts are often examples.
        prices = [p for p in prices if p.period in ("maand", "jaar")]
        first_sentence = next((s.strip() for s in page.text.split("\n") if 40 <= len(s.strip()) <= 300), "")
        comp = Competitor(
            name=name, domain=_domain(page.url), url=page.url,
            description=first_sentence or page.title or "(geen omschrijving gevonden)",
            kind="direct" if overlap >= MIN_OVERLAP + 1 else "mogelijk" if overlap >= MIN_OVERLAP else "indirect",
            prices=prices, source="web",
        )
        return comp, page.text

    # --- analysis -------------------------------------------------------------
    def analyse(self, idea: Idea, prior: list[Evidence] | None = None) -> tuple[CompetitionAnalysis, list[Evidence]]:
        prior = prior or []
        web = not isinstance(self.search, NoSearch) or any(
            e.source_quality is SourceQuality.COMPANY for e in prior)
        inspected = [r for r in (self._inspect(u, idea) for u in self._candidate_urls(idea, prior)) if r]
        competitors = [c for c, _ in inspected]
        gaps: list[str] = []
        mode = f"zoeken: {self.search.name}" if web else "geen webzoekfunctie"

        if self.llm is not None and inspected:
            competitors, gaps = self._classify_llm(idea, inspected)
            mode += " + llm-beoordeling"
        elif self.llm is not None and not web:
            competitors = self._known_llm(idea)
            mode += " + modelkennis (te verifiëren)"

        competitors = [c for c in competitors if c.kind != "geen"]
        evidence = self._evidence(competitors)
        direct = [c for c in competitors if c.kind == "direct" and c.source == "web"]
        if not web:
            saturation, reason = "onbekend", "Geen webzoekfunctie: concurrentie niet onderzocht."
        elif len(direct) <= 1:
            saturation, reason = "laag", f"{len(direct)} directe concurrent(en) gevonden via zoeken."
        elif len(direct) <= 5:
            saturation, reason = "gemiddeld", f"{len(direct)} directe concurrenten gevonden via zoeken."
        else:
            saturation, reason = "hoog", f"{len(direct)} directe concurrenten gevonden via zoeken."

        monthly = sorted(p.monthly_eur for c in competitors if c.source == "web"
                         for p in c.prices if p.monthly_eur)
        analysis = CompetitionAnalysis(
            mode=mode, competitors=competitors, saturation=saturation, saturation_reason=reason,
            price_min_month=monthly[0] if monthly else None,
            price_median_month=round(statistics.median(monthly), 2) if monthly else None,
            price_max_month=monthly[-1] if monthly else None,
            verified_price_count=len(monthly), gaps=gaps,
        )
        return analysis, evidence

    def _classify_llm(self, idea: Idea, inspected: list[tuple[Competitor, str]]):
        from kansenradar.llm import Tier

        listing = "\n\n".join(f"[{i}] {c.name} ({c.domain})\n{text[:1500]}"
                              for i, (c, text) in enumerate(inspected))
        prompt = (f"IDEE: {idea.title}\nDoelgroep: {idea.target_customer}\nProbleem: {idea.problem}\n"
                  f"Oplossing: {idea.one_liner}\n\nKANDIDATEN:\n{listing}")
        try:
            out = self.llm.structured(task="competitors", tier=Tier.EXTRACTION, prompt=prompt,
                                      schema=_LLMCompetitors, system_extra=CLASSIFY_INSTRUCTIONS,
                                      max_tokens=3_000)
        except Exception as exc:
            log.warning("LLM-concurrentiebeoordeling mislukt (%s); regels gebruikt.", exc)
            return [c for c, _ in inspected], []
        result = []
        verdicts = {v.index: v for v in out.competitors}
        for i, (c, _) in enumerate(inspected):
            v = verdicts.get(i)
            if v is not None and v.kind in ("direct", "indirect", "geen"):
                c = c.model_copy(update={"kind": v.kind, "description": v.description,
                                         "strengths": v.strengths, "weaknesses": v.weaknesses})
            result.append(c)
        return result, out.gaps

    def _known_llm(self, idea: Idea) -> list[Competitor]:
        from kansenradar.llm import Tier

        prompt = (f"IDEE: {idea.title}\nDoelgroep: {idea.target_customer}\nProbleem: {idea.problem}\n"
                  f"Oplossing: {idea.one_liner}")
        try:
            out = self.llm.structured(task="known_competitors", tier=Tier.ANALYSIS, prompt=prompt,
                                      schema=_LLMKnownList, system_extra=KNOWN_INSTRUCTIONS,
                                      max_tokens=2_000)
        except Exception as exc:
            log.warning("LLM-kennis over concurrenten niet beschikbaar (%s).", exc)
            return []
        return [Competitor(name=k.name, domain=k.domain, url=f"https://{k.domain}",
                           description=k.description, kind=k.kind if k.kind in ("direct", "indirect") else "mogelijk",
                           source="modelkennis") for k in out.competitors]

    @staticmethod
    def _evidence(competitors: list[Competitor]) -> list[Evidence]:
        out: list[Evidence] = []
        for c in competitors:
            if c.source == "modelkennis":
                out.append(Evidence(
                    claim=f"Mogelijke {c.kind}e concurrent: {c.name} ({c.domain}) — {c.description}",
                    type=EvidenceType.HYPOTHESIS, topic=ResearchTopic.COMPETITORS.value,
                    reasoning="Genoemd op basis van modelkennis; bestaan en aanbod nog te verifiëren."))
                continue
            out.append(Evidence(
                claim=f"{c.kind.capitalize()} concurrent: {c.name} ({c.domain}) — {c.description}",
                type=EvidenceType.ESTIMATE, topic=ResearchTopic.COMPETITORS.value, source_url=c.url,
                source_quality=SourceQuality.COMPANY,
                reasoning="Bestaan vastgesteld via de eigen website; de indeling direct/indirect is een inschatting."))
            for p in c.prices:
                period = {"maand": "per maand", "jaar": "per jaar", "eenmalig": "eenmalig"}.get(p.period, "(periode onbekend)")
                check = verify_quote(p.quote, p.quote)  # quote was cut from the page itself
                out.append(Evidence.create(
                    claim=f"{c.name} vraagt € {p.amount_eur:g} {period}.", type=EvidenceType.FACT,
                    topic=ResearchTopic.PRICING.value, source_url=p.url, source_title=c.name,
                    source_quality=SourceQuality.COMPANY, quote=p.quote, verified=check.found,
                    retrieved_at=datetime.now(UTC)))
        return out
