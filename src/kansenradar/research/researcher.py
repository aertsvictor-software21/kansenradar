"""Market researcher: ten research topics per idea, evidence with verified sources.

Per idea: open data (Eurostat) + one templated web query per topic + pages
supplied by the user. Every claim from a page must pass quote verification.
Topics without a verified FEIT are listed as open questions.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import UTC, datetime

from kansenradar.domain import (
    Evidence,
    EvidenceType,
    Idea,
    ResearchSummary,
    ResearchTopic,
    TopicCoverage,
)
from kansenradar.history.embeddings import tokens
from kansenradar.research.extract import RuleExtractor
from kansenradar.research.fetch import FetchSkipped, PageFetcher
from kansenradar.research.opendata import EurostatClient, market_size_evidence
from kansenradar.research.search import NoSearch, SearchProvider

log = logging.getLogger(__name__)
T = ResearchTopic

QUESTIONS: dict[ResearchTopic, str] = {
    T.DEMAND: "Is er aantoonbare vraag naar een oplossing voor {kern} bij {target}?",
    T.CUSTOMERS: "Wie zijn de potentiële klanten en hoeveel {target} zijn er in België en Nederland?",
    T.PROBLEM: "Hoeveel tijd, geld of risico kost {kern} voor {target}?",
    T.CURRENT_SOLUTION: "Hoe lossen {target} het probleem rond {kern} vandaag op?",
    T.COMPETITORS: "Welke bedrijven bieden al een oplossing voor {kern} aan {target}?",
    T.MARKET_SIZE: "Hoe groot is de markt voor {kern} in België en Nederland?",
    T.PRICING: "Hoeveel betalen {target} voor vergelijkbare oplossingen ({kern})?",
    T.REACH: "Via welke kanalen (federaties, beurzen, media) zijn {target} te bereiken?",
    T.REGULATION: "Welke regelgeving is relevant voor {kern} bij {target}?",
    T.TRENDS: "Welke trends beïnvloeden {kern} bij {target}?",
}
QUERIES: dict[ResearchTopic, str] = {
    T.DEMAND: "{target} {kern} probleem",
    T.CUSTOMERS: "aantal {target} België Nederland",
    T.PROBLEM: "{target} {kern} tijdverlies kosten",
    T.CURRENT_SOLUTION: "{target} {kern} hoe oplossen software",
    T.COMPETITORS: "{kern} software {target}",
    T.MARKET_SIZE: "marktomvang {kern} België Nederland",
    T.PRICING: "{kern} software prijs per maand",
    T.REACH: "beroepsfederatie vakbeurs {target}",
    T.REGULATION: "{target} {kern} wetgeving verplichting",
    T.TRENDS: "{target} {kern} trend digitalisering",
}
USER_URL_TOPICS = (T.DEMAND, T.PRICING, T.COMPETITORS, T.MARKET_SIZE, T.CURRENT_SOLUTION)


_COMMON_VERBS = frozenset("""omzet maakt maken verliezen verlies besteden gebruiken gebruikt helpt
    helpen zoeken vinden lopen regelt regelen werken werkt kunnen moeten worden willen geven""".split())


def key_terms(idea: Idea, n: int = 3) -> str:
    """Most telling words: long content words from solution and problem, excluding the
    target-customer words and common verbs (long Dutch compounds carry the most meaning)."""
    target = set(tokens(idea.target_customer))
    candidates: list[str] = []
    for t in tokens(f"{idea.one_liner} {idea.problem}"):
        if t not in target and t not in _COMMON_VERBS and len(t) > 4 and t not in candidates:
            candidates.append(t)
    best = sorted(candidates, key=len, reverse=True)[:n]
    return " ".join(t for t in candidates if t in best)  # keep original order


def short_target(idea: Idea) -> str:
    return " ".join(t for t in idea.target_customer.split() if not any(c.isdigit() for c in t))[:60]


def coverage(evidence: list[Evidence], max_age_days: int) -> dict[ResearchTopic, TopicCoverage]:
    cov = {t: TopicCoverage() for t in ResearchTopic}
    by_value = {t.value: t for t in ResearchTopic}
    for ev in evidence:
        topic = by_value.get(ev.topic)
        if topic is None:
            continue
        kind = ev.effective_type(max_age_days)
        if kind is EvidenceType.FACT:
            cov[topic].facts += 1
        elif kind is EvidenceType.ESTIMATE:
            cov[topic].estimates += 1
        else:
            cov[topic].hypotheses += 1
    return cov


class MarketResearcher:
    def __init__(
        self,
        search: SearchProvider | None = None,
        fetcher: PageFetcher | None = None,
        extractor=None,
        opendata: EurostatClient | None = None,
        pages_per_query: int = 2,
        max_age_days: int = 365,
    ) -> None:
        self.search = search or NoSearch()
        self.fetcher = fetcher or PageFetcher()
        self.extractor = extractor or RuleExtractor()
        self.opendata = opendata
        self.pages_per_query = pages_per_query
        self.max_age_days = max_age_days

    def research(self, idea: Idea, extra_urls: list[str] | None = None) -> tuple[ResearchSummary, list[Evidence]]:
        fmt = {"target": short_target(idea), "kern": key_terms(idea)}
        evidence: list[Evidence] = []
        stats = Counter()

        if self.opendata is not None:
            evidence += market_size_evidence(idea, self.opendata)

        fetched: set[str] = set()
        skipped: set[str] = set()

        def process(url: str, topics: tuple[ResearchTopic, ...]):
            if url in skipped:
                return
            try:
                page = self.fetcher.fetch(url)
            except FetchSkipped as exc:
                log.info("Overgeslagen %s: %s", url, exc)
                skipped.add(url)
                return
            fetched.add(url)
            for topic in topics:
                result = self.extractor.extract(page, topic, QUESTIONS[topic].format(**fmt))
                evidence.extend(result.evidence)
                stats["rejected"] += result.rejected_quotes

        if not isinstance(self.search, NoSearch):
            for topic, template in QUERIES.items():
                query = template.format(**fmt)
                stats["queries"] += 1
                for hit in self.search.search(query, max_results=self.pages_per_query):
                    process(hit.url, (topic,))
        for url in extra_urls or []:
            process(url, USER_URL_TOPICS)

        # Same quote from the same source only once.
        unique: dict[tuple, Evidence] = {}
        for ev in evidence:
            unique.setdefault((ev.source_url, ev.quote, ev.topic), ev)
        evidence = list(unique.values())

        cov = coverage(evidence + _screening_hypotheses(idea), self.max_age_days)
        open_q = [QUESTIONS[t].format(**fmt) for t, c in cov.items() if not c.has_fact]
        mode = " + ".join(filter(None, [
            f"zoeken: {self.search.name}" if not isinstance(self.search, NoSearch) else "geen webzoekfunctie",
            "open data (Eurostat)" if self.opendata is not None else "",
            f"extractie: {self.extractor.name}",
            f"{len(extra_urls)} eigen bron(nen)" if extra_urls else "",
        ]))
        summary = ResearchSummary(
            mode=mode, researched_at=datetime.now(UTC), coverage=cov,
            queries_run=stats["queries"], pages_fetched=len(fetched),
            pages_skipped=len(skipped), quotes_rejected=stats["rejected"], open_questions=open_q,
        )
        return summary, evidence


def _screening_hypotheses(idea: Idea) -> list[Evidence]:
    """Screening findings count as hypotheses in the coverage (they are already stored)."""
    if not idea.screening:
        return []
    mapping = {"pijn en urgentie": T.PROBLEM, "betalingsbereidheid": T.PRICING,
               "bereikbaarheid klanten": T.REACH, "marktomvang BE/NL": T.MARKET_SIZE,
               "concurrentiedruk": T.COMPETITORS, "uitvoerbaarheid": T.REGULATION}
    return [
        Evidence(claim=s.motivation, type=EvidenceType.HYPOTHESIS,
                 topic=mapping[s.criterion.value].value, reasoning="screening")
        for s in idea.screening.scores if s.known and s.criterion.value in mapping
    ]
