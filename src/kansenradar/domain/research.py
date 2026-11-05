"""Market research model: the ten research topics and the per-idea summary."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ResearchTopic(StrEnum):
    DEMAND = "aantoonbare vraag"
    CUSTOMERS = "potentiële klanten"
    PROBLEM = "opgelost probleem"
    CURRENT_SOLUTION = "huidige oplossing"
    COMPETITORS = "concurrenten"
    MARKET_SIZE = "marktgrootte"
    PRICING = "betalingsbereidheid en prijzen"
    REACH = "bereikbaarheid klanten"
    REGULATION = "regelgeving"
    TRENDS = "trends"


class TopicCoverage(BaseModel):
    facts: int = 0
    estimates: int = 0
    hypotheses: int = 0

    @property
    def has_fact(self) -> bool:
        return self.facts > 0


class ResearchSummary(BaseModel):
    mode: str  # e.g. "zoeken: tavily + open data + llm-extractie"
    researched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    coverage: dict[ResearchTopic, TopicCoverage]
    queries_run: int = 0
    pages_fetched: int = 0
    pages_skipped: int = 0  # robots.txt, errors, non-HTML
    quotes_rejected: int = 0  # quotes that could not be verified in the source
    open_questions: list[str] = Field(default_factory=list)

    @property
    def topics_with_facts(self) -> int:
        return sum(c.has_fact for c in self.coverage.values())
