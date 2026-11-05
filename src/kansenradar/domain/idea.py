"""Idea domain model."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

from kansenradar.domain.enums import IdeaStatus, Sector
from kansenradar.domain.competition import CompetitionAnalysis
from kansenradar.domain.finance import FinancialModel
from kansenradar.domain.research import ResearchSummary
from kansenradar.domain.review import CriticalReview
from kansenradar.domain.scorecard import Scorecard
from kansenradar.domain.screening import ScreeningResult


class OriginalityStrategy(StrEnum):
    MARKET_COMBINATION = "combinatie van bestaande markten"
    INTERNATIONAL_ARBITRAGE = "internationaal model, lokaal onderontwikkeld"
    UNUSUAL_NICHE = "ongebruikelijke niche"
    NEW_TECH_APPLICATION = "nieuwe toepassing van bestaande technologie"
    FUNDAMENTAL_IMPROVEMENT = "fundamentele verbetering van bestaande oplossing"
    PROBLEM_DRIVEN = "probleemgedreven (onopgelost probleem)"


class IdeaCandidate(BaseModel):
    """An idea as produced by the generation module (all text in Dutch)."""

    title: str = Field(min_length=3, max_length=120)
    one_liner: str = Field(min_length=10, max_length=300)
    problem: str
    target_customer: str
    sector: Sector
    markets: list[str] = Field(default_factory=lambda: ["BE", "NL"])
    business_model: str
    strategy: OriginalityStrategy
    inspiration: str | None = Field(
        default=None, description="Bv. buitenlands voorbeeld (HYPOTHESE tot geverifieerd)."
    )
    why_now: str | None = Field(default=None, description="Waarom is dit nu kansrijk?")
    differentiator: str | None = Field(default=None, description="Wat maakt dit niet generiek?")
    generated_by: str = Field(default="manueel", description="llm | sjabloon | manueel")
    seed_key: str | None = Field(default=None, description="Herkomst in de originaliteitsmotor.")

    def fingerprint_text(self) -> str:
        """Text used for embeddings/dedup: the essence, not the marketing wording."""
        return (
            f"Probleem: {self.problem}\nDoelgroep: {self.target_customer}\n"
            f"Oplossing: {self.one_liner}\nVerdienmodel: {self.business_model}"
        )


class OriginalityAssessment(BaseModel):
    """Separate label; deliberately NOT part of the 100-point commercial score."""

    score: float = Field(ge=0, le=10)
    label: str  # laag | gemiddeld | hoog
    novelty_vs_history: float = Field(ge=0, le=1)
    strategy: OriginalityStrategy
    llm_rating: float | None = Field(default=None, ge=0, le=10)
    explanation: str


class Idea(IdeaCandidate):
    id: str = Field(default_factory=lambda: f"idea_{uuid4().hex[:12]}")
    status: IdeaStatus = IdeaStatus.GENERATED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_id: str | None = None
    duplicate_of: str | None = None
    similar_to: list[str] = Field(default_factory=list)
    dedup_note: str | None = None
    quality_flags: list[str] = Field(default_factory=list)
    originality: OriginalityAssessment | None = None
    screening: ScreeningResult | None = None
    research_priority: float | None = Field(default=None, description="Volgorde voor fase 4.")
    research: ResearchSummary | None = None
    competition: CompetitionAnalysis | None = None
    finance: FinancialModel | None = None
    review: CriticalReview | None = None
    scorecard: Scorecard | None = None
