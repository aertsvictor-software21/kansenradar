"""Screening result model (fast first-pass assessment, knowledge only, no web)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Criterion(StrEnum):
    PAIN = "pijn en urgentie"
    WILLINGNESS_TO_PAY = "betalingsbereidheid"
    REACHABILITY = "bereikbaarheid klanten"
    MARKET_SIZE = "marktomvang BE/NL"
    COMPETITION = "concurrentiedruk"  # 5 = weinig druk, 1 = gedomineerd
    FEASIBILITY = "uitvoerbaarheid"   # incl. regelgeving, kapitaal, techniek


# Weights for the screening priority (not the final 100-point score of phase 8).
CRITERION_WEIGHTS: dict[Criterion, float] = {
    Criterion.PAIN: 0.20,
    Criterion.WILLINGNESS_TO_PAY: 0.20,
    Criterion.REACHABILITY: 0.15,
    Criterion.MARKET_SIZE: 0.15,
    Criterion.COMPETITION: 0.15,
    Criterion.FEASIBILITY: 0.15,
}

# A score of 1 on one of these is a knock-out.
KNOCKOUT_CRITERIA = frozenset({
    Criterion.PAIN, Criterion.WILLINGNESS_TO_PAY, Criterion.COMPETITION, Criterion.FEASIBILITY,
})


class ScreeningDecision(StrEnum):
    PASS = "DOOR"
    DOUBT = "TWIJFEL"
    REJECT = "AFGEWEZEN"


class CriterionScore(BaseModel):
    criterion: Criterion
    score: int = Field(ge=1, le=5)
    known: bool = Field(description="False = geen kennis, score is neutraal (3).")
    motivation: str


class ScreeningResult(BaseModel):
    decision: ScreeningDecision
    scores: list[CriterionScore]
    weighted_score: float = Field(ge=1, le=5)
    confidence: str  # laag | gemiddeld | hoog
    knockouts: list[str] = Field(default_factory=list)
    key_assumptions: list[str] = Field(default_factory=list)
    research_questions: list[str] = Field(default_factory=list)
    summary: str
    screened_by: str  # regels | llm

    @model_validator(mode="after")
    def _all_criteria_once(self) -> ScreeningResult:
        got = [s.criterion for s in self.scores]
        if sorted(got) != sorted(Criterion):
            raise ValueError("Elke screeningscriterium moet precies één keer gescoord worden.")
        return self

    def score_of(self, criterion: Criterion) -> int:
        return next(s.score for s in self.scores if s.criterion is criterion)
