"""Scorecard: 100-point commercial score, confidence and final conclusion."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from kansenradar.domain.enums import Conclusion

# Max points per criterion, as specified.
CRITERIA_MAX: dict[str, int] = {
    "Marktvraag": 20,
    "Winstgevendheid": 15,
    "Klantenwerving": 15,
    "Onderscheidend vermogen": 15,
    "Schaalbaarheid": 10,
    "Benodigd startkapitaal": 10,
    "Uitvoerbaarheid": 10,
    "Markttiming": 5,
}


class SubScore(BaseModel):
    criterion: str
    points: float
    max_points: int
    rationale: str
    capped_by: str | None = Field(default=None, description="Bewijsplafond dat de score beperkte.")


class Scorecard(BaseModel):
    subscores: list[SubScore]
    total: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=100, description="Betrouwbaarheid in %.")
    confidence_reasons: list[str]
    conclusion: Conclusion
    conclusion_reason: str
    scored_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def sub(self, name: str) -> SubScore:
        return next(s for s in self.subscores if s.criterion == name)
