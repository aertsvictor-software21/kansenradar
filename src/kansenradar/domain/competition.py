"""Competition analysis model."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PricePoint(BaseModel):
    amount_eur: float
    period: str  # maand | jaar | eenmalig | onbekend
    monthly_eur: float | None  # normalised; None if not recurring
    quote: str
    url: str


class Competitor(BaseModel):
    name: str
    domain: str
    url: str
    description: str
    kind: str = Field(description="direct | indirect | mogelijk")
    prices: list[PricePoint] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    source: str = Field(description="web | modelkennis")


class CompetitionAnalysis(BaseModel):
    mode: str
    competitors: list[Competitor]
    saturation: str  # laag | gemiddeld | hoog | onbekend
    saturation_reason: str
    price_min_month: float | None = None
    price_median_month: float | None = None
    price_max_month: float | None = None
    verified_price_count: int = 0
    gaps: list[str] = Field(default_factory=list, description="Mogelijke gaten in de markt (hypotheses).")

    @property
    def direct(self) -> list[Competitor]:
        return [c for c in self.competitors if c.kind == "direct"]
