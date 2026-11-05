"""Critical investor review (red team) and first validation test."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from kansenradar.domain.enums import EvidenceType


class InvestorQuestion(StrEnum):
    SATURATED = "Is de markt verzadigd?"
    WILL_PAY = "Wil de klant werkelijk betalen?"
    COPYABLE = "Is het product eenvoudig te kopiëren?"
    CAC_TOO_HIGH = "Zijn de acquisitiekosten te hoog?"
    MARGINS_OK = "Zijn de marges aantrekkelijk genoeg?"
    MOAT = "Bestaat er een duurzaam concurrentievoordeel?"


class InvestorCheck(BaseModel):
    question: InvestorQuestion
    verdict: str  # ja | nee | onbekend
    alarming: bool
    explanation: str
    basis: EvidenceType
    source: str  # regels | llm


class Risk(BaseModel):
    title: str
    description: str
    likelihood: int = Field(ge=1, le=5)
    impact: int = Field(ge=1, le=5)
    mitigation: str
    basis: EvidenceType
    source: str

    @property
    def severity(self) -> int:
        return self.likelihood * self.impact

    @property
    def fatal(self) -> bool:
        return self.likelihood >= 4 and self.impact == 5 and self.basis is not EvidenceType.HYPOTHESIS


class ValidationTest(BaseModel):
    critical_assumption: str
    method: str
    steps: list[str]
    success_criteria: list[str]
    kill_criteria: list[str]
    duration_weeks: int
    cost_eur: str


class CriticalReview(BaseModel):
    checks: list[InvestorCheck]
    risks: list[Risk]  # top 5, most severe first
    fatal_flaws: list[str]
    summary: str
    validation_test: ValidationTest
    reviewed_by: str  # regels | regels + llm

    def check(self, q: InvestorQuestion) -> InvestorCheck:
        return next(c for c in self.checks if c.question is q)
