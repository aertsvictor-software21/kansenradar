"""Financial model: assumptions (with basis) and scenario results."""

from __future__ import annotations

from pydantic import BaseModel, Field

from kansenradar.domain.enums import EvidenceType


class Assumption(BaseModel):
    key: str
    label: str
    low: float
    mid: float
    high: float
    unit: str
    basis: EvidenceType
    source: str = Field(description="Herkomst en redenering, in het Nederlands.")
    higher_is_better: bool = True

    def for_scenario(self, scenario: str) -> float:
        """conservatief = ongunstige kant, optimistisch = gunstige kant."""
        if scenario == "realistisch":
            return self.mid
        favourable = self.high if self.higher_is_better else self.low
        unfavourable = self.low if self.higher_is_better else self.high
        return unfavourable if scenario == "conservatief" else favourable


class ScenarioResult(BaseModel):
    scenario: str
    startup_capital: float
    monthly_fixed_costs: float
    marketing_per_month: float
    cac: float
    price_per_month: float
    gross_margin_pct: float
    customers_m12: int
    customers_m36: int
    revenue_m12: float
    revenue_m36: float
    revenue_year1: float
    revenue_year3: float
    net_month36: float
    net_year3: float
    operational_break_even_month: int | None
    payback_month: int | None
    capital_needed: float
    ltv: float
    ltv_cac: float
    cac_payback_months: float | None
    market_cap_reached_month: int | None


class FinancialModel(BaseModel):
    revenue_model: str
    horizon_months: int
    assumptions: list[Assumption]
    scenarios: dict[str, ScenarioResult]
    scalability: str  # laag | gemiddeld | hoog
    scalability_reason: str
    notes: list[str] = Field(default_factory=list)

    def assumption(self, key: str) -> Assumption:
        return next(a for a in self.assumptions if a.key == key)
