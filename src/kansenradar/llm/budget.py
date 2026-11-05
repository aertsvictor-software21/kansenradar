"""Cost tracking with a hard ceiling per pipeline run."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from kansenradar.llm.pricing import TokenUsage, cost_usd, worst_case_output_cost


class BudgetExceededError(RuntimeError):
    pass


class CostEntry(BaseModel):
    task: str
    model: str
    usage: TokenUsage
    cost_usd: float
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RunBudget:
    def __init__(self, max_usd: float) -> None:
        if max_usd <= 0:
            raise ValueError("max_usd moet positief zijn.")
        self.max_usd = max_usd
        self.entries: list[CostEntry] = []

    @property
    def spent_usd(self) -> float:
        return sum(e.cost_usd for e in self.entries)

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.max_usd - self.spent_usd)

    def ensure_can_afford(self, model: str, max_tokens: int) -> None:
        """Refuse a call whose worst-case output cost would break the ceiling."""
        projected = self.spent_usd + worst_case_output_cost(model, max_tokens)
        if projected > self.max_usd:
            raise BudgetExceededError(
                f"Budget overschreden: besteed ${self.spent_usd:.2f}, "
                f"deze aanroep kan tot ${projected:.2f} gaan (limiet ${self.max_usd:.2f})."
            )

    def record(self, task: str, model: str, usage: TokenUsage) -> CostEntry:
        entry = CostEntry(task=task, model=model, usage=usage, cost_usd=cost_usd(model, usage))
        self.entries.append(entry)
        return entry

    def by_task(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for e in self.entries:
            totals[e.task] = totals.get(e.task, 0.0) + e.cost_usd
        return totals
