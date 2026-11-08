"""Screening run over all newly generated ideas."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from kansenradar.domain import Evidence, EvidenceType, Idea, IdeaStatus, ScreeningDecision
from kansenradar.screening.decision import research_priority
from kansenradar.screening.llm import screen_with_llm
from kansenradar.screening.rules import screen_by_rules
from kansenradar.storage import Repository

_STATUS = {
    ScreeningDecision.PASS: IdeaStatus.SCREENED_PASS,
    ScreeningDecision.DOUBT: IdeaStatus.SCREENED_DOUBT,
    ScreeningDecision.REJECT: IdeaStatus.SCREENED_OUT,
}


@dataclass
class ScreeningReport:
    mode: str
    ideas: list[Idea] = field(default_factory=list)

    def ranking(self) -> list[Idea]:
        kept = [i for i in self.ideas if i.research_priority is not None]
        return sorted(kept, key=lambda i: i.research_priority, reverse=True)

    def summary(self) -> str:
        if not self.ideas:
            return "Geen nieuwe ideeën om te screenen. Genereer eerst ideeën."
        c = Counter(i.screening.decision for i in self.ideas)
        return (f"Screening ({self.mode}): {len(self.ideas)} ideeën — "
                f"door {c[ScreeningDecision.PASS]}, twijfel {c[ScreeningDecision.DOUBT]}, "
                f"afgewezen {c[ScreeningDecision.REJECT]}.")


def _evidence(idea: Idea) -> list[Evidence]:
    source = "modelkennis (LLM)" if idea.screening.screened_by == "llm" else "regels op de ideetekst"
    return [
        Evidence(
            claim=f"{s.criterion.value}: score {s.score}/5 — {s.motivation}",
            type=EvidenceType.HYPOTHESIS, topic=f"screening: {s.criterion.value}",
            reasoning=f"Screening op basis van {source}; nog niet geverifieerd met bronnen.",
        )
        for s in idea.screening.scores if s.known
    ]


def screen_ideas(repo: Repository, llm=None, limit: int | None = None,
                 batch_size: int = 3) -> ScreeningReport:
    ideas = repo.list_ideas(IdeaStatus.GENERATED)
    if limit is not None:
        ideas = ideas[:limit]
    report = ScreeningReport(mode="llm" if llm else "regels")
    for start in range(0, len(ideas), batch_size):
        chunk = ideas[start:start + batch_size]
        results = screen_with_llm(llm, chunk) if llm else [screen_by_rules(i) for i in chunk]
        for idea, result in zip(chunk, results):
            idea.screening = result
            idea.status = _STATUS[result.decision]
            idea.research_priority = research_priority(
                result, idea.originality.score if idea.originality else None)
            repo.save_idea(idea)
            repo.add_evidence(idea.id, _evidence(idea))
            report.ideas.append(idea)
    return report
