"""Research run: the highest-priority screened ideas, in order."""

from __future__ import annotations

from dataclasses import dataclass, field

from kansenradar.domain import Idea, IdeaStatus
from kansenradar.research.researcher import MarketResearcher
from kansenradar.storage import Repository

RESEARCHABLE = (IdeaStatus.SCREENED_PASS, IdeaStatus.SCREENED_DOUBT)


@dataclass
class ResearchReport:
    ideas: list[Idea] = field(default_factory=list)

    def summary(self) -> str:
        if not self.ideas:
            return "Geen gescreende ideeën om te onderzoeken. Voer eerst 'screen' uit."
        lines = [f"Marktonderzoek: {len(self.ideas)} idee(ën)."]
        for i in self.ideas:
            r = i.research
            lines.append(
                f"  {i.id}  onderwerpen met feit: {r.topics_with_facts}/10 | pagina's {r.pages_fetched} "
                f"(overgeslagen {r.pages_skipped}) | verworpen citaten {r.quotes_rejected} | {i.title}"
            )
        return "\n".join(lines)


def research_queue(repo: Repository) -> list[Idea]:
    ideas = [i for i in repo.list_ideas() if i.status in RESEARCHABLE and i.research is None]
    return sorted(ideas, key=lambda i: i.research_priority or 0, reverse=True)


def research_ideas(repo: Repository, researcher: MarketResearcher, top_n: int = 3,
                   idea_ids: list[str] | None = None,
                   extra_urls: dict[str, list[str]] | None = None) -> ResearchReport:
    if idea_ids:
        ideas = [i for i in (repo.get_idea(x) for x in idea_ids) if i is not None]
    else:
        ideas = research_queue(repo)[:top_n]
    report = ResearchReport()
    for idea in ideas:
        summary, evidence = researcher.research(idea, (extra_urls or {}).get(idea.id))
        idea.research = summary
        idea.status = IdeaStatus.RESEARCHED
        repo.add_evidence(idea.id, evidence)
        repo.save_idea(idea)
        report.ideas.append(idea)
    return report
