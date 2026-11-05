"""Competition step for researched ideas."""

from __future__ import annotations

from kansenradar.competition.analyzer import CompetitorAnalyzer
from kansenradar.domain import Idea, IdeaStatus
from kansenradar.storage import Repository


def analyse_competition(repo: Repository, analyzer: CompetitorAnalyzer,
                        idea_ids: list[str] | None = None) -> list[Idea]:
    if idea_ids:
        ideas = [i for i in (repo.get_idea(x) for x in idea_ids) if i is not None]
    else:
        ideas = [i for i in repo.list_ideas(IdeaStatus.RESEARCHED) if i.competition is None]
    done = []
    for idea in ideas:
        analysis, evidence = analyzer.analyse(idea, repo.evidence_for(idea.id))
        idea.competition = analysis
        repo.add_evidence(idea.id, evidence)
        repo.save_idea(idea)
        done.append(idea)
    return done
