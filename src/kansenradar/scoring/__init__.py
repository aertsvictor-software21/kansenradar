"""Scoring and ranking step."""

from __future__ import annotations

from kansenradar.domain import Idea, IdeaStatus
from kansenradar.scoring.scorer import CONCLUSION_RANK, rank, score_idea


def score_ideas(repo, idea_ids: list[str] | None = None, max_age_days: int = 365) -> list[Idea]:
    if idea_ids:
        ideas = [i for i in (repo.get_idea(x) for x in idea_ids) if i is not None]
    else:
        ideas = [i for i in repo.list_ideas() if i.status in (IdeaStatus.RESEARCHED, IdeaStatus.ANALYSED)]
    for idea in ideas:
        idea.scorecard = score_idea(idea, repo.evidence_for(idea.id), max_age_days)
        idea.status = IdeaStatus.ANALYSED
        repo.save_idea(idea)
    return rank(ideas)


__all__ = ["CONCLUSION_RANK", "rank", "score_idea", "score_ideas"]
