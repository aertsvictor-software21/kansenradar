"""Finance step for researched ideas."""

from __future__ import annotations

from kansenradar.domain import Idea, IdeaStatus
from kansenradar.finance.model import build_financial_model
from kansenradar.storage import Repository


def analyse_finance(repo: Repository, idea_ids: list[str] | None = None,
                    marketing_per_month: float = 750) -> list[Idea]:
    if idea_ids:
        ideas = [i for i in (repo.get_idea(x) for x in idea_ids) if i is not None]
    else:
        ideas = [i for i in repo.list_ideas(IdeaStatus.RESEARCHED) if i.finance is None]
    for idea in ideas:
        idea.finance = build_financial_model(idea, repo.evidence_for(idea.id), marketing_per_month)
        repo.save_idea(idea)
    return ideas
