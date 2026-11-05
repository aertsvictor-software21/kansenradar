"""Critical review (red team) step."""

from __future__ import annotations

from kansenradar.domain import CriticalReview, Idea, IdeaStatus
from kansenradar.review.llm import dossier, enrich_with_llm
from kansenradar.review.rules import investor_checks, review_by_rules, rule_risks, validation_test


def review_idea(idea: Idea, llm=None) -> CriticalReview:
    base = review_by_rules(idea)
    return enrich_with_llm(llm, idea, base) if llm is not None else base


def review_ideas(repo, llm=None, idea_ids: list[str] | None = None) -> list[Idea]:
    if idea_ids:
        ideas = [i for i in (repo.get_idea(x) for x in idea_ids) if i is not None]
    else:
        ideas = [i for i in repo.list_ideas(IdeaStatus.RESEARCHED) if i.review is None]
    for idea in ideas:
        idea.review = review_idea(idea, llm)
        repo.save_idea(idea)
    return ideas


__all__ = ["dossier", "enrich_with_llm", "investor_checks", "review_by_rules", "review_idea",
           "review_ideas", "rule_risks", "validation_test"]
