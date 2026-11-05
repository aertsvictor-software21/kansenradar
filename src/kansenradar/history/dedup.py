"""Duplicate detection against the historical database.

Decision flow for a new idea:
  1. Exact match on the normalised fingerprint            -> DUPLICAAT
  2. Overall embedding similarity >= duplicate threshold  -> DUPLICAAT
  3. Overall similarity <  similar threshold              -> UNIEK
  4. Grey zone: a judge decides                           -> DUPLICAAT or VERWANT

Design choice: when in doubt, mark as VERWANT (related) instead of discarding.
A wrongly discarded idea is lost for good; a wrongly kept one only costs some
extra analysis. Related ideas keep a link to the earlier idea and its outcome.

Judges:
  - HeuristicJudge: no LLM; requires target customer, problem AND solution to match.
  - LLMJudge: asks an LLM; falls back to the heuristic if no provider is available.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, Field

from kansenradar.domain import Idea, IdeaCandidate, IdeaStatus
from kansenradar.history.embeddings import Embedder, normalize
from kansenradar.storage import Repository, SimilarIdea, cosine

log = logging.getLogger(__name__)


class DedupVerdict(StrEnum):
    UNIQUE = "UNIEK"
    RELATED = "VERWANT"
    DUPLICATE = "DUPLICAAT"


class DuplicateCheck(BaseModel):
    verdict: DedupVerdict
    match_id: str | None = None
    match_title: str | None = None
    similarity: float | None = None
    judged_by: str = Field(description="exact | drempel | heuristiek | llm")
    reason: str
    earlier_outcome: str | None = None


class JudgeResult(BaseModel):
    same_idea: bool
    related: bool = Field(default=True, description="Deelt minstens doelgroep, probleem of oplossing.")
    explanation: str
    judged_by: str


class Judge(Protocol):
    def judge(self, candidate: IdeaCandidate, existing: Idea) -> JudgeResult: ...


# A field counts as "shared" (-> VERWANT) at this fraction of its duplicate minimum.
RELATED_FACTOR = 0.75


class HeuristicJudge:
    """Field-by-field comparison. Precision first: all three core fields must match."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def field_similarities(self, a: IdeaCandidate, b: IdeaCandidate) -> dict[str, float]:
        pairs = {
            "doelgroep": (a.target_customer, b.target_customer),
            "probleem": (a.problem, b.problem),
            "oplossing": (a.one_liner, b.one_liner),
        }
        vectors = self.embedder.embed([t for pair in pairs.values() for t in pair])
        return {
            name: cosine(vectors[2 * i], vectors[2 * i + 1]) for i, name in enumerate(pairs)
        }

    def judge(self, candidate: IdeaCandidate, existing: Idea) -> JudgeResult:
        t = self.embedder.thresholds
        sims = self.field_similarities(candidate, existing)
        minimums = {"doelgroep": t.field_target, "probleem": t.field_problem,
                    "oplossing": t.field_solution}
        failing = [k for k, v in sims.items() if v < minimums[k]]
        shared = [k for k, v in sims.items() if v >= RELATED_FACTOR * minimums[k]]
        detail = ", ".join(f"{k} {v:.2f}" for k, v in sims.items())
        if not failing:
            return JudgeResult(
                same_idea=True, judged_by="heuristiek",
                explanation=f"Doelgroep, probleem en oplossing komen overeen ({detail}).",
            )
        if not shared:
            return JudgeResult(
                same_idea=False, related=False, judged_by="heuristiek",
                explanation=f"Geen enkel kernveld komt overeen ({detail}).",
            )
        return JudgeResult(
            same_idea=False, judged_by="heuristiek",
            explanation=f"Gedeeld: {', '.join(shared)}; verschilt op {', '.join(failing)} ({detail}).",
        )


class _LLMJudgement(BaseModel):
    same_idea: bool = Field(description="True als het in essentie hetzelfde businessidee is.")
    related: bool = Field(description="True als ze doelgroep, probleem of oplossing delen.")
    explanation: str = Field(description="Korte uitleg in het Nederlands.")


_JUDGE_INSTRUCTIONS = """\
Beoordeel of twee businessideeën in essentie HETZELFDE idee zijn. Hetzelfde betekent: \
dezelfde doelgroep, hetzelfde kernprobleem en een vergelijkbare oplossing. Een ander \
probleem voor dezelfde doelgroep, of dezelfde technologie voor een andere doelgroep, \
is NIET hetzelfde idee. Twijfel je, antwoord dan false. Geef daarnaast aan of ze verwant zijn \
(ze delen de doelgroep, het kernprobleem of de oplossing)."""


class LLMJudge:
    def __init__(self, llm_client, fallback: Judge) -> None:
        self.llm = llm_client
        self.fallback = fallback

    def judge(self, candidate: IdeaCandidate, existing: Idea) -> JudgeResult:
        from kansenradar.llm import BudgetExceededError, LLMError, Tier

        prompt = (
            f"IDEE A (nieuw):\n{candidate.fingerprint_text()}\n\n"
            f"IDEE B (bestaand):\n{existing.fingerprint_text()}"
        )
        try:
            out = self.llm.structured(
                task="dedup_judge", tier=Tier.SCREENING, prompt=prompt,
                schema=_LLMJudgement, system_extra=_JUDGE_INSTRUCTIONS, max_tokens=1_000,
            )
        except (LLMError, BudgetExceededError, ConnectionError, OSError) as exc:
            log.warning("LLM-rechter niet beschikbaar (%s); heuristiek gebruikt.", exc)
            return self.fallback.judge(candidate, existing)
        return JudgeResult(same_idea=out.same_idea, related=out.related or out.same_idea,
                           explanation=out.explanation, judged_by="llm")


def _exact_key(c: IdeaCandidate) -> str:
    return normalize(f"{c.problem} {c.target_customer} {c.one_liner}")


def _earlier_outcome(match: SimilarIdea) -> str | None:
    if match.conclusion:
        return f"eerdere conclusie: {match.conclusion}"
    if match.status == IdeaStatus.SCREENED_OUT.value:
        return "eerder afgewezen bij screening"
    if match.status == IdeaStatus.REJECTED_GENERIC.value:
        return "eerder afgewezen als generiek"
    if match.status in (IdeaStatus.IN_RESEARCH.value, IdeaStatus.ANALYSED.value):
        return f"eerder al {match.status}"
    return None


class DuplicateChecker:
    def __init__(self, repo: Repository, embedder: Embedder, judge: Judge | None = None) -> None:
        self.repo = repo
        self.embedder = embedder
        self.judge = judge or HeuristicJudge(embedder)

    def check(self, candidate: IdeaCandidate) -> tuple[DuplicateCheck, list[float]]:
        embedding = self.embedder.embed([candidate.fingerprint_text()])[0]
        matches = self.repo.most_similar(embedding, self.embedder.name, limit=3)
        if not matches:
            return DuplicateCheck(
                verdict=DedupVerdict.UNIQUE, judged_by="drempel",
                reason="Geen vergelijkbare ideeën in de historische database.",
            ), embedding

        best = matches[0]
        thresholds = self.embedder.thresholds
        base = dict(match_id=best.idea_id, match_title=best.title,
                    similarity=round(best.similarity, 4), earlier_outcome=_earlier_outcome(best))
        existing = self.repo.get_idea(best.idea_id)

        if existing and _exact_key(existing) == _exact_key(candidate):
            return DuplicateCheck(verdict=DedupVerdict.DUPLICATE, judged_by="exact",
                                  reason="Identiek aan een eerder idee.", **base), embedding
        if best.similarity >= thresholds.duplicate:
            return DuplicateCheck(
                verdict=DedupVerdict.DUPLICATE, judged_by="drempel",
                reason=f"Zeer hoge gelijkenis ({best.similarity:.2f}) met '{best.title}'.", **base,
            ), embedding
        if best.similarity < thresholds.similar or existing is None:
            return DuplicateCheck(
                verdict=DedupVerdict.UNIQUE, judged_by="drempel",
                reason=f"Hoogste gelijkenis {best.similarity:.2f} ligt onder de drempel.", **base,
            ), embedding

        result = self.judge.judge(candidate, existing)
        verdict = (DedupVerdict.DUPLICATE if result.same_idea
                   else DedupVerdict.RELATED if result.related else DedupVerdict.UNIQUE)
        return DuplicateCheck(verdict=verdict, judged_by=result.judged_by,
                              reason=result.explanation, **base), embedding

    def register(self, candidate: IdeaCandidate, run_id: str | None = None) -> tuple[Idea, DuplicateCheck]:
        """Check a candidate and store it (also duplicates, so the history is complete)."""
        check, embedding = self.check(candidate)
        idea = Idea(**candidate.model_dump(), run_id=run_id)
        note = check.reason + (f" ({check.earlier_outcome})" if check.earlier_outcome else "")
        if check.verdict is DedupVerdict.DUPLICATE:
            idea.status = IdeaStatus.DUPLICATE
            idea.duplicate_of = check.match_id
            idea.dedup_note = note
        elif check.verdict is DedupVerdict.RELATED:
            idea.similar_to = [check.match_id] if check.match_id else []
            idea.dedup_note = note
        self.repo.save_idea(idea, embedding, self.embedder.name)
        return idea, check

    def register_many(self, candidates: list[IdeaCandidate], run_id: str | None = None):
        """Sequential, so ideas within the same batch are also deduplicated."""
        return [self.register(c, run_id) for c in candidates]


def reindex(repo: Repository, embedder: Embedder) -> int:
    """Re-embed all ideas with a (new) embedder. Returns the number updated."""
    ideas = repo.ideas_without_embedding(embedder.name)
    if not ideas:
        return 0
    vectors = embedder.embed([i.fingerprint_text() for i in ideas])
    for idea, vec in zip(ideas, vectors):
        repo.set_embedding(idea.id, vec, embedder.name)
    return len(ideas)


__all__ = [
    "DedupVerdict",
    "DuplicateCheck",
    "DuplicateChecker",
    "HeuristicJudge",
    "JudgeResult",
    "LLMJudge",
    "reindex",
]
