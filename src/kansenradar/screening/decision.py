"""Deterministic screening decision. Screeners deliver scores; this code decides.

Rules:
  - A score of 1 on a knock-out criterion          -> AFGEWEZEN (reason shown)
  - Weighted score < REJECT_BELOW                   -> AFGEWEZEN (weak across the board)
  - Weighted score >= PASS_FROM and no score <= 1   -> DOOR
  - Otherwise                                       -> TWIJFEL (continues, low priority)

Confidence is never 'hoog' at screening: there are no verified sources yet.
"""

from __future__ import annotations

from kansenradar.domain import (
    CRITERION_WEIGHTS,
    KNOCKOUT_CRITERIA,
    CriterionScore,
    ScreeningDecision,
    ScreeningResult,
)

REJECT_BELOW = 2.6
PASS_FROM = 3.4


def decide(
    scores: list[CriterionScore],
    *,
    screened_by: str,
    key_assumptions: list[str],
    research_questions: list[str],
    summary: str,
) -> ScreeningResult:
    weighted = round(sum(CRITERION_WEIGHTS[s.criterion] * s.score for s in scores), 2)
    knockouts = [
        f"{s.criterion.value}: {s.motivation}"
        for s in scores if s.criterion in KNOCKOUT_CRITERIA and s.score == 1
    ]
    if knockouts:
        decision = ScreeningDecision.REJECT
    elif weighted < REJECT_BELOW:
        decision = ScreeningDecision.REJECT
        knockouts = [f"te zwak over de hele lijn (gewogen {weighted:.2f} < {REJECT_BELOW})"]
    elif weighted >= PASS_FROM and min(s.score for s in scores) > 1:
        decision = ScreeningDecision.PASS
    else:
        decision = ScreeningDecision.DOUBT

    known = sum(s.known for s in scores)
    confidence = "gemiddeld" if screened_by == "llm" and known >= 5 else "laag"
    return ScreeningResult(
        decision=decision, scores=scores, weighted_score=weighted, confidence=confidence,
        knockouts=knockouts, key_assumptions=key_assumptions,
        research_questions=research_questions, summary=summary, screened_by=screened_by,
    )


# Originality is a pure tiebreaker: its maximum contribution (0.03) is smaller than the
# smallest commercial step (1 point on the lightest criterion: 0.15 / 4 * 0.97 = 0.036).
ORIGINALITY_WEIGHT = 0.03


def research_priority(result: ScreeningResult, originality_score: float | None) -> float | None:
    """Order for market research. Commercial signals decide; originality only breaks ties."""
    if result.decision is ScreeningDecision.REJECT:
        return None
    commercial = (result.weighted_score - 1) / 4  # 0..1
    orig = (originality_score or 0) / 10
    priority = (1 - ORIGINALITY_WEIGHT) * commercial + ORIGINALITY_WEIGHT * orig
    if result.decision is ScreeningDecision.DOUBT:
        priority *= 0.7
    return round(priority, 3)
