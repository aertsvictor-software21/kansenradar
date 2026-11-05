"""LLM screener: scores ideas in batches (saves free-tier calls), code decides.

The model only provides criterion scores, assumptions and research questions.
The decision is made by decision.decide(). Deterministic rule findings that are
certain (no revenue model) override the model. If the LLM fails or returns an
invalid result for an idea, the rule-based screener takes over for that idea.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from kansenradar.domain import Criterion, CriterionScore, Idea, ScreeningResult
from kansenradar.screening.decision import decide
from kansenradar.screening.rules import screen_by_rules

log = logging.getLogger(__name__)


class _LLMCriterion(BaseModel):
    criterion: Criterion
    score: int = Field(ge=1, le=5)
    known: bool = Field(description="false als je hier geen betrouwbare kennis over hebt")
    motivation: str


class _LLMScreen(BaseModel):
    idea_index: int
    scores: list[_LLMCriterion]
    key_assumptions: list[str] = Field(description="Aannames waarop het idee steunt (hypotheses).")
    research_questions: list[str] = Field(description="Vragen die marktonderzoek moet beantwoorden.")
    summary: str = Field(description="Kritische samenvatting in 1-2 zinnen.")


class _LLMScreenBatch(BaseModel):
    results: list[_LLMScreen]


SCREENING_INSTRUCTIONS = f"""\
Taak: snelle, kritische screening van businessideeën voor België en Nederland, als een \
sceptische investeerder. Je hebt geen internet; gebruik alleen wat je betrouwbaar weet.

Scoor elk idee op ELK van deze criteria (1-5):
- "{Criterion.PAIN.value}": hoe pijnlijk/dringend is het probleem? 1 = nice-to-have.
- "{Criterion.WILLINGNESS_TO_PAY.value}": betaalt deze doelgroep hier echt voor? 1 = verwacht gratis.
- "{Criterion.REACHABILITY.value}": hoe makkelijk en goedkoop bereik je deze klanten?
- "{Criterion.MARKET_SIZE.value}": is de doelgroep in BE/NL groot genoeg voor een rendabel bedrijf?
- "{Criterion.COMPETITION.value}": 5 = weinig concurrentie, 1 = gedomineerd door grote of gratis spelers.
- "{Criterion.FEASIBILITY.value}": haalbaar voor een kleine start (regelgeving, kapitaal, techniek)? \
1 = bv. bankvergunning of medische certificering vereist.

Regels:
- Score 1 alleen bij een duidelijke dealbreaker, en leg uit waarom.
- Weet je iets niet: known=false en score 3. Gok niet.
- Wees streng: de meeste ideeën verdienen geen hoge scores.
- Onderzoeksvragen moeten concreet en toetsbaar zijn (wie, hoeveel, welke prijs)."""


def _format(ideas: list[Idea]) -> str:
    parts = []
    for i, idea in enumerate(ideas):
        parts.append(
            f"[{i}] {idea.title}\nDoelgroep: {idea.target_customer}\nProbleem: {idea.problem}\n"
            f"Oplossing: {idea.one_liner}\nVerdienmodel: {idea.business_model}\n"
            f"Waarom nu: {idea.why_now or '-'}\nOnderscheidend: {idea.differentiator or '-'}"
        )
    return "\n\n".join(parts)


def _merge(idea: Idea, item: _LLMScreen) -> ScreeningResult:
    scores = [CriterionScore(**s.model_dump()) for s in item.scores]
    rules = screen_by_rules(idea)
    # Certain rule finding overrides the model: no revenue model is a knock-out.
    rule_wtp = rules.score_of(Criterion.WILLINGNESS_TO_PAY)
    if rule_wtp == 1:
        scores = [s for s in scores if s.criterion is not Criterion.WILLINGNESS_TO_PAY]
        scores.append(next(s for s in rules.scores if s.criterion is Criterion.WILLINGNESS_TO_PAY))
    questions = list(dict.fromkeys(item.research_questions + rules.research_questions))
    return decide(scores, screened_by="llm", key_assumptions=item.key_assumptions,
                  research_questions=questions, summary=item.summary)


def screen_with_llm(llm, ideas: list[Idea]) -> list[ScreeningResult]:
    from kansenradar.llm import Tier

    try:
        batch = llm.structured(
            task="screening", tier=Tier.SCREENING, prompt=_format(ideas),
            schema=_LLMScreenBatch, system_extra=SCREENING_INSTRUCTIONS, max_tokens=6_000,
        )
        by_index = {r.idea_index: r for r in batch.results}
    except Exception as exc:  # provider down, limits, invalid output
        log.warning("LLM-screening mislukt (%s); regels gebruikt.", exc)
        by_index = {}

    results = []
    for i, idea in enumerate(ideas):
        item = by_index.get(i)
        if item is not None:
            try:
                results.append(_merge(idea, item))
                continue
            except ValueError as exc:  # e.g. a criterion missing or duplicated
                log.info("Ongeldige screening voor idee %d (%s); regels gebruikt.", i, exc)
        results.append(screen_by_rules(idea))
    return results
