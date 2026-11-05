"""LLM red team on top of the rule review.

The model plays a sceptical investor and tries to prove the idea will fail. It may
add risks and fill checks the rules could not answer ('onbekend'). Data-based
verdicts from the rules (finance, competition with web data) cannot be overruled.
Everything the model contributes is a HYPOTHESE.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from kansenradar.domain import (
    CriticalReview,
    EvidenceType,
    Idea,
    InvestorCheck,
    InvestorQuestion,
    Risk,
)

log = logging.getLogger(__name__)


class _LLMCheck(BaseModel):
    question: InvestorQuestion
    verdict: str = Field(description="ja | nee | onbekend")
    explanation: str


class _LLMRisk(BaseModel):
    title: str
    description: str
    likelihood: int = Field(ge=1, le=5)
    impact: int = Field(ge=1, le=5)
    mitigation: str


class _LLMReview(BaseModel):
    checks: list[_LLMCheck]
    risks: list[_LLMRisk] = Field(description="De grootste risico's, ernstigste eerst.")
    failure_story: str = Field(description="Het meest waarschijnlijke scenario waarin dit idee mislukt.")


RED_TEAM_INSTRUCTIONS = """\
Je bent een bijzonder kritische investeerder. Je doel: bewijzen waarom dit idee zou mislukken. \
Beantwoord elke investeerdersvraag (ja/nee/onbekend) en benoem de grootste risico's met \
kans en impact (1-5). Baseer je op het dossier; zeg 'onbekend' als het dossier en je kennis \
tekortschieten. Verzin geen feiten, cijfers of bedrijven."""

# For these questions a data-based rule verdict wins over the model.
DATA_DRIVEN = {InvestorQuestion.CAC_TOO_HIGH, InvestorQuestion.MARGINS_OK, InvestorQuestion.SATURATED}
ALARMING_IF = {
    InvestorQuestion.SATURATED: "ja", InvestorQuestion.WILL_PAY: "nee", InvestorQuestion.COPYABLE: "ja",
    InvestorQuestion.CAC_TOO_HIGH: "ja", InvestorQuestion.MARGINS_OK: "nee", InvestorQuestion.MOAT: "nee",
}


def dossier(idea: Idea) -> str:
    lines = [f"IDEE: {idea.title}", f"Omschrijving: {idea.one_liner}", f"Probleem: {idea.problem}",
             f"Doelgroep: {idea.target_customer}", f"Verdienmodel: {idea.business_model}",
             f"Onderscheidend: {idea.differentiator or '-'}"]
    if idea.screening:
        lines.append("Screening: " + "; ".join(f"{s.criterion.value} {s.score}/5" for s in idea.screening.scores))
    if idea.research:
        lines.append(f"Onderzoek: {idea.research.topics_with_facts}/10 onderwerpen met geverifieerd feit.")
    if idea.competition:
        c = idea.competition
        lines.append(f"Concurrentie: verzadiging {c.saturation}; " + ", ".join(
            f"{x.name} ({x.kind})" for x in c.competitors[:8]))
    if idea.finance:
        r = idea.finance.scenarios["realistisch"]
        lines.append(f"Financieel (realistisch, op aannames): prijs € {r.price_per_month:g}/mnd, CAC € {r.cac:g}, "
                     f"LTV/CAC {r.ltv_cac:.1f}, omzet jaar 3 € {r.revenue_year3:,.0f}, break-even maand "
                     f"{r.operational_break_even_month or '>36'}.")
    return "\n".join(lines)


def enrich_with_llm(llm, idea: Idea, base: CriticalReview) -> CriticalReview:
    from kansenradar.llm import Tier

    try:
        out = llm.structured(task="red_team", tier=Tier.ANALYSIS, prompt=dossier(idea),
                             schema=_LLMReview, system_extra=RED_TEAM_INSTRUCTIONS, max_tokens=4_000)
    except Exception as exc:
        log.warning("LLM-red team niet beschikbaar (%s); alleen regels.", exc)
        return base

    llm_checks = {c.question: c for c in out.checks if c.verdict in ("ja", "nee", "onbekend")}
    checks = []
    for rule in base.checks:
        lc = llm_checks.get(rule.question)
        keep_rule = lc is None or (rule.question in DATA_DRIVEN and rule.basis is not EvidenceType.HYPOTHESIS) \
            or (rule.verdict != "onbekend" and lc.verdict == "onbekend")
        if keep_rule:
            checks.append(rule)
        else:
            checks.append(InvestorCheck(
                question=rule.question, verdict=lc.verdict, alarming=lc.verdict == ALARMING_IF[rule.question]
                or lc.verdict == "onbekend", explanation=f"{lc.explanation} (regels: {rule.explanation})",
                basis=EvidenceType.HYPOTHESIS, source="llm"))

    seen = {r.title.lower() for r in base.risks}
    extra = [Risk(title=r.title, description=r.description, likelihood=r.likelihood, impact=r.impact,
                  mitigation=r.mitigation, basis=EvidenceType.HYPOTHESIS, source="llm")
             for r in out.risks if r.title.lower() not in seen]
    risks = sorted(base.risks + extra, key=lambda r: r.severity, reverse=True)[:5]
    fatal = [f"{r.title}: {r.description}" for r in risks if r.fatal]
    alarming = sum(c.alarming for c in checks)
    summary = (f"{alarming}/6 investeerdersvragen zijn alarmerend of onbeantwoord. "
               + ("Fatale gebreken gevonden." if fatal else "Geen fatale gebreken op basis van bewijs.")
               + f" Waarschijnlijkste faalscenario: {out.failure_story}")
    return base.model_copy(update={"checks": checks, "risks": risks, "fatal_flaws": fatal,
                                   "summary": summary, "reviewed_by": "regels + llm"})
