"""Idea generator: LLM mode with automatic fallback to template mode.

Template mode needs no LLM: it composes ideas from the catalog (problem x
technology lens). These ideas are plainer, are marked generated_by="sjabloon",
and serve to keep the platform usable offline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import BaseModel, Field

from kansenradar.domain import IdeaCandidate
from kansenradar.generation.originality import Seed

log = logging.getLogger(__name__)


@dataclass
class GeneratedCandidate:
    candidate: IdeaCandidate
    seed: Seed
    llm_rating: float | None = None


class _LLMIdea(BaseModel):
    title: str = Field(description="Korte, concrete naam van het concept (max 80 tekens).")
    one_liner: str = Field(description="Eén zin: wat het is en voor wie.")
    problem: str = Field(description="Het concrete probleem, met wie er last van heeft en wat het kost.")
    target_customer: str = Field(description="Specifieke doelgroep (type, grootte, regio). Nooit 'bedrijven' of 'iedereen'.")
    business_model: str = Field(description="Hoe er verdiend wordt.")
    why_now: str = Field(description="Waarom nu? Trend, technologie of regelgeving (als hypothese).")
    differentiator: str = Field(description="Waarom dit niet generiek of eenvoudig te kopiëren is.")
    inspiration: str | None = Field(default=None, description="Buitenlands voorbeeld, enkel als je zeker bent dat het bestaat.")
    originality_self_rating: float = Field(ge=0, le=10, description="Eerlijke inschatting van originaliteit.")


class _LLMIdeaBatch(BaseModel):
    ideas: list[_LLMIdea]


GENERATION_INSTRUCTIONS = """\
Taak: bedenk businessideeën voor België en/of Nederland op basis van de opdracht.

Eisen per idee:
- Specifieke doelgroep (bv. 'zelfstandige kinesitherapeuten in Vlaanderen', niet 'zorgverleners').
- Concreet probleem dat geld, tijd of risico kost; geen 'nice to have'.
- Een duidelijk verdienmodel waarvoor de klant aantoonbaar zou kunnen betalen.
- Benoem waarom het niet generiek is en niet eenvoudig te kopiëren.
- Geen 'Uber/Airbnb voor X', geen modewoorden, geen ideeën die al door grote spelers volledig gedekt worden.
- Vermeld bestaande bedrijven of cijfers alleen als je er zeker van bent; alles wat je beweert \
wordt later geverifieerd en is voorlopig een hypothese.
- Liever minder ideeën van hoge kwaliteit dan het gevraagde aantal met opvulling. Een lege lijst mag."""


def _template_candidate(seed: Seed) -> IdeaCandidate:
    ps, tech = seed.problem, seed.tech
    assert ps is not None and tech is not None
    title = f"{tech.capability[:1].upper()}{tech.capability[1:]} voor {ps.target_customer}"
    return IdeaCandidate(
        title=title[:120],
        one_liner=(f"Maandelijkse dienst voor {ps.target_customer} die {ps.deliverable} volledig uit handen neemt, "
                   "met gratis AI-tools." if tech.key == "dienst"
                   else f"Tool voor {ps.target_customer} die {tech.solution_pattern} {ps.deliverable}.")[:300],
        problem=ps.problem,
        target_customer=ps.target_customer,
        sector=ps.sector,
        business_model=ps.typical_model,
        strategy=seed.strategy,
        differentiator=f"Toepassing van {tech.capability} op één specifiek probleem van deze doelgroep "
                       "(sjabloon; onderscheidend vermogen nog te onderzoeken).",
        generated_by="sjabloon",
        seed_key=seed.key,
    )


def founder_instructions(founder) -> str:
    if founder is None:
        return ""
    extra = (f"\n\nRandvoorwaarden van de oprichter: startkapitaal maximaal € {founder.capital_eur:g}, "
             f"{founder.hours_per_week:g} uur per week.")
    if founder.bootstrapped:
        extra += (" De oprichter is beginner en heeft geen budget voor advertenties of softwareontwikkeling. "
                  "Zoek ideeën die binnen 30 dagen eerste omzet kunnen opleveren: geen vergunning, geen voorraad, "
                  "geen personeel, niets bouwen vooraf. Een dienst die eerst manueel (met gratis tools en AI) "
                  "geleverd wordt en per maand gefactureerd wordt, is ideaal. Vermeld het woord 'dienstverlening' "
                  "in het verdienmodel als het om een dienst gaat.")
    return extra


class IdeaGenerator:
    def __init__(self, llm=None, founder=None) -> None:
        self.llm = llm
        self.founder = founder

    def generate(self, seeds: list[Seed], per_seed: int = 2) -> list[GeneratedCandidate]:
        out: list[GeneratedCandidate] = []
        for seed in seeds:
            if self.llm is not None:
                try:
                    out.extend(self._generate_llm(seed, per_seed))
                    continue
                except Exception as exc:  # provider down, limits, invalid output
                    log.warning("LLM-generatie mislukt voor %s (%s); sjabloon gebruikt.", seed.key, exc)
            if seed.template_capable:
                out.append(GeneratedCandidate(_template_candidate(seed), seed))
        return out

    def _generate_llm(self, seed: Seed, per_seed: int) -> list[GeneratedCandidate]:
        from kansenradar.llm import Tier

        prompt = (
            f"Opdracht ({seed.strategy.value}, sector: {seed.sector.value}):\n{seed.brief}\n\n"
            f"Geef maximaal {per_seed} ideeën."
        )
        batch = self.llm.structured(
            task="generate_ideas", tier=Tier.ANALYSIS, prompt=prompt, schema=_LLMIdeaBatch,
            system_extra=GENERATION_INSTRUCTIONS + founder_instructions(self.founder), max_tokens=4_000,
        )
        results = []
        for item in batch.ideas[:per_seed]:
            try:
                cand = IdeaCandidate(
                    title=item.title[:120], one_liner=item.one_liner[:300], problem=item.problem,
                    target_customer=item.target_customer, sector=seed.sector,
                    business_model=item.business_model, strategy=seed.strategy,
                    inspiration=item.inspiration, why_now=item.why_now,
                    differentiator=item.differentiator, generated_by="llm", seed_key=seed.key,
                )
            except ValueError as exc:
                log.info("Ongeldig idee overgeslagen (%s): %s", seed.key, exc)
                continue
            results.append(GeneratedCandidate(cand, seed, item.originality_self_rating))
        return results
