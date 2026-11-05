"""Originality engine: systematic seeds for idea generation + originality label.

A seed is a concrete creative brief (strategy x sector x lens). Seeds are drawn
round-robin across strategies for diversity, reproducibly (fixed RNG seed), and
seeds already used in earlier runs are skipped so the history keeps expanding.

The originality score is a separate label (0-10). It is deliberately NOT part
of the 100-point commercial score: originality must never outweigh viability.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations

from kansenradar.domain import IdeaCandidate, OriginalityAssessment, OriginalityStrategy, Sector
from kansenradar.generation.catalog import (
    INTERNATIONAL_PATTERNS,
    PROBLEM_SEEDS,
    REGULATORY_TRIGGERS,
    TECH_LENSES,
    ProblemSeed,
    TechLens,
)
from kansenradar.history.embeddings import Thresholds

OS = OriginalityStrategy


@dataclass(frozen=True)
class Seed:
    key: str
    strategy: OriginalityStrategy
    sector: Sector
    brief: str
    problem: ProblemSeed | None = None
    tech: TechLens | None = None

    @property
    def template_capable(self) -> bool:
        return self.problem is not None and self.tech is not None


def _all_seeds() -> list[Seed]:
    seeds: list[Seed] = []
    for ps in PROBLEM_SEEDS:
        seeds.append(Seed(
            f"prob:{ps.key}", OS.PROBLEM_DRIVEN, ps.sector,
            f"Vertrek van dit (te verifiëren) probleem: {ps.problem} Doelgroep: {ps.target_customer}. "
            "Bedenk een oplossing die dit probleem fundamenteel wegneemt.", problem=ps))
        seeds.append(Seed(
            f"niche:{ps.key}", OS.UNUSUAL_NICHE, ps.sector,
            f"Startpunt: {ps.problem} Zoek binnen '{ps.target_customer}' een kleinere, ongebruikelijke "
            "subniche met een acuter probleem en hogere betalingsbereidheid.", problem=ps))
        for tech in (t for t in TECH_LENSES if t.key in ps.tech_fit):
            seeds.append(Seed(
                f"tech:{ps.key}:{tech.key}", OS.NEW_TECH_APPLICATION, ps.sector,
                f"Pas '{tech.capability}' toe op dit probleem: {ps.problem} Doelgroep: {ps.target_customer}.",
                problem=ps, tech=tech))
    for pattern in INTERNATIONAL_PATTERNS:
        for sector in Sector:
            seeds.append(Seed(
                f"intl:{pattern.key}:{sector.name}", OS.INTERNATIONAL_ARBITRAGE, sector,
                f"Businessmodel dat in {pattern.region} lijkt te werken (HYPOTHESE, te verifiëren): "
                f"{pattern.pattern}. Pas dit toe in de sector '{sector.value}' in België/Nederland en leg "
                "uit waarom het daar nog onderontwikkeld zou zijn."))
    for a, b in combinations(list(Sector), 2):
        seeds.append(Seed(
            f"combo:{a.name}:{b.name}", OS.MARKET_COMBINATION, a,
            f"Combineer de markten '{a.value}' en '{b.value}' op een onverwachte maar commercieel "
            "logische manier (bv. de klanten van de ene markt bedienen met de middelen van de andere)."))
    for sector in Sector:
        seeds.append(Seed(
            f"improve:{sector.name}", OS.FUNDAMENTAL_IMPROVEMENT, sector,
            f"Kies een gangbare, bestaande oplossing in '{sector.value}' en verbeter ze fundamenteel "
            "(bv. tien keer sneller, goedkoper of eenvoudiger), niet incrementeel."))
    for trig in REGULATORY_TRIGGERS:
        for sector in Sector:
            seeds.append(Seed(
                f"reg:{trig.key}:{sector.name}", OS.PROBLEM_DRIVEN, sector,
                f"Regelgevende trigger (HYPOTHESE, exacte inhoud en data te verifiëren): {trig.description}. "
                f"Welk nieuw, concreet probleem ontstaat hierdoor in '{sector.value}' "
                f"(markten: {', '.join(trig.markets)})?"))
    return seeds


class OriginalityEngine:
    def __init__(self) -> None:
        self.catalog = _all_seeds()

    def seeds(
        self,
        count: int,
        sectors: list[Sector] | None = None,
        strategies: list[OriginalityStrategy] | None = None,
        exclude_keys: set[str] | None = None,
        rng_seed: int = 0,
        template_only: bool = False,
    ) -> list[Seed]:
        """Pick diverse seeds. At most one seed per catalog problem per run. In template
        mode a problem used in an earlier run is skipped entirely, because template
        ideas for the same problem are near-identical.
        """
        exclude_keys = exclude_keys or set()
        used_problems = {k.split(":")[1] for k in exclude_keys
                         if k.startswith(("tech:", "prob:", "niche:"))}
        pool = [
            s for s in self.catalog
            if s.key not in exclude_keys
            and (not sectors or s.sector in sectors)
            and (not strategies or s.strategy in strategies)
            and (not template_only or (s.template_capable and s.problem.key not in used_problems))
        ]
        rng = random.Random(rng_seed)
        by_strategy: dict[OriginalityStrategy, list[Seed]] = {}
        for s in pool:
            by_strategy.setdefault(s.strategy, []).append(s)
        for group in by_strategy.values():
            rng.shuffle(group)
        order = sorted(by_strategy, key=lambda st: st.value)
        picked: list[Seed] = []
        problems_in_run: set[str] = set()
        while len(picked) < count and any(by_strategy.values()):
            for st in order:
                while by_strategy[st] and len(picked) < count:
                    seed = by_strategy[st].pop()
                    if seed.problem and seed.problem.key in problems_in_run:
                        continue
                    if seed.problem:
                        problems_in_run.add(seed.problem.key)
                    picked.append(seed)
                    break
        return picked


_STRATEGY_WEIGHT = {
    OS.MARKET_COMBINATION: 1.0,
    OS.INTERNATIONAL_ARBITRAGE: 0.8,
    OS.UNUSUAL_NICHE: 0.8,
    OS.NEW_TECH_APPLICATION: 0.7,
    OS.FUNDAMENTAL_IMPROVEMENT: 0.6,
    OS.PROBLEM_DRIVEN: 0.5,
}


def assess_originality(
    candidate: IdeaCandidate,
    similarity_to_history: float | None,
    thresholds: Thresholds,
    llm_rating: float | None = None,
) -> OriginalityAssessment:
    """Combine novelty vs. our own history, strategy type and (optional) LLM rating.

    Novelty is relative to the embedder's duplicate threshold, so the scale is
    comparable across embedders. Note: novelty vs. our history says nothing about
    novelty vs. the market; competitor research (phase 5) covers that.
    """
    sim = similarity_to_history or 0.0
    novelty = max(0.0, 1.0 - min(1.0, sim / thresholds.duplicate))
    weight = _STRATEGY_WEIGHT[candidate.strategy]
    if llm_rating is not None:
        score = 10 * (0.5 * novelty + 0.2 * weight + 0.3 * llm_rating / 10)
    else:
        score = 10 * (0.7 * novelty + 0.3 * weight)
    score = round(min(10.0, max(0.0, score)), 1)
    label = "laag" if score < 4 else "gemiddeld" if score < 7 else "hoog"
    parts = [f"nieuwheid t.o.v. eerdere ideeën {novelty:.2f}", f"strategie '{candidate.strategy.value}'"]
    if llm_rating is not None:
        parts.append(f"inschatting model {llm_rating:.0f}/10")
    return OriginalityAssessment(
        score=score, label=label, novelty_vs_history=round(novelty, 3), strategy=candidate.strategy,
        llm_rating=llm_rating,
        explanation="Gebaseerd op " + ", ".join(parts) + ". Telt niet mee in de commerciële score.",
    )
