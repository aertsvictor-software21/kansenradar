"""Generation run: seeds -> ideas -> generic filter -> dedup -> originality -> store."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from kansenradar.domain import Idea, IdeaStatus, OriginalityStrategy, Sector
from kansenradar.generation.generator import IdeaGenerator
from kansenradar.generation.originality import OriginalityEngine, assess_originality
from kansenradar.generation.quality import check_quality
from kansenradar.history import DuplicateChecker, HeuristicJudge, LLMJudge
from kansenradar.history.embeddings import Embedder
from kansenradar.llm import RunBudget
from kansenradar.storage import Repository


@dataclass
class GenerationReport:
    run_id: str
    mode: str
    seeds_used: int
    ideas: list[Idea] = field(default_factory=list)

    @property
    def counts(self) -> Counter:
        return Counter(i.status for i in self.ideas)

    @property
    def new_ideas(self) -> list[Idea]:
        return [i for i in self.ideas if i.status is IdeaStatus.GENERATED]

    def summary(self) -> str:
        c = self.counts
        if self.seeds_used == 0:
            return (f"Run {self.run_id} ({self.mode}): geen nieuwe invalshoeken meer beschikbaar.\n"
                    + ("  De sjablooncatalogus is uitgeput. Stel een gratis LLM-provider in "
                       "(zie README) of breid de catalogus uit." if self.mode == "sjabloon"
                       else "  Alle invalshoeken voor deze selectie zijn al gebruikt."))
        lines = [
            f"Run {self.run_id} ({self.mode}): {self.seeds_used} invalshoeken, {len(self.ideas)} ideeën.",
            f"  Nieuw en door naar screening: {c[IdeaStatus.GENERATED]}",
            f"  Afgewezen als generiek:       {c[IdeaStatus.REJECTED_GENERIC]}",
            f"  Duplicaat van eerder idee:    {c[IdeaStatus.DUPLICATE]}",
        ]
        related = sum(1 for i in self.new_ideas if i.similar_to)
        if related:
            lines.append(f"  (waarvan verwant aan eerder idee: {related})")
        return "\n".join(lines)


def run_generation(
    repo: Repository,
    embedder: Embedder,
    llm=None,
    count: int = 10,
    per_seed: int = 2,
    sectors: list[Sector] | None = None,
    strategies: list[OriginalityStrategy] | None = None,
    rng_seed: int = 0,
    budget: RunBudget | None = None,
    founder=None,
) -> GenerationReport:
    engine = OriginalityEngine()
    used = repo.used_seed_keys()
    seeds = []
    if count <= 0:  # e.g. 'run --aantal 0': only continue with existing ideas
        founder = None
    if founder is not None and founder.bootstrapped:
        # Services first: they fit a founder with almost no capital.
        seeds = [sd for sd in engine.catalog
                 if sd.tech is not None and sd.tech.key == "dienst" and sd.key not in used][: max(1, count // 2)]
    seeds += [] if count <= 0 else engine.seeds(
        count - len(seeds), sectors=sectors, strategies=strategies,
        exclude_keys=used | {sd.key for sd in seeds}, rng_seed=rng_seed, template_only=llm is None,
    )
    run_id = repo.start_run(
        max_cost_usd=budget.max_usd if budget else 0.0,
        params={"count": count, "per_seed": per_seed, "mode": "llm" if llm else "sjabloon",
                "sectors": [s.name for s in sectors or []],
                "strategies": [s.name for s in strategies or []], "rng_seed": rng_seed},
    )
    judge = LLMJudge(llm, HeuristicJudge(embedder)) if llm else HeuristicJudge(embedder)
    checker = DuplicateChecker(repo, embedder, judge)
    report = GenerationReport(run_id=run_id, mode="llm" if llm else "sjabloon", seeds_used=len(seeds))

    status = "klaar"
    try:
        for gen in IdeaGenerator(llm, founder).generate(seeds, per_seed=per_seed):
            quality = check_quality(gen.candidate)
            idea, check = checker.register(gen.candidate, run_id=run_id)
            idea.quality_flags = quality.flags
            idea.originality = assess_originality(
                gen.candidate, check.similarity, embedder.thresholds, gen.llm_rating
            )
            if idea.status is not IdeaStatus.DUPLICATE and quality.is_generic:
                idea.status = IdeaStatus.REJECTED_GENERIC
            repo.save_idea(idea)
            report.ideas.append(idea)
    except Exception:
        status = "mislukt"
        raise
    finally:
        repo.finish_run(run_id, status, budget.entries if budget else [])
    return report
