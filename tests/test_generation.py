from collections import Counter

import pytest

from kansenradar.cli import main as cli_main
from kansenradar.domain import IdeaCandidate, IdeaStatus, OriginalityStrategy, Sector
from kansenradar.generation import (
    IdeaGenerator,
    OriginalityEngine,
    assess_originality,
    check_quality,
    run_generation,
)
from kansenradar.generation.catalog import PROBLEM_SEEDS, TECH_LENSES
from kansenradar.generation.generator import _LLMIdea, _LLMIdeaBatch
from kansenradar.history import HashingEmbedder
from kansenradar.history.dedup import _LLMJudgement
from kansenradar.llm import FakeBackend, LLMClient, RunBudget
from kansenradar.storage import Repository, make_engine

OS = OriginalityStrategy


@pytest.fixture
def repo():
    return Repository(make_engine("sqlite:///:memory:"))


def good_candidate(**kw):
    data = dict(
        title="Werfverslag uit spraak",
        one_liner="App die ingesproken notities omzet in werfverslagen en meerwerkfacturen.",
        problem="Kleine aannemers verliezen elke week avonden aan werfverslagen, meerwerk en facturatie.",
        target_customer="aannemers met 1-20 werknemers in Vlaanderen",
        sector=Sector.TRADITIONAL, business_model="abonnement per gebruiker",
        strategy=OS.NEW_TECH_APPLICATION, differentiator="Gebouwd rond de werfroutine, niet rond kantoorsoftware.",
        generated_by="llm",
    )
    data.update(kw)
    return IdeaCandidate(**data)


# --- catalog & engine ----------------------------------------------------------------

def test_catalog_integrity():
    tech_keys = {t.key for t in TECH_LENSES}
    assert len({p.key for p in PROBLEM_SEEDS}) == len(PROBLEM_SEEDS)
    for p in PROBLEM_SEEDS:
        assert p.deliverable and p.tech_fit and set(p.tech_fit) <= tech_keys
    assert len({s.key for s in OriginalityEngine().catalog}) == len(OriginalityEngine().catalog)


def test_every_sector_has_a_problem_seed():
    assert {p.sector for p in PROBLEM_SEEDS} == set(Sector)


def test_seeds_are_diverse_reproducible_and_one_per_problem():
    engine = OriginalityEngine()
    seeds = engine.seeds(12, rng_seed=3)
    assert [s.key for s in seeds] == [s.key for s in engine.seeds(12, rng_seed=3)]
    assert len(Counter(s.strategy for s in seeds)) == 6  # all strategies represented
    problems = [s.problem.key for s in seeds if s.problem]
    assert len(problems) == len(set(problems))


def test_seed_filters_and_exclusion():
    engine = OriginalityEngine()
    seeds = engine.seeds(5, sectors=[Sector.LOGISTICS], strategies=[OS.NEW_TECH_APPLICATION])
    assert seeds and all(s.sector is Sector.LOGISTICS and s.strategy is OS.NEW_TECH_APPLICATION for s in seeds)
    again = engine.seeds(5, sectors=[Sector.LOGISTICS], strategies=[OS.NEW_TECH_APPLICATION],
                         exclude_keys={s.key for s in seeds})
    assert not {s.key for s in seeds} & {s.key for s in again}


def test_template_mode_skips_used_problems():
    engine = OriginalityEngine()
    first = engine.seeds(100, template_only=True)
    assert len(first) == len(PROBLEM_SEEDS)
    assert all(s.template_capable for s in first)
    assert engine.seeds(100, template_only=True, exclude_keys={s.key for s in first}) == []


# --- quality filter ----------------------------------------------------------------------

def test_specific_idea_passes_quality():
    q = check_quality(good_candidate())
    assert not q.is_generic and q.flags == []


@pytest.mark.parametrize("change, flag", [
    (dict(target_customer="Bedrijven"), "doelgroep te breed"),
    (dict(markets=["DE"]), "niet gericht op België of Nederland"),
])
def test_hard_generic_signals(change, flag):
    q = check_quality(good_candidate(**change))
    assert q.is_generic and any(f.startswith(flag) for f in q.flags)


def test_two_soft_signals_make_generic():
    q = check_quality(good_candidate(one_liner="Het Uber voor aannemers: een revolutionair platform voor werven."))
    assert q.is_generic and len(q.flags) >= 2
    one = check_quality(good_candidate(one_liner="Een disruptief hulpmiddel dat werfverslagen maakt uit spraak."))
    assert not one.is_generic and len(one.flags) == 1


# --- originality --------------------------------------------------------------------------

def test_originality_prefers_novel_and_combination_ideas():
    t = HashingEmbedder.thresholds
    novel = assess_originality(good_candidate(strategy=OS.MARKET_COMBINATION), 0.05, t)
    close = assess_originality(good_candidate(strategy=OS.PROBLEM_DRIVEN), 0.7, t)
    assert novel.score > close.score
    assert novel.label == "hoog" and close.label == "laag"
    assert "Telt niet mee" in novel.explanation
    with_llm = assess_originality(good_candidate(), 0.05, t, llm_rating=2)
    assert with_llm.llm_rating == 2 and with_llm.score < assess_originality(good_candidate(), 0.05, t, 10).score


# --- generator -----------------------------------------------------------------------------

def llm_idea(**kw):
    data = dict(
        title="Werfverslag uit spraak", one_liner="App die ingesproken notities omzet in werfverslagen.",
        problem="Kleine aannemers verliezen elke week avonden aan werfverslagen en meerwerkfacturen.",
        target_customer="aannemers met 1-20 werknemers", business_model="abonnement",
        why_now="Spraakherkenning in het Nederlands is goedkoop en goed geworden (hypothese).",
        differentiator="Werkt offline op de werf.", originality_self_rating=6,
    )
    data.update(kw)
    return _LLMIdea(**data)


def make_llm(handlers):
    return LLMClient(FakeBackend(handlers), RunBudget(max_calls=50), free=True)


def test_llm_generation_maps_fields_and_skips_invalid():
    batch = _LLMIdeaBatch(ideas=[llm_idea(), llm_idea(title="x"), llm_idea(title="Derde", problem="Nog een probleem van aannemers met meerwerk.")])
    gen = IdeaGenerator(make_llm({"generate_ideas": batch}))
    seed = OriginalityEngine().seeds(1, strategies=[OS.UNUSUAL_NICHE])[0]
    out = gen.generate([seed], per_seed=3)
    assert len(out) == 2  # title "x" too short -> skipped
    c = out[0].candidate
    assert c.generated_by == "llm" and c.seed_key == seed.key and c.strategy is OS.UNUSUAL_NICHE
    assert out[0].llm_rating == 6


def test_llm_failure_falls_back_to_template():
    def down(_):
        raise ConnectionError("provider offline")
    seeds = OriginalityEngine().seeds(2, template_only=True)
    out = IdeaGenerator(make_llm({"generate_ideas": down})).generate(seeds)
    assert [g.candidate.generated_by for g in out] == ["sjabloon", "sjabloon"]


# --- pipeline -------------------------------------------------------------------------------

def test_template_pipeline_end_to_end(repo):
    report = run_generation(repo, HashingEmbedder(), llm=None, count=10)
    assert report.mode == "sjabloon" and len(report.ideas) == 10
    assert all(i.originality is not None for i in report.ideas)
    assert report.counts[IdeaStatus.DUPLICATE] == 0
    assert len(repo.used_seed_keys()) == 10
    second = run_generation(repo, HashingEmbedder(), llm=None, count=100)
    assert len(second.ideas) == len(PROBLEM_SEEDS) - 10
    exhausted = run_generation(repo, HashingEmbedder(), llm=None, count=5)
    assert exhausted.seeds_used == 0 and "uitgeput" in exhausted.summary()


def test_llm_pipeline_rejects_generic_and_duplicates(repo):
    def ideas(prompt):
        return _LLMIdeaBatch(ideas=[
            llm_idea(),
            llm_idea(title="Platform voor iedereen", target_customer="iedereen"),
        ])
    llm = make_llm({
        "generate_ideas": ideas,
        "dedup_judge": _LLMJudgement(same_idea=True, related=True, explanation="Zelfde idee."),
    })
    report = run_generation(repo, HashingEmbedder(), llm=llm, count=2, per_seed=2,
                            budget=llm.budget)
    statuses = [i.status for i in report.ideas]
    assert statuses[0] is IdeaStatus.GENERATED
    assert IdeaStatus.REJECTED_GENERIC in statuses or IdeaStatus.DUPLICATE in statuses
    # second seed returns the same first idea -> duplicate of the stored one
    assert IdeaStatus.DUPLICATE in statuses
    assert report.mode == "llm" and llm.budget.spent_usd == 0.0


def test_cli_generate_and_show(tmp_path, monkeypatch, capsys):
    from kansenradar.config import get_settings
    monkeypatch.setenv("KR_DATABASE_URL", f"sqlite:///{tmp_path / 'kr.db'}")
    monkeypatch.setenv("KR_EMBEDDER", "hashing")
    monkeypatch.setenv("KR_LLM_PROVIDER", "none")
    get_settings.cache_clear()
    try:
        assert cli_main(["genereer", "--aantal", "3"]) == 0
        out = capsys.readouterr().out
        assert "sjablonen (geen LLM)" in out and "Nieuw en door naar screening: 3" in out
        idea_id = next(w for w in out.split() if w.startswith("idea_"))
        assert cli_main(["toon", idea_id]) == 0
        assert "Originaliteit:" in capsys.readouterr().out
        assert cli_main(["genereer", "--aantal", "1", "--sector", "logistiek"]) == 0
    finally:
        get_settings.cache_clear()
