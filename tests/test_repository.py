from datetime import UTC, datetime

import pytest

from kansenradar.domain import Evidence, EvidenceType, Idea, IdeaStatus, OriginalityStrategy, Sector
from kansenradar.llm import RunBudget, TokenUsage
from kansenradar.storage import Repository, cosine, make_engine


@pytest.fixture
def repo():
    return Repository(make_engine("sqlite:///:memory:"))


def sample_idea(**kw):
    data = dict(
        title="Digitale werfdagboeken voor kleine aannemers",
        one_liner="App die het verplichte werfverslag automatisch opmaakt uit foto's en spraak.",
        problem="Kleine aannemers verliezen uren aan administratie van werven.",
        target_customer="Aannemers met 1-20 werknemers in Vlaanderen en Nederland",
        sector=Sector.TRADITIONAL,
        business_model="SaaS-abonnement per gebruiker",
        strategy=OriginalityStrategy.NEW_TECH_APPLICATION,
    )
    data.update(kw)
    return Idea(**data)


def test_idea_roundtrip(repo):
    idea = sample_idea()
    repo.save_idea(idea)
    loaded = repo.get_idea(idea.id)
    assert loaded == idea


def test_most_similar_ignores_duplicates_and_other_models(repo):
    a = sample_idea()
    b = sample_idea(title="Ander idee", status=IdeaStatus.DUPLICATE)
    c = sample_idea(title="Derde idee")
    repo.save_idea(a, [1.0, 0.0], "m1")
    repo.save_idea(b, [1.0, 0.0], "m1")
    repo.save_idea(c, [1.0, 0.0], "m2")
    hits = repo.most_similar([0.9, 0.1], "m1")
    assert [h.idea_id for h in hits] == [a.id]
    assert hits[0].similarity == pytest.approx(cosine([0.9, 0.1], [1.0, 0.0]))
    assert hits[0].status == IdeaStatus.GENERATED.value


def test_evidence_roundtrip_keeps_fact_rules(repo):
    idea = sample_idea()
    repo.save_idea(idea)
    fact = Evidence(
        claim="Feit", type=EvidenceType.FACT, topic="marktvraag", source_url="https://x.be",
        quote="citaat", verified=True, retrieved_at=datetime(2026, 10, 1, tzinfo=UTC),
    )
    hyp = Evidence.create(claim="Klanten betalen graag", type=EvidenceType.FACT, topic="prijs")
    repo.add_evidence(idea.id, [fact, hyp])
    types = sorted(e.type.value for e in repo.evidence_for(idea.id))
    assert types == ["FEIT", "HYPOTHESE"]


def test_run_costs_are_persisted(repo):
    run_id = repo.start_run(max_cost_usd=5.0)
    budget = RunBudget(5.0)
    budget.record("screen", "claude-haiku-5-5", TokenUsage(input_tokens=10_000, output_tokens=1_000))
    repo.finish_run(run_id, "klaar", budget.entries)
    from kansenradar.storage.models import RunRow
    with repo.session() as s:
        run = s.get(RunRow, run_id)
        assert run.status == "klaar"
        assert run.spent_usd == pytest.approx(budget.spent_usd)
