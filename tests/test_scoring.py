from datetime import UTC, datetime

from kansenradar.domain import (
    CompetitionAnalysis,
    Conclusion,
    Evidence,
    EvidenceType,
    Idea,
    IdeaStatus,
    OriginalityStrategy,
    ResearchSummary,
    ResearchTopic,
    Sector,
    SourceQuality,
    TopicCoverage,
)
from kansenradar.finance import build_financial_model
from kansenradar.research.researcher import coverage
from kansenradar.review import review_by_rules
from kansenradar.scoring import rank, score_idea, score_ideas
from kansenradar.screening import screen_by_rules
from kansenradar.storage import Repository, make_engine

NOW = datetime(2026, 10, 1, tzinfo=UTC)


def fact(topic, quality=SourceQuality.OFFICIAL, quote="citaat met 12 woorden uit de bron"):
    return Evidence(claim="Geverifieerd feit", type=EvidenceType.FACT, topic=topic.value,
                    source_url="https://bron.be", quote=quote, verified=True, retrieved_at=NOW,
                    source_quality=quality)


def build(evidence=(), saturation=None, prices=0, **kw):
    data = dict(title="Werfverslag uit spraak", one_liner="App die werfverslagen maakt uit spraak.",
                problem="Kleine aannemers verliezen avonden aan werfverslagen en lopen omzet mis.",
                target_customer="aannemers met 1-20 werknemers", sector=Sector.TRADITIONAL,
                business_model="abonnement per gebruiker", strategy=OriginalityStrategy.NEW_TECH_APPLICATION,
                generated_by="llm", differentiator="Integratie met bestaande bouwsoftware en eigen data.",
                why_now="Spraakherkenning in het Nederlands is goed en goedkoop geworden.",
                status=IdeaStatus.RESEARCHED)
    data.update(kw)
    i = Idea(**data)
    i.screening = screen_by_rules(i)
    evidence = list(evidence)
    i.research = ResearchSummary(mode="t", coverage=coverage(evidence, 365))
    if saturation:
        i.competition = CompetitionAnalysis(mode="zoeken", competitors=[], saturation=saturation,
                                            saturation_reason="", verified_price_count=prices,
                                            price_min_month=29 if prices else None,
                                            price_median_month=45 if prices else None,
                                            price_max_month=59 if prices else None)
    i.finance = build_financial_model(i, evidence)
    i.review = review_by_rules(i)
    return i, evidence


def test_points_add_up_and_respect_maxima():
    i, ev = build()
    sc = score_idea(i, ev)
    assert sum(s.max_points for s in sc.subscores) == 100
    assert sc.total == round(sum(s.points for s in sc.subscores), 1)
    assert all(0 <= s.points <= s.max_points and s.rationale for s in sc.subscores)


def test_without_evidence_caps_apply_and_conclusion_is_more_research():
    i, ev = build()
    sc = score_idea(i, ev)
    assert sc.sub("Marktvraag").points <= 8 and sc.sub("Marktvraag").capped_by
    assert sc.sub("Winstgevendheid").points <= 7
    assert sc.sub("Onderscheidend vermogen").points <= 7
    assert sc.confidence < 35
    assert sc.conclusion in (Conclusion.MORE_RESEARCH, Conclusion.REJECT)


def test_evidence_raises_confidence_and_lifts_caps():
    topics = [ResearchTopic.DEMAND, ResearchTopic.PROBLEM, ResearchTopic.MARKET_SIZE, ResearchTopic.PRICING,
              ResearchTopic.COMPETITORS, ResearchTopic.REACH, ResearchTopic.TRENDS, ResearchTopic.CUSTOMERS]
    ev = [fact(t) for t in topics]
    weak_i, weak_ev = build()
    strong_i, strong_ev = build(ev, saturation="laag", prices=3)
    weak, strong = score_idea(weak_i, weak_ev), score_idea(strong_i, strong_ev)
    assert strong.confidence > 65 > weak.confidence
    assert strong.sub("Marktvraag").capped_by is None
    assert strong.total > weak.total
    assert strong.conclusion in (Conclusion.PROMISING, Conclusion.VERY_INTERESTING, Conclusion.EXPERIMENT)


def test_company_sources_count_less_than_official():
    topics = list(ResearchTopic)[:6]
    official, _ = build([fact(t) for t in topics], saturation="gemiddeld")
    company, _ = build([fact(t, SourceQuality.COMPANY) for t in topics], saturation="gemiddeld")
    a = score_idea(official, [fact(t) for t in topics]).confidence
    b = score_idea(company, [fact(t, SourceQuality.COMPANY) for t in topics]).confidence
    assert a - b == 10


def test_fatal_flaw_rejects_regardless_of_score():
    ev = [fact(t) for t in ResearchTopic]
    i, _ = build(ev, saturation="hoog", prices=3)
    assert i.review.fatal_flaws
    sc = score_idea(i, ev)
    assert sc.conclusion is Conclusion.REJECT and sc.conclusion_reason.startswith("Fataal gebrek")


def test_outdated_facts_do_not_count():
    old = [Evidence(claim="Oud feit hier", type=EvidenceType.FACT, topic=ResearchTopic.DEMAND.value,
                    source_url="https://b.be", quote="oud citaat uit de bron hier", verified=True,
                    retrieved_at=NOW, published_at=datetime(2020, 1, 1, tzinfo=UTC))]
    i, _ = build(old)
    assert score_idea(i, old).sub("Marktvraag").capped_by is not None


def test_rank_orders_by_conclusion_then_score_and_persists():
    repo = Repository(make_engine("sqlite:///:memory:"))
    strong, sev = build([fact(t) for t in list(ResearchTopic)[:8]], saturation="laag", prices=3, title="Sterk idee")
    weak, _ = build(title="Zwak idee")
    repo.save_idea(strong)
    repo.add_evidence(strong.id, sev)
    repo.save_idea(weak)
    ranked = score_ideas(repo)
    assert [i.title for i in ranked] == ["Sterk idee", "Zwak idee"]
    assert repo.get_idea(strong.id).status is IdeaStatus.ANALYSED
    from kansenradar.storage.models import IdeaRow
    with repo.session() as s:
        row = s.get(IdeaRow, strong.id)
        assert row.conclusion == ranked[0].scorecard.conclusion.value and row.score_total > 0
    assert rank([weak]) == []  # unscored ideas are not ranked
