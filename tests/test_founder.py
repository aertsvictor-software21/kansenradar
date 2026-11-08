from kansenradar.config import FounderProfile, Settings, founder_from
from kansenradar.domain import Idea, IdeaStatus, OriginalityStrategy, Sector
from kansenradar.finance import build_financial_model, revenue_model
from kansenradar.generation import run_generation
from kansenradar.generation.generator import founder_instructions
from kansenradar.history import HashingEmbedder
from kansenradar.review import review_by_rules
from kansenradar.scoring import score_idea
from kansenradar.screening import screen_by_rules
from kansenradar.storage import Repository, make_engine

FOUNDER = FounderProfile(capital_eur=50, hours_per_week=20)


def service(**kw):
    data = dict(title="Reviewbeheer voor lokale handelaars",
                one_liner="Maandelijkse dienst die Google-profiel en reviews van lokale zaken beheert.",
                problem="Lokale handelaars verliezen klanten omdat hun Google-bedrijfsprofiel verouderd is en reviews onbeantwoord blijven.",
                target_customer="lokale handelaars (kappers, garages, restaurants)", sector=Sector.PROFESSIONAL_SERVICES,
                business_model="maandelijkse dienstverlening per zaak", strategy=OriginalityStrategy.PROBLEM_DRIVEN,
                status=IdeaStatus.RESEARCHED)
    data.update(kw)
    i = Idea(**data)
    i.screening = screen_by_rules(i)
    return i


def test_founder_from_settings(monkeypatch):
    # The user's own environment may already contain a founder profile.
    monkeypatch.delenv("KR_FOUNDER_CAPITAL_EUR", raising=False)
    monkeypatch.delenv("KR_FOUNDER_HOURS_PER_WEEK", raising=False)
    assert founder_from(Settings(_env_file=None)) is None
    f = founder_from(Settings(founder_capital_eur=50, founder_hours_per_week=10))
    assert f.capital_eur == 50 and f.bootstrapped


def test_service_model_is_cheap_and_capacity_limited():
    i = service()
    assert revenue_model(i) == "dienst"
    m = build_financial_model(i, [], founder=FOUNDER)
    r = m.scenarios["realistisch"]
    assert r.startup_capital <= 50 and r.marketing_per_month == 0
    assert r.customers_m36 <= round(20 / 3)  # capacity of the founder
    assert m.scalability == "laag"
    assert any("eigen uren" in n for n in m.notes) and any("geen advertentiebudget" in n for n in m.notes)


def test_capital_scored_against_founder_budget():
    svc = service()
    svc.finance = build_financial_model(svc, [], founder=FOUNDER)
    svc.review = review_by_rules(svc)
    software = service(title="Reviewsoftware", business_model="abonnement per gebruiker")
    software.finance = build_financial_model(software, [], founder=FOUNDER)
    software.review = review_by_rules(software)
    a = score_idea(svc, [], founder=FOUNDER).sub("Benodigd startkapitaal")
    b = score_idea(software, [], founder=FOUNDER).sub("Benodigd startkapitaal")
    assert a.points == 10 and "haalbaar" in a.rationale
    assert b.points < 2 and "onvoldoende" in b.rationale


def test_service_gets_first_paying_client_test():
    i = service()
    i.finance = build_financial_model(i, [], founder=FOUNDER)
    v = review_by_rules(i).validation_test
    assert v.method.startswith("Eerste betalende klant") and "2 betalende klanten" in v.success_criteria[0]


def test_bootstrapped_generation_prefers_services_and_constrains_llm_prompt():
    repo = Repository(make_engine("sqlite:///:memory:"))
    report = run_generation(repo, HashingEmbedder(), llm=None, count=6, founder=FOUNDER)
    keys = [i.seed_key for i in report.ideas]
    assert sum(k.endswith(":dienst") for k in keys) >= 3
    assert "30 dagen" in founder_instructions(FOUNDER) and "€ 50" in founder_instructions(FOUNDER)
    assert founder_instructions(None) == ""
