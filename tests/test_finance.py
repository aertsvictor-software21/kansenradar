from datetime import UTC, datetime

import pytest

from kansenradar.domain import (
    CompetitionAnalysis,
    Evidence,
    EvidenceType,
    Idea,
    OriginalityStrategy,
    ResearchTopic,
    Sector,
)
from kansenradar.finance import build_financial_model, revenue_model, simulate
from kansenradar.screening import screen_by_rules

BASE = dict(price=100, cac=100, marketing=600, organic=0, churn=0, margin=100, fixed=0, startup=0,
            market=1_000_000, penetration=100)


def idea(**kw):
    data = dict(title="Werfverslag uit spraak", one_liner="App voor werfverslagen.",
                problem="Kleine aannemers verliezen avonden aan werfverslagen en administratie.",
                target_customer="aannemers met 1-20 werknemers", sector=Sector.TRADITIONAL,
                business_model="abonnement per gebruiker", strategy=OriginalityStrategy.NEW_TECH_APPLICATION)
    data.update(kw)
    return Idea(**data)


def test_hand_computed_ramp_and_break_even():
    r = simulate(BASE, "x")
    # new customers months 1..6: 1,2,3,4,5,6 (ramp), then 6/month
    assert r.customers_m12 == 21 + 6 * 6
    assert r.revenue_m12 == pytest.approx(5700)
    # net(t) = customers*100 - 600 -> month 3: 6*100-600 = 0
    assert r.operational_break_even_month == 3
    assert r.ltv == pytest.approx(100 / 0.01)  # churn floored at 1%


def test_capital_needed_and_payback():
    r = simulate({**BASE, "startup": 1000}, "x")
    # cumulative: -1000, -1500, -1800, -1800, -1400, ... -> deepest -1800
    assert r.capital_needed == pytest.approx(1800)
    # cumulative: -1500, -1800, -1800, -1400, -500, +1000 -> paid back in month 6
    assert r.payback_month == 6


def test_market_ceiling_caps_customers():
    r = simulate({**BASE, "market": 100, "penetration": 10}, "x")
    assert r.customers_m36 == 10
    assert r.market_cap_reached_month == 5


def test_never_break_even():
    r = simulate({**BASE, "price": 1, "fixed": 5000}, "x")
    assert r.operational_break_even_month is None and r.payback_month is None


def test_scenarios_take_unfavourable_side_when_conservative():
    m = build_financial_model(idea(), [])
    c, o = m.scenarios["conservatief"], m.scenarios["optimistisch"]
    assert c.price_per_month < o.price_per_month
    assert c.cac > o.cac and c.startup_capital > o.startup_capital
    assert c.revenue_year3 < m.scenarios["realistisch"].revenue_year3 < o.revenue_year3
    assert all(a.basis is EvidenceType.HYPOTHESIS for a in m.assumptions)
    assert any("vóór vergoeding" in n for n in m.notes)
    assert any("niet onderbouwd" in n for n in m.notes)


def test_verified_data_replaces_rules_of_thumb():
    comp = CompetitionAnalysis(mode="t", competitors=[], saturation="gemiddeld", saturation_reason="",
                               price_min_month=29, price_median_month=45, price_max_month=59,
                               verified_price_count=3)
    facts = [Evidence(claim="Eurostat-telling", type=EvidenceType.FACT, topic=ResearchTopic.MARKET_SIZE.value,
                      source_url="https://ec.europa.eu/x", quote=f"ENT_NR {cc} F43 2023 = {n}",
                      verified=True, retrieved_at=datetime(2026, 1, 1, tzinfo=UTC))
             for cc, n in (("BE", 80_000), ("NL", 120_000))]
    m = build_financial_model(idea(competition=comp), facts)
    price, market = m.assumption("price"), m.assumption("market")
    assert (price.low, price.mid, price.high) == (29, 45, 59) and price.basis is EvidenceType.ESTIMATE
    assert market.mid == 70_000 and market.basis is EvidenceType.ESTIMATE
    assert not any("niet onderbouwd" in n for n in m.notes)


def test_regulation_raises_startup_capital():
    regulated = idea(title="Kredietapp voor aannemers", one_liner="App die kredieten regelt.",
                     problem="Aannemers verliezen omzet door trage kredietaanvragen bij banken.")
    regulated.screening = screen_by_rules(regulated)
    m = build_financial_model(regulated, [])
    assert m.assumption("startup").mid >= 12_000
    assert any("Regelgeving" in n for n in m.notes)


def test_revenue_model_detection_and_one_time():
    assert revenue_model(idea()) == "abonnement_b2b"
    assert revenue_model(idea(target_customer="jonge ouders")) == "abonnement_b2c"
    assert revenue_model(idea(business_model="commissie per rit")) == "transactie"
    one = build_financial_model(idea(business_model="betaald rapport plus commissie van aannemers"), [])
    assert one.revenue_model == "transactie"
    once = build_financial_model(idea(business_model="eenmalige aankoop plus updates"), [])
    assert once.revenue_model == "eenmalig"
    r = once.scenarios["realistisch"]
    assert r.cac_payback_months is None and r.customers_m36 < 60  # no recurring base
