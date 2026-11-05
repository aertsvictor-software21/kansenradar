from kansenradar.domain import (
    CompetitionAnalysis,
    Competitor,
    EvidenceType,
    Idea,
    InvestorQuestion,
    OriginalityStrategy,
    ResearchSummary,
    ResearchTopic,
    Sector,
    TopicCoverage,
)
from kansenradar.finance import build_financial_model
from kansenradar.llm import FakeBackend, LLMClient, RunBudget
from kansenradar.review import review_by_rules, review_idea
from kansenradar.review.llm import _LLMCheck, _LLMReview, _LLMRisk
from kansenradar.screening import screen_by_rules

Q = InvestorQuestion


def idea(**kw) -> Idea:
    data = dict(title="Werfverslag uit spraak", one_liner="App die werfverslagen maakt uit spraak.",
                problem="Kleine aannemers verliezen avonden aan werfverslagen en administratie.",
                target_customer="aannemers met 1-20 werknemers", sector=Sector.TRADITIONAL,
                business_model="abonnement per gebruiker", strategy=OriginalityStrategy.NEW_TECH_APPLICATION,
                generated_by="sjabloon")
    data.update(kw)
    i = Idea(**data)
    i.screening = screen_by_rules(i)
    i.finance = build_financial_model(i, [])
    return i


def comp(saturation, n_direct=0, prices=0):
    return CompetitionAnalysis(
        mode="zoeken: tavily", saturation=saturation, saturation_reason=f"{n_direct} directe",
        competitors=[Competitor(name=f"C{i}", domain=f"c{i}.be", url=f"https://c{i}.be", description="d",
                                kind="direct", source="web") for i in range(n_direct)],
        price_min_month=29 if prices else None, price_median_month=45 if prices else None,
        price_max_month=59 if prices else None, verified_price_count=prices)


def research(n_facts):
    cov = {t: TopicCoverage(facts=1 if i < n_facts else 0) for i, t in enumerate(ResearchTopic)}
    return ResearchSummary(mode="t", coverage=cov)


def test_offline_template_idea_is_criticised_but_has_no_fatal_flaw():
    r = review_by_rules(idea())
    assert r.check(Q.SATURATED).verdict == "onbekend" and r.check(Q.SATURATED).alarming
    assert r.check(Q.WILL_PAY).verdict == "onbekend"
    assert r.check(Q.COPYABLE).verdict == "ja" and r.check(Q.MOAT).verdict == "nee"
    assert "vuistregel" in r.check(Q.CAC_TOO_HIGH).explanation
    assert len(r.risks) <= 5 and r.risks == sorted(r.risks, key=lambda x: x.severity, reverse=True)
    assert any(x.title == "Onvoldoende bewijs" for x in r.risks)
    assert r.fatal_flaws == []  # hypotheses never produce fatal flaws
    assert r.validation_test.method.startswith("Probleeminterviews")
    assert "€ 49" in r.validation_test.critical_assumption


def test_evidence_based_saturation_is_fatal():
    i = idea()
    i.competition = comp("hoog", n_direct=7)
    i.research = research(6)
    r = review_by_rules(i)
    assert r.check(Q.SATURATED).verdict == "ja"
    assert any(f.startswith("Verzadigde markt") for f in r.fatal_flaws)


def test_verified_prices_answer_willingness_to_pay():
    i = idea()
    i.competition = comp("gemiddeld", n_direct=3, prices=3)
    r = review_by_rules(i)
    c = r.check(Q.WILL_PAY)
    assert c.verdict == "ja" and not c.alarming and c.basis is EvidenceType.FACT
    assert "niet voor dit product" in c.explanation


def test_bad_unit_economics_on_hypotheses_is_a_risk_not_fatal():
    i = idea(target_customer="jonge ouders", problem="Jonge ouders verliezen geld aan dure babyspullen die snel te klein zijn.")
    i.screening = screen_by_rules(i)
    i.finance = build_financial_model(i, [])
    i.finance.scenarios["realistisch"].ltv_cac = 0.8
    r = review_by_rules(i)
    risk = next(x for x in r.risks if x.title == "Klantenwerving te duur")
    assert risk.likelihood == 4 and risk.impact == 5 and not risk.fatal
    assert r.validation_test.method.startswith("Wachtlijst")


def test_transaction_model_gets_concierge_test():
    i = idea(business_model="commissie per rit")
    i.finance = build_financial_model(i, [])
    assert review_by_rules(i).validation_test.method.startswith("Concierge")


def test_differentiator_with_moat_words_is_not_called_copyable():
    r = review_by_rules(idea(generated_by="llm", differentiator="Exclusieve integratie met bouwsoftware en eigen data."))
    assert r.check(Q.COPYABLE).verdict == "onbekend" and r.check(Q.MOAT).verdict == "onbekend"


def llm_review():
    return _LLMReview(
        checks=[_LLMCheck(question=Q.WILL_PAY, verdict="nee", explanation="Aannemers verwachten dit gratis bij hun boekhoudpakket."),
                _LLMCheck(question=Q.SATURATED, verdict="ja", explanation="Veel spelers."),
                _LLMCheck(question=Q.COPYABLE, verdict="onbekend", explanation="?")],
        risks=[_LLMRisk(title="Bundeling door boekhoudpakketten", description="Grote pakketten voegen dit gratis toe.",
                        likelihood=4, impact=5, mitigation="Focus op werf, niet op kantoor.")],
        failure_story="Een groot boekhoudpakket lanceert dezelfde functie gratis.")


def test_llm_red_team_fills_unknowns_adds_risks_but_never_fatal():
    i = idea()
    i.competition = comp("laag", n_direct=1)  # web data: not saturated
    llm = LLMClient(FakeBackend({"red_team": llm_review()}), RunBudget(max_calls=3), free=True)
    r = review_idea(i, llm)
    assert r.reviewed_by == "regels + llm"
    assert r.check(Q.WILL_PAY).verdict == "nee" and r.check(Q.WILL_PAY).source == "llm"
    assert r.check(Q.SATURATED).verdict == "nee"  # data-based rule verdict kept
    assert r.check(Q.COPYABLE).verdict == "ja"   # model's 'onbekend' does not erase a rule answer
    bundling = next(x for x in r.risks if x.title.startswith("Bundeling"))
    assert bundling.basis is EvidenceType.HYPOTHESIS and not bundling.fatal
    assert "boekhoudpakket" in r.summary


def test_llm_down_keeps_rule_review():
    def down(_):
        raise ConnectionError("x")
    llm = LLMClient(FakeBackend({"red_team": down}), RunBudget(max_calls=3), free=True)
    assert review_idea(idea(), llm).reviewed_by == "regels"
