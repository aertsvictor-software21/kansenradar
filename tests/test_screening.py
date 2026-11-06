import pytest

from kansenradar.cli import main as cli_main
from kansenradar.domain import (
    Criterion,
    CriterionScore,
    EvidenceType,
    Idea,
    IdeaStatus,
    OriginalityStrategy,
    ScreeningDecision,
    Sector,
)
from kansenradar.llm import FakeBackend, LLMClient, RunBudget
from kansenradar.screening import decide, research_priority, screen_by_rules, screen_ideas, screen_with_llm
from kansenradar.screening.llm import _LLMCriterion, _LLMScreen, _LLMScreenBatch
from kansenradar.storage import Repository, make_engine


def idea(**kw) -> Idea:
    data = dict(
        title="Werfverslag uit spraak",
        one_liner="App die ingesproken notities omzet in werfverslagen en meerwerkfacturen.",
        problem="Kleine aannemers verliezen avonden aan werfverslagen en lopen omzet mis door vergeten meerwerk.",
        target_customer="aannemers met 1-20 werknemers", sector=Sector.TRADITIONAL,
        business_model="abonnement per gebruiker", strategy=OriginalityStrategy.NEW_TECH_APPLICATION,
    )
    data.update(kw)
    return Idea(**data)


def scores(**override) -> list[CriterionScore]:
    base = {c: 4 for c in Criterion}
    base.update({Criterion[k]: v for k, v in override.items()})
    return [CriterionScore(criterion=c, score=v, known=True, motivation=f"test {v}") for c, v in base.items()]


def run_decide(s, by="llm"):
    return decide(s, screened_by=by, key_assumptions=[], research_questions=[], summary="t")


# --- decision rules ----------------------------------------------------------------

def test_strong_idea_passes_with_medium_confidence_at_most():
    r = run_decide(scores())
    assert r.decision is ScreeningDecision.PASS and r.weighted_score == 4.0
    assert r.confidence == "gemiddeld"
    assert run_decide(scores(), by="regels").confidence == "laag"


@pytest.mark.parametrize("criterion", ["PAIN", "WILLINGNESS_TO_PAY", "COMPETITION", "FEASIBILITY"])
def test_knockout_rejects_even_with_high_average(criterion):
    r = run_decide(scores(**{criterion: 1}))
    assert r.decision is ScreeningDecision.REJECT
    assert r.knockouts and r.knockouts[0].startswith(Criterion[criterion].value)


def test_market_size_one_is_not_a_knockout_but_blocks_pass():
    r = run_decide(scores(MARKET_SIZE=1))
    assert r.decision is ScreeningDecision.DOUBT


def test_weak_across_the_board_is_rejected():
    r = run_decide(scores(**{c.name: 2 for c in Criterion}))
    assert r.decision is ScreeningDecision.REJECT and "te zwak" in r.knockouts[0]


def test_result_requires_every_criterion_once():
    with pytest.raises(ValueError):
        run_decide(scores()[:-1])


def test_priority_commercial_first_originality_tiebreak():
    strong = run_decide(scores())
    weak = run_decide(scores(MARKET_SIZE=3, REACHABILITY=3, COMPETITION=3))
    assert research_priority(strong, 0) > research_priority(weak, 10)  # originality can't win
    # not even against the smallest possible commercial difference (1 point, lightest criterion)
    one_point = run_decide(scores(MARKET_SIZE=3))
    assert research_priority(strong, 0) > research_priority(one_point, 10)
    assert research_priority(strong, 9) > research_priority(strong, 1)
    assert research_priority(run_decide(scores(PAIN=1)), 10) is None


# --- rule screener -----------------------------------------------------------------

def test_rules_b2b_software_idea():
    r = screen_by_rules(idea())
    assert r.screened_by == "regels" and r.confidence == "laag"
    assert r.score_of(Criterion.PAIN) == 4
    assert r.score_of(Criterion.WILLINGNESS_TO_PAY) == 4
    assert r.score_of(Criterion.REACHABILITY) == 4
    unknown = {s.criterion for s in r.scores if not s.known}
    assert unknown == {Criterion.MARKET_SIZE, Criterion.COMPETITION}
    assert any("Hoeveel aannemers" in q for q in r.research_questions)


def test_rules_reject_missing_revenue_model():
    r = screen_by_rules(idea(business_model="nog te bepalen"))
    assert r.decision is ScreeningDecision.REJECT
    assert "verdienmodel" in r.knockouts[0]


def test_rules_flag_regulation_marketplace_and_consumers():
    r = screen_by_rules(idea(
        title="Kredietplatform voor ouders", one_liner="Marktplaats die ouders koppelt aan leningen.",
        problem="Jonge ouders betalen te veel kosten voor kleine leningen en verliezen geld aan rente.",
        target_customer="jonge ouders", business_model="commissie per lening"))
    assert r.score_of(Criterion.FEASIBILITY) == 2
    assert r.score_of(Criterion.REACHABILITY) == 1  # consumer (2) minus marketplace
    assert any("vergunningen" in q for q in r.research_questions)
    assert any("marktplaats" in a for a in r.key_assumptions)


def test_rules_health_data_lowers_feasibility():
    r = screen_by_rules(idea(target_customer="zelfstandige kinesitherapeuten",
                             problem="Kinesitherapeuten verliezen uren aan verslagen per patiënt."))
    assert r.score_of(Criterion.FEASIBILITY) == 3
    assert any("AVG" in q for q in r.research_questions)


def test_rules_avoid_false_pain_signals():
    r = screen_by_rules(idea(problem="Duurzame bedrijven met vaste klanten willen een mooiere website laten maken."))
    assert r.score_of(Criterion.PAIN) == 2


# --- LLM screener --------------------------------------------------------------------

def llm_result(index, **override):
    base = {c: 4 for c in Criterion}
    base.update({Criterion[k]: v for k, v in override.items()})
    return _LLMScreen(
        idea_index=index,
        scores=[_LLMCriterion(criterion=c, score=v, known=True, motivation="m") for c, v in base.items()],
        key_assumptions=["Aannemers betalen 30 euro per maand."],
        research_questions=["Hoeveel aannemers gebruiken al werfsoftware?"],
        summary="Kritische samenvatting.",
    )


def make_llm(handler):
    return LLMClient(FakeBackend({"screening": handler}), RunBudget(max_calls=20), free=True)


def test_llm_screening_code_decides_and_merges_questions():
    batch = _LLMScreenBatch(results=[llm_result(0), llm_result(1, COMPETITION=1)])
    results = screen_with_llm(make_llm(batch), [idea(), idea(title="Tweede idee")])
    assert [r.decision for r in results] == [ScreeningDecision.PASS, ScreeningDecision.REJECT]
    assert results[0].screened_by == "llm"
    assert results[0].research_questions[0] == "Hoeveel aannemers gebruiken al werfsoftware?"
    assert any("Hoeveel aannemers met" in q for q in results[0].research_questions)


def test_llm_cannot_overrule_missing_revenue_model():
    results = screen_with_llm(make_llm(_LLMScreenBatch(results=[llm_result(0)])),
                              [idea(business_model="later beslissen")])
    assert results[0].decision is ScreeningDecision.REJECT


def test_llm_missing_or_invalid_result_falls_back_to_rules():
    bad = llm_result(1)
    bad.scores = bad.scores[:3]
    results = screen_with_llm(make_llm(_LLMScreenBatch(results=[llm_result(0), bad])),
                              [idea(), idea(title="Tweede idee"), idea(title="Derde idee")])
    assert [r.screened_by for r in results] == ["llm", "regels", "regels"]


def test_llm_down_falls_back_to_rules():
    def down(_):
        raise ConnectionError("offline")
    results = screen_with_llm(make_llm(down), [idea()])
    assert results[0].screened_by == "regels"


# --- pipeline & CLI -------------------------------------------------------------------

def test_pipeline_updates_status_priority_and_stores_hypotheses():
    repo = Repository(make_engine("sqlite:///:memory:"))
    good, bad = idea(), idea(title="Zonder model", business_model="onbekend")
    dup = idea(title="Dubbel", status=IdeaStatus.DUPLICATE)
    for i in (good, bad, dup):
        repo.save_idea(i)
    report = screen_ideas(repo, batch_size=1)
    assert len(report.ideas) == 2  # duplicate is not screened
    assert repo.get_idea(good.id).status is IdeaStatus.SCREENED_PASS
    assert repo.get_idea(bad.id).status is IdeaStatus.SCREENED_OUT
    assert repo.get_idea(bad.id).research_priority is None
    assert [i.id for i in report.ranking()] == [good.id]
    evidence = repo.evidence_for(good.id)
    assert evidence and all(e.type is EvidenceType.HYPOTHESIS for e in evidence)
    assert screen_ideas(repo).summary().startswith("Geen nieuwe ideeën")


def test_cli_screen(tmp_path, monkeypatch, capsys):
    from kansenradar.config import get_settings
    monkeypatch.setenv("KR_DATABASE_URL", f"sqlite:///{tmp_path / 'kr.db'}")
    monkeypatch.setenv("KR_EMBEDDER", "hashing")
    monkeypatch.setenv("KR_LLM_PROVIDER", "none")
    get_settings.cache_clear()
    try:
        cli_main(["genereer", "--aantal", "4"])
        capsys.readouterr()
        assert cli_main(["screen"]) == 0
        out = capsys.readouterr().out
        assert "Screening (regels): 4 ideeën" in out and "lage betrouwbaarheid" in out
        idea_id = next(w for w in out.split() if w.startswith("idea_"))
        cli_main(["toon", idea_id])
        assert "Onderzoeksvragen" in capsys.readouterr().out
    finally:
        get_settings.cache_clear()


def test_rules_medical_diagnostics_is_regulated():
    r = screen_by_rules(idea(title="EdgeAI Diagnostiek Suite", one_liner="AI die röntgenbeelden analyseert.",
                             problem="Radiologiepraktijken verliezen tijd aan het beoordelen van beelden en missen afwijkingen.",
                             target_customer="radiologiepraktijken"))
    from kansenradar.domain import Criterion
    assert r.score_of(Criterion.FEASIBILITY) == 2


def test_dienstverlening_is_not_a_loan():
    r = screen_by_rules(idea(business_model="maandelijkse dienstverlening per zaak"))
    from kansenradar.domain import Criterion
    assert r.score_of(Criterion.FEASIBILITY) == 4
    assert screen_by_rules(idea(one_liner="App die kleine leningen aan aannemers verstrekt.")).score_of(Criterion.FEASIBILITY) == 2
