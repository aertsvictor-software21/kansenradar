import json

from kansenradar.calibration import build_case, report, run_calibration
from kansenradar.domain import Conclusion
from kansenradar.scoring import score_idea

BASE = json.load(open("kalibratie/archetypes.json", encoding="utf-8"))[0]  # strong case A


def score(case):
    idea, ev = build_case(json.loads(json.dumps(case)))
    return score_idea(idea, ev)


def test_archetypes_all_within_expected_band():
    results = run_calibration()
    assert all(r.ok for r in results), report(results)
    share = sum(r.got is Conclusion.VERY_INTERESTING for r in results) / len(results)
    assert share <= 0.2  # optimism check


def test_more_evidence_never_lowers_confidence():
    fewer = dict(BASE, feiten=dict(list(BASE["feiten"].items())[:3]))
    assert score(BASE).confidence >= score(fewer).confidence


def test_worse_unit_economics_never_raise_score():
    worse = dict(BASE, financieel=dict(BASE["financieel"], cac=[600, 800, 1000]))
    assert score(worse).total <= score(BASE).total


def test_higher_saturation_never_raises_score():
    sat = dict(BASE, concurrentie=dict(BASE["concurrentie"], verzadiging="gemiddeld", direct=4))
    assert score(sat).total <= score(BASE).total


def test_removing_evidence_moves_strong_case_to_more_research():
    bare = dict(BASE, feiten={}, concurrentie=None, financieel={})
    assert score(bare).conclusion is Conclusion.MORE_RESEARCH
