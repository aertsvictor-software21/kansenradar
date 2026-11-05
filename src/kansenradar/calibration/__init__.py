"""Calibration: run the assessment chain on labelled cases and report agreement.

A case (JSON) describes an idea, the evidence found (topic + source quality), the
competition situation, optionally validated financial numbers, and the acceptable
conclusions. The chain is the offline one: rule screening -> finance -> rule red
team -> scoring. This tests the decision logic; real-world validity requires real
cases with a known outcome.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kansenradar.domain import (
    CompetitionAnalysis,
    Competitor,
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
)
from kansenradar.finance import build_financial_model
from kansenradar.finance.model import SCENARIOS, scalability, simulate
from kansenradar.research.researcher import coverage
from kansenradar.review import review_by_rules
from kansenradar.scoring import score_idea
from kansenradar.screening import screen_by_rules

DEFAULT_CASES = Path(__file__).resolve().parents[3] / "kalibratie" / "archetypes.json"


@dataclass
class CaseResult:
    name: str
    synthetic: bool
    expected: list[Conclusion]
    got: Conclusion
    total: float
    confidence: float

    @property
    def ok(self) -> bool:
        return self.got in self.expected


def build_case(case: dict) -> tuple[Idea, list[Evidence]]:
    now = datetime.now(UTC)
    idea = Idea(
        title=case["naam"], one_liner=case["omschrijving"], problem=case["probleem"],
        target_customer=case["doelgroep"], business_model=case["verdienmodel"],
        sector=Sector[case.get("sector", "SAAS")],
        strategy=OriginalityStrategy[case.get("strategie", "PROBLEM_DRIVEN")],
        differentiator=case.get("onderscheidend"), why_now=case.get("waarom_nu"),
        generated_by="manueel", status=IdeaStatus.RESEARCHED,
    )
    evidence = [
        Evidence(claim=f"Kalibratiefeit over {t}", type=EvidenceType.FACT, topic=ResearchTopic[t].value,
                 source_url="https://kalibratie.local", quote="kalibratiecitaat uit de bron",
                 verified=True, retrieved_at=now, source_quality=SourceQuality[q])
        for t, q in case.get("feiten", {}).items()
    ]
    idea.screening = screen_by_rules(idea)
    idea.research = ResearchSummary(mode="kalibratie", coverage=coverage(evidence, 365))
    c = case.get("concurrentie")
    if c:
        prices = c.get("prijzen", [])
        idea.competition = CompetitionAnalysis(
            mode="kalibratie", saturation=c["verzadiging"], saturation_reason="kalibratiegeval",
            competitors=[Competitor(name=f"C{i}", domain=f"c{i}.example", url=f"https://c{i}.example",
                                    description="kalibratie", kind="direct", source="web")
                         for i in range(c.get("direct", 0))],
            price_min_month=min(prices) if prices else None,
            price_median_month=sorted(prices)[len(prices) // 2] if prices else None,
            price_max_month=max(prices) if prices else None, verified_price_count=len(prices))
    fin = build_financial_model(idea, evidence)
    overrides = case.get("financieel", {})
    if overrides:
        basis = EvidenceType(overrides.pop("basis", "SCHATTING"))
        for a in fin.assumptions:
            if a.key in overrides:
                v = overrides[a.key]
                a.low, a.mid, a.high = (v, v, v) if not isinstance(v, list) else v
                a.basis, a.source = basis, "Gevalideerd kalibratiecijfer."
        fin.scenarios = {sc: simulate({a.key: a.for_scenario(sc) for a in fin.assumptions}, sc) for sc in SCENARIOS}
        fin.scalability, fin.scalability_reason = scalability(fin.scenarios, fin.revenue_model)
    idea.finance = fin
    idea.review = review_by_rules(idea)
    return idea, evidence


def run_calibration(path: Path = DEFAULT_CASES) -> list[CaseResult]:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    results = []
    for case in cases:
        idea, evidence = build_case(dict(case))
        sc = score_idea(idea, evidence)
        results.append(CaseResult(name=case["naam"], synthetic=case.get("synthetisch", True),
                                  expected=[Conclusion(x) for x in case["verwacht"]], got=sc.conclusion,
                                  total=sc.total, confidence=sc.confidence))
    return results


def report(results: list[CaseResult]) -> str:
    ok = sum(r.ok for r in results)
    lines = [f"Kalibratie: {ok}/{len(results)} gevallen binnen de verwachte conclusie."]
    synth = sum(r.synthetic for r in results)
    if synth:
        lines.append(f"Let op: {synth} gevallen zijn synthetisch (toetsen de logica, niet de werkelijkheid).")
    for r in results:
        mark = "OK  " if r.ok else "FOUT"
        lines.append(f"  {mark} {r.name}: {r.got.value} ({r.total:.0f}/100, {r.confidence:.0f}%)"
                     + ("" if r.ok else f" — verwacht: {' of '.join(c.value for c in r.expected)}"))
    dist = Counter(r.got for r in results)
    share = dist[Conclusion.VERY_INTERESTING] / len(results) if results else 0
    lines.append(f"Optimismecontrole: {share:.0%} van de gevallen 'ZEER INTERESSANTE OPPORTUNITEIT'"
                 + (" — WAARSCHUWING: te mild?" if share > 0.2 else " (zeldzaam, zoals bedoeld)."))
    return "\n".join(lines)


__all__ = ["DEFAULT_CASES", "CaseResult", "build_case", "report", "run_calibration"]
