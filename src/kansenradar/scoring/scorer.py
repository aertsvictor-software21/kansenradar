"""Deterministic scoring: 100 points, evidence caps, confidence and conclusion.

Every sub-score has a rationale. Evidence caps ensure an idea cannot score high
on a criterion without verified facts. Confidence measures how well the whole
analysis is supported, independently of how attractive the idea looks.

Conclusion (in this order):
  1. Fatal flaw from the red team (evidence-based)          -> AFWIJZEN
  2. Rejected at screening, or total < 40                   -> AFWIJZEN
  3. Confidence < 35%                                       -> MEER ONDERZOEK NODIG
  4. Total >= 75 and confidence >= 65%                      -> ZEER INTERESSANTE OPPORTUNITEIT
  5. Total >= 62 and confidence >= 50%                      -> KANSRIJK
  6. Total >= 50                                            -> EXPERIMENTEREN
  7. Otherwise (40-49)                                      -> MEER ONDERZOEK NODIG
"""

from __future__ import annotations

from kansenradar.domain import (
    CRITERIA_MAX,
    Conclusion,
    Criterion,
    Evidence,
    EvidenceType,
    Idea,
    InvestorQuestion,
    ResearchTopic,
    Scorecard,
    ScreeningDecision,
    SourceQuality,
    SubScore,
)

T, Q = ResearchTopic, InvestorQuestion
STRONG_SOURCES = {SourceQuality.OFFICIAL, SourceQuality.INDUSTRY, SourceQuality.MEDIA}


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# Desk research alone never justifies high confidence; only real customer validation does.
DESK_RESEARCH_CONFIDENCE_CAP = 60.0
VALIDATION_TOPIC = "klantvalidatie"


def _domain(url: str | None) -> str:
    from urllib.parse import urlsplit
    host = urlsplit(url or "").netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _corroborated(facts: list[Evidence]) -> bool:
    """A topic counts as supported with facts from >= 2 different websites, or from an
    official or industry source. One blog post alone is not enough."""
    domains = {_domain(f.source_url) for f in facts}
    return len(domains) >= 2 or any(
        f.source_quality in (SourceQuality.OFFICIAL, SourceQuality.INDUSTRY) for f in facts)


def _facts(evidence: list[Evidence], max_age: int) -> dict[str, list[Evidence]]:
    """Facts per topic, keeping only topics whose facts are corroborated."""
    out: dict[str, list[Evidence]] = {}
    for ev in evidence:
        if ev.effective_type(max_age) is EvidenceType.FACT:
            out.setdefault(ev.topic, []).append(ev)
    return {t: f for t, f in out.items() if _corroborated(f) or t == VALIDATION_TOPIC}


def _sub(name: str, fraction: float, rationale: str, cap: tuple[float, str] | None = None) -> SubScore:
    mx = CRITERIA_MAX[name]
    points = round(_clamp(fraction) * mx, 1)
    capped_by = None
    if cap is not None and points > cap[0]:
        points, capped_by = cap[0], cap[1]
    return SubScore(criterion=name, points=points, max_points=mx, rationale=rationale, capped_by=capped_by)


def score_idea(idea: Idea, evidence: list[Evidence], max_age_days: int = 365) -> Scorecard:
    scr, fin, comp, rev, res = idea.screening, idea.finance, idea.competition, idea.review, idea.research
    facts = _facts(evidence, max_age_days)
    has = lambda *topics: any(facts.get(t.value) for t in topics)  # noqa: E731
    s = (lambda c: scr.score_of(c)) if scr else (lambda c: 3)
    subs: list[SubScore] = []

    # 1. Marktvraag (20)
    frac = 0.5 * (s(Criterion.PAIN) - 1) / 4 + 0.5 * (s(Criterion.MARKET_SIZE) - 1) / 4
    if has(T.DEMAND, T.PROBLEM) and has(T.MARKET_SIZE, T.CUSTOMERS):
        frac += 0.15
    cap = None
    if not has(T.DEMAND, T.PROBLEM, T.CUSTOMERS, T.MARKET_SIZE):
        cap = (8, "geen geverifieerd feit over vraag, probleem, klanten of marktgrootte")
    elif not has(T.DEMAND, T.PROBLEM):
        cap = (12, "enkel marktomvang onderbouwd, vraag zelf niet aangetoond")
    subs.append(_sub("Marktvraag", frac, f"Pijn {s(Criterion.PAIN)}/5, marktomvang {s(Criterion.MARKET_SIZE)}/5"
                     + ("; vraag en markt met feiten onderbouwd." if frac > 0.5 + 0.15 else "."), cap))

    # 2. Winstgevendheid (15)
    if fin:
        r = fin.scenarios["realistisch"]
        be = r.operational_break_even_month
        profit = 0 if r.net_year3 <= 0 else (1 if be and be <= 12 else 0.7 if be and be <= 24 else 0.4 if be else 0.1)
        frac = (_clamp((r.gross_margin_pct - 40) / 45) + _clamp(r.ltv_cac / 5) + profit) / 3
        net = f"{r.net_year3:,.0f}".replace(",", ".")
        rationale = (f"Realistisch: brutomarge {r.gross_margin_pct:.0f}%, LTV/CAC {r.ltv_cac:.1f}, "
                     f"netto jaar 3 € {net}, break-even maand {be or '>36'}.")
    else:
        frac, rationale = 0.3, "Geen financieel model."
    priced = (comp and comp.verified_price_count >= 2) or has(T.PRICING)
    subs.append(_sub("Winstgevendheid", frac, rationale,
                     None if priced else (7, "geen geverifieerde prijzen; winstgevendheid rust op vuistregels")))

    # 3. Klantenwerving (15)
    cac_ok = rev and rev.check(Q.CAC_TOO_HIGH).verdict == "nee"
    frac = 0.6 * (s(Criterion.REACHABILITY) - 1) / 4 + 0.4 * (1 if cac_ok else 0.3)
    subs.append(_sub("Klantenwerving", frac,
                     f"Bereikbaarheid {s(Criterion.REACHABILITY)}/5; LTV/CAC {'voldoende' if cac_ok else 'onvoldoende of onbekend'}.",
                     None if has(T.REACH) else (9, "kanalen om klanten te bereiken niet met feiten onderbouwd")))

    # 4. Onderscheidend vermogen (15)
    sat = {"laag": 0.4, "gemiddeld": 0.27, "hoog": 0.07}.get(comp.saturation if comp else "onbekend", 0.2)
    moat = 0.33 if rev and rev.check(Q.MOAT).verdict in ("ja", "onbekend") else 0.07
    copy = {"nee": 0.27, "onbekend": 0.13}.get(rev.check(Q.COPYABLE).verdict if rev else "onbekend", 0.0)
    subs.append(_sub("Onderscheidend vermogen", sat + moat + copy,
                     f"Verzadiging {comp.saturation if comp else 'onbekend'}; concurrentievoordeel "
                     f"{rev.check(Q.MOAT).verdict if rev else 'onbekend'}; kopieerbaar "
                     f"{rev.check(Q.COPYABLE).verdict if rev else 'onbekend'}.",
                     None if comp and comp.saturation != "onbekend"
                     else (7, "concurrentie niet onderzocht via het web")))

    # 5. Schaalbaarheid (10)
    sc = fin.scalability if fin else "gemiddeld"
    rules_only = fin is not None and all(
        fin.assumption(k).basis is EvidenceType.HYPOTHESIS for k in ("price", "cac", "churn", "margin"))
    subs.append(_sub("Schaalbaarheid", {"hoog": 0.9, "gemiddeld": 0.6, "laag": 0.2}[sc],
                     fin.scalability_reason if fin else "Geen financieel model.",
                     (6, "eenheidseconomie rust volledig op vuistregels") if rules_only or fin is None else None))

    # 6. Benodigd startkapitaal (10): less capital needed = more points
    need = fin.scenarios["realistisch"].capital_needed if fin else 20_000
    frac = 1.0 if need <= 5_000 else 0.8 if need <= 15_000 else 0.6 if need <= 40_000 else 0.4 if need <= 100_000 else 0.2
    subs.append(_sub("Benodigd startkapitaal", frac,
                     f"Benodigd kapitaal realistisch € {need:,.0f} (incl. aanloopverliezen).".replace(",", ".")))

    # 7. Uitvoerbaarheid (10)
    subs.append(_sub("Uitvoerbaarheid", (s(Criterion.FEASIBILITY) - 1) / 4,
                     f"Uitvoerbaarheid {s(Criterion.FEASIBILITY)}/5 (regelgeving, techniek, kapitaal)."))

    # 8. Markttiming (5)
    frac = 0.4 + (0.2 if idea.why_now else 0) + (0.2 if (idea.seed_key or "").startswith("reg:") else 0) \
        + (0.3 if has(T.TRENDS, T.REGULATION) else 0)
    subs.append(_sub("Markttiming", frac,
                     "Waarom nu: " + (idea.why_now or "niet onderbouwd") + ".",
                     None if has(T.TRENDS, T.REGULATION) else (3, "geen feit over trends of regelgeving")))

    total = round(sum(x.points for x in subs), 1)
    confidence, reasons = _confidence(idea, evidence, facts, max_age_days)
    conclusion, why = _conclusion(idea, total, confidence)
    return Scorecard(subscores=subs, total=total, confidence=confidence, confidence_reasons=reasons,
                     conclusion=conclusion, conclusion_reason=why)


def _confidence(idea: Idea, evidence, facts, max_age) -> tuple[float, list[str]]:
    res, comp, rev = idea.research, idea.competition, idea.review
    reasons = []
    n_topics = sum(1 for t in ResearchTopic if facts.get(t.value))
    coverage = 50 * n_topics / 10
    reasons.append(f"{n_topics}/10 onderzoeksonderwerpen met bevestigd bewijs (≥ 2 websites of een "
                   f"officiële/sectorbron) (+{coverage:.0f}).")
    key = [(T.DEMAND, T.PROBLEM), (T.MARKET_SIZE, T.CUSTOMERS), (T.PRICING,), (T.COMPETITORS,)]
    key_hits = sum(any(facts.get(t.value) for t in group) for group in key)
    reasons.append(f"{key_hits}/4 kernonderwerpen (vraag, markt, prijs, concurrentie) onderbouwd (+{7.5 * key_hits:.0f}).")
    all_facts = [e for lst in facts.values() for e in lst]
    strong = sum(e.source_quality in STRONG_SOURCES for e in all_facts)
    quality = 10 * strong / len(all_facts) if all_facts else 0
    reasons.append(f"{strong}/{len(all_facts)} feiten uit sterke bronnen (officieel, sector, media) (+{quality:.0f}).")
    completeness = (5 if comp and comp.saturation != "onbekend" else 0) + (5 if rev else 0)
    reasons.append(f"Volledigheid analyse (+{completeness}).")
    total = min(100.0, coverage + 7.5 * key_hits + quality + completeness)
    if not facts.get(VALIDATION_TOPIC) and total > DESK_RESEARCH_CONFIDENCE_CAP:
        total = DESK_RESEARCH_CONFIDENCE_CAP
        reasons.append(f"Zonder klantvalidatie maximaal {DESK_RESEARCH_CONFIDENCE_CAP:.0f}% "
                       "(enkel bureauonderzoek).")
    return round(total, 1), reasons


def _conclusion(idea: Idea, total: float, confidence: float) -> tuple[Conclusion, str]:
    rev, scr = idea.review, idea.screening
    if rev and rev.fatal_flaws:
        return Conclusion.REJECT, "Fataal gebrek: " + rev.fatal_flaws[0]
    if scr and scr.decision is ScreeningDecision.REJECT:
        return Conclusion.REJECT, "Afgewezen bij screening: " + "; ".join(scr.knockouts)
    if total < 40:
        return Conclusion.REJECT, f"Totaalscore {total:.0f}/100 is te laag."
    if confidence < 35:
        return Conclusion.MORE_RESEARCH, (f"Betrouwbaarheid {confidence:.0f}% is te laag om te oordelen "
                                          f"(score {total:.0f}/100). Voer de validatietest uit of voeg bronnen toe.")
    if total >= 75 and confidence >= 65:
        return Conclusion.VERY_INTERESTING, f"Score {total:.0f}/100 met betrouwbaarheid {confidence:.0f}%."
    if total >= 62 and confidence >= 50:
        return Conclusion.PROMISING, f"Score {total:.0f}/100 met betrouwbaarheid {confidence:.0f}%."
    if total >= 50:
        return Conclusion.EXPERIMENT, (f"Score {total:.0f}/100: de moeite waard om goedkoop te testen "
                                       "met de validatietest.")
    return Conclusion.MORE_RESEARCH, f"Score {total:.0f}/100: twijfelgeval, meer onderzoek nodig."


CONCLUSION_RANK = {
    Conclusion.VERY_INTERESTING: 0, Conclusion.PROMISING: 1, Conclusion.EXPERIMENT: 2,
    Conclusion.MORE_RESEARCH: 3, Conclusion.REJECT: 4,
}


def rank(ideas: list[Idea]) -> list[Idea]:
    scored = [i for i in ideas if i.scorecard]
    return sorted(scored, key=lambda i: (CONCLUSION_RANK[i.scorecard.conclusion],
                                         -i.scorecard.total, -i.scorecard.confidence))
