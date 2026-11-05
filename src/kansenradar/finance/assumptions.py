"""Build the assumption set for an idea.

Order of preference for every number: verified data (competitor prices, Eurostat
counts) -> estimate derived from that data -> rule of thumb (HYPOTHESE). Rules of
thumb are typical ranges for small Benelux start-ups; they are explicitly labelled
as hypotheses and must be validated (phase 7 validation test).
"""

from __future__ import annotations

import re

from kansenradar.domain import Assumption, Criterion, Evidence, EvidenceType, Idea, ResearchTopic
from kansenradar.history.embeddings import normalize

H, E = EvidenceType.HYPOTHESIS, EvidenceType.ESTIMATE

# Rules of thumb per revenue model: (low, mid, high)
DEFAULTS = {
    "abonnement_b2b": dict(price=(25, 49, 89), cac=(200, 350, 600), churn=(1.5, 2.5, 4.0),
                           margin=(75, 82, 88), organic=(1, 2, 4)),
    "abonnement_b2c": dict(price=(5, 9, 15), cac=(20, 35, 60), churn=(3, 5, 8),
                           margin=(70, 80, 88), organic=(5, 10, 20)),
    "transactie": dict(price=(15, 30, 60), cac=(50, 90, 150), churn=(4, 6, 10),
                       margin=(80, 88, 93), organic=(2, 4, 8)),
    "eenmalig": dict(price=(30, 60, 120), cac=(15, 25, 45), churn=(100, 100, 100),
                     margin=(60, 75, 85), organic=(5, 10, 20)),
}
B2C_WORDS = ("ouders", "consument", "gezinnen", "particulier", "mantelzorger", "kopers", "baasjes",
             "eigenaars van", "studenten", "senioren")


def revenue_model(idea: Idea) -> str:
    model = normalize(idea.business_model)
    if any(w in model for w in ("commissie", "per transactie", "per rit", "per lening")):
        return "transactie"
    if any(w in model for w in ("eenmalig", "aankoop", "per rapport", "betaald rapport")):
        return "eenmalig"
    target = normalize(idea.target_customer)
    b2c = any(w in target for w in B2C_WORDS) and "verhuurder" not in target
    return "abonnement_b2c" if b2c else "abonnement_b2b"


def _market_facts(evidence: list[Evidence]) -> int | None:
    """Sum of the latest Eurostat enterprise counts (per country and NACE code)."""
    total, seen = 0, set()
    for ev in evidence:
        if ev.type is not EvidenceType.FACT or ev.topic != ResearchTopic.MARKET_SIZE.value:
            continue
        m = re.match(r"ENT_NR (\w+) (\w+) (\d{4}) = (\d+)", ev.quote or "")
        if m and (m.group(1), m.group(2)) not in seen:
            seen.add((m.group(1), m.group(2)))
            total += int(m.group(4))
    return total or None


def build_assumptions(idea: Idea, evidence: list[Evidence],
                      marketing_per_month: float = 750) -> tuple[str, list[Assumption], list[str]]:
    rm = revenue_model(idea)
    d = DEFAULTS[rm]
    notes: list[str] = []
    out: list[Assumption] = []

    comp = idea.competition
    if comp and comp.verified_price_count >= 2 and rm != "eenmalig":
        lo = comp.price_min_month
        mid = comp.price_median_month
        hi = comp.price_max_month
        out.append(Assumption(
            key="price", label="Prijs per klant per maand", low=lo, mid=mid, high=hi, unit="EUR/maand",
            basis=E, source=f"Afgeleid van {comp.verified_price_count} geverifieerde prijzen van concurrenten "
                            f"(min/mediaan/max)."))
    else:
        out.append(Assumption(
            key="price", label="Prijs per klant per maand" if rm != "eenmalig" else "Prijs per verkoop",
            low=d["price"][0], mid=d["price"][1], high=d["price"][2],
            unit="EUR/maand" if rm != "eenmalig" else "EUR/verkoop", basis=H,
            source="Vuistregel voor dit verdienmodel; geen geverifieerde concurrentieprijzen."))

    market = _market_facts(evidence)
    if market:
        out.append(Assumption(
            key="market", label="Potentiële klanten (BE+NL)", low=round(market * 0.2), mid=round(market * 0.35),
            high=round(market * 0.5), unit="klanten", basis=E,
            source=f"20-50% van {market:,} ondernemingen volgens Eurostat (afbakening doelgroep is een "
                   "schatting).".replace(",", ".")))
    else:
        base = (20_000, 50_000, 100_000) if rm in ("abonnement_b2c", "eenmalig") else (2_000, 5_000, 10_000)
        out.append(Assumption(key="market", label="Potentiële klanten (BE+NL)", low=base[0], mid=base[1],
                              high=base[2], unit="klanten", basis=H,
                              source="Geen officiële telling beschikbaar; ordegrootte als hypothese."))
        notes.append("Marktomvang is niet onderbouwd met officiële cijfers.")

    out += [
        Assumption(key="penetration", label="Maximaal marktaandeel na 3 jaar", low=1, mid=3, high=6, unit="%",
                   basis=H, source="Vuistregel voor een kleine starter zonder groot marketingbudget."),
        Assumption(key="cac", label="Klantenwervingskosten (CAC)", low=d["cac"][0], mid=d["cac"][1],
                   high=d["cac"][2], unit="EUR/klant", basis=H, higher_is_better=False,
                   source="Vuistregel per verdienmodel; te valideren met een eerste campagne."),
        Assumption(key="churn", label="Klantverloop per maand", low=d["churn"][0], mid=d["churn"][1],
                   high=d["churn"][2], unit="%", basis=H, higher_is_better=False,
                   source="Vuistregel per verdienmodel." + (" Eenmalige aankoop: geen terugkerende omzet."
                                                            if rm == "eenmalig" else "")),
        Assumption(key="margin", label="Brutomarge", low=d["margin"][0], mid=d["margin"][1], high=d["margin"][2],
                   unit="%", basis=H, source="Vuistregel: hosting, betaalkosten, support, AI-gebruik."),
        Assumption(key="organic", label="Organische nieuwe klanten per maand", low=d["organic"][0],
                   mid=d["organic"][1], high=d["organic"][2], unit="klanten/maand", basis=H,
                   source="Mond-tot-mond en zoekverkeer; vuistregel."),
        Assumption(key="marketing", label="Marketingbudget per maand", low=marketing_per_month,
                   mid=marketing_per_month, high=marketing_per_month, unit="EUR/maand", basis=H,
                   source="Beleidskeuze (instelbaar), geen onzekerheid."),
        Assumption(key="fixed", label="Vaste kosten per maand (excl. vergoeding oprichter)", low=150, mid=250,
                   high=450, unit="EUR/maand", basis=H, higher_is_better=False,
                   source="Tools, hosting, boekhouder, verzekering voor een kleine digitale starter."),
    ]

    capital = [800, 2_000, 5_000]
    reason = "MVP gebouwd door oprichter(s); kosten voor juridisch, merk, tools en buffer."
    if idea.screening and idea.screening.score_of(Criterion.FEASIBILITY) <= 2:
        capital = [c + 10_000 for c in capital]
        reason += " +10.000 voor vergunning/certificering/juridisch advies (regelgeving)."
        notes.append("Regelgeving kan startkapitaal en doorlooptijd sterk verhogen.")
    if any(w in normalize(f"{idea.one_liner} {idea.business_model}") for w in ("hardware", "sensor", "toestel", "apparaat")):
        capital = [c + 25_000 for c in capital]
        reason += " +25.000 voor hardware-ontwikkeling."
    out.append(Assumption(key="startup", label="Startkapitaal", low=capital[0], mid=capital[1], high=capital[2],
                          unit="EUR", basis=H, higher_is_better=False, source=reason))
    notes.append("Nettoresultaat is vóór vergoeding van de oprichter(s).")
    return rm, out, notes
