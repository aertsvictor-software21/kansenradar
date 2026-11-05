"""Rule-based red team: tries to make the idea fail using the data of phases 3-6."""

from __future__ import annotations

from kansenradar.domain import (
    Criterion,
    CriticalReview,
    EvidenceType,
    Idea,
    InvestorCheck,
    InvestorQuestion,
    Risk,
    ValidationTest,
)
from kansenradar.history.embeddings import normalize

Q = InvestorQuestion
F, E, H = EvidenceType.FACT, EvidenceType.ESTIMATE, EvidenceType.HYPOTHESIS
# Fewer realistically reachable customers than this cannot carry a business.
MIN_REACHABLE_CUSTOMERS = 50
MOAT_WORDS = ("data", "netwerk", "exclusie", "integratie", "certific", "licentie", "community",
              "merk", "patent", "partnership", "koppeling", "schaal", "overstapkost")


def _check(q, verdict, alarming, explanation, basis) -> InvestorCheck:
    return InvestorCheck(question=q, verdict=verdict, alarming=alarming, explanation=explanation,
                         basis=basis, source="regels")


def investor_checks(idea: Idea) -> list[InvestorCheck]:
    comp, fin, scr = idea.competition, idea.finance, idea.screening
    real = fin.scenarios["realistisch"] if fin else None
    checks = []

    # 1. Saturation
    if comp is None or comp.saturation == "onbekend":
        checks.append(_check(Q.SATURATED, "onbekend", True,
                             "Concurrentie is niet onderzocht; verzadiging kan niet worden uitgesloten.", H))
    else:
        verdict = "ja" if comp.saturation == "hoog" else "nee"
        checks.append(_check(Q.SATURATED, verdict, comp.saturation == "hoog",
                             f"Verzadiging {comp.saturation}: {comp.saturation_reason}", E))

    # 2. Willingness to pay
    if scr and scr.score_of(Criterion.WILLINGNESS_TO_PAY) == 1:
        checks.append(_check(Q.WILL_PAY, "nee", True, "Geen duidelijk verdienmodel.", E))
    elif comp and comp.verified_price_count >= 2:
        checks.append(_check(Q.WILL_PAY, "ja", False,
                             f"Klanten betalen al voor alternatieven: € {comp.price_min_month:g}–"
                             f"{comp.price_max_month:g} per maand ({comp.verified_price_count} geverifieerde "
                             "prijzen). Dat bewijst betalingsbereidheid voor de categorie, niet voor dit product.", F))
    else:
        checks.append(_check(Q.WILL_PAY, "onbekend", True,
                             "Geen geverifieerde prijzen of betaalbewijs gevonden.", H))

    # 3. Copyability
    diff = normalize(idea.differentiator or "")
    moat_hits = [w for w in MOAT_WORDS if w in diff]
    if idea.generated_by == "sjabloon" or not moat_hits:
        checks.append(_check(Q.COPYABLE, "ja", True,
                             "Softwareoplossing zonder aantoonbare drempel (data, netwerk, integraties, "
                             "certificering); een concurrent kan dit vermoedelijk snel nabouwen.", H))
    else:
        checks.append(_check(Q.COPYABLE, "onbekend", False,
                             f"Mogelijke drempels genoemd ({', '.join(moat_hits)}); nog niet bewezen.", H))

    # 4. CAC
    if real is None:
        checks.append(_check(Q.CAC_TOO_HIGH, "onbekend", True, "Geen financieel model.", H))
    else:
        cac_basis = fin.assumption("cac").basis
        too_high = real.ltv_cac < 3
        checks.append(_check(Q.CAC_TOO_HIGH, "ja" if too_high else "nee", too_high,
                             f"LTV/CAC realistisch {real.ltv_cac:.1f} (norm ≥ 3); CAC € {real.cac:g}"
                             + (" — CAC is een niet-gevalideerde vuistregel." if cac_basis is H else "."),
                             cac_basis if cac_basis is not F else E))

    # 5. Margins
    if real is None:
        checks.append(_check(Q.MARGINS_OK, "onbekend", True, "Geen financieel model.", H))
    else:
        ok = real.gross_margin_pct >= 60
        checks.append(_check(Q.MARGINS_OK, "ja" if ok else "nee", not ok,
                             f"Brutomarge realistisch {real.gross_margin_pct:.0f}% (drempel 60%).",
                             fin.assumption("margin").basis))

    # 6. Moat
    if moat_hits and idea.generated_by != "sjabloon":
        checks.append(_check(Q.MOAT, "onbekend", True,
                             f"Voordeel mogelijk via {', '.join(moat_hits)}, maar nog niet aangetoond.", H))
    else:
        checks.append(_check(Q.MOAT, "nee", True, "Geen duurzaam concurrentievoordeel aantoonbaar.", H))
    return checks


def rule_risks(idea: Idea, checks: list[InvestorCheck]) -> list[Risk]:
    risks: list[Risk] = []
    by_q = {c.question: c for c in checks}
    comp, fin, scr, res = idea.competition, idea.finance, idea.screening, idea.research

    if by_q[Q.SATURATED].verdict == "ja":
        risks.append(Risk(title="Verzadigde markt", description=by_q[Q.SATURATED].explanation,
                          likelihood=4, impact=5, basis=E, source="regels",
                          mitigation="Kies een smallere niche waar de grote spelers zwak zijn."))
    if by_q[Q.WILL_PAY].alarming:
        risks.append(Risk(title="Betalingsbereidheid onbewezen", description=by_q[Q.WILL_PAY].explanation,
                          likelihood=4 if by_q[Q.WILL_PAY].verdict == "nee" else 3, impact=5,
                          basis=by_q[Q.WILL_PAY].basis, source="regels",
                          mitigation="Vraag vooraf om betaling of een intentieverklaring (zie validatietest)."))
    if by_q[Q.COPYABLE].verdict == "ja":
        risks.append(Risk(title="Eenvoudig te kopiëren", description=by_q[Q.COPYABLE].explanation,
                          likelihood=3, impact=4, basis=H, source="regels",
                          mitigation="Bouw een drempel op: eigen data, integraties, sectorpartnerschappen."))
    if fin:
        real, cons = fin.scenarios["realistisch"], fin.scenarios["conservatief"]
        if real.ltv_cac < 3:
            basis = fin.assumption("cac").basis
            risks.append(Risk(title="Klantenwerving te duur", description=by_q[Q.CAC_TOO_HIGH].explanation,
                              likelihood=4 if real.ltv_cac < 1 else 3, impact=5,
                              basis=basis, source="regels",
                              mitigation="Test kanalen met lage kosten (federaties, partners, doorverwijzing)."))
        if cons.operational_break_even_month is None:
            risks.append(Risk(title="Financieel kwetsbaar",
                              description=f"Conservatief scenario: geen break-even binnen 36 maanden; "
                                          f"benodigd kapitaal € {cons.capital_needed:,.0f}.".replace(",", "."),
                              likelihood=3, impact=4, basis=H, source="regels",
                              mitigation="Valideer prijs en CAC vóór investeringen; houd vaste kosten minimaal."))
        reachable = fin.assumption("market").mid * fin.assumption("penetration").mid / 100
        if reachable < MIN_REACHABLE_CUSTOMERS:
            risks.append(Risk(title="Markt te klein",
                              description=f"Realistisch slechts ~{reachable:.0f} bereikbare klanten; dat draagt "
                                          "geen levensvatbaar bedrijf.",
                              likelihood=4, impact=5, basis=fin.assumption("market").basis, source="regels",
                              mitigation="Verbreed naar aanpalende doelgroepen of landen, of kies een ander idee."))
        elif real.market_cap_reached_month and real.market_cap_reached_month <= 24:
            risks.append(Risk(title="Markt te klein",
                              description=f"Het marktplafond wordt in maand {real.market_cap_reached_month} bereikt.",
                              likelihood=4, impact=4, basis=fin.assumption("market").basis, source="regels",
                              mitigation="Verbreed naar aanpalende doelgroepen of landen."))
        if real.gross_margin_pct < 60:
            risks.append(Risk(title="Te lage marges", description=by_q[Q.MARGINS_OK].explanation,
                              likelihood=4, impact=4, basis=fin.assumption("margin").basis, source="regels",
                              mitigation="Herbekijk prijsstelling of kostenstructuur."))
    if scr and scr.score_of(Criterion.FEASIBILITY) <= 2:
        risks.append(Risk(title="Regelgeving", description=next(
            s.motivation for s in scr.scores if s.criterion is Criterion.FEASIBILITY),
            likelihood=3, impact=5, basis=H, source="regels",
            mitigation="Laat vooraf juridisch toetsen welke vergunningen nodig zijn."))
    if scr and scr.score_of(Criterion.REACHABILITY) <= 2:
        risks.append(Risk(title="Klanten moeilijk bereikbaar",
                          description=next(s.motivation for s in scr.scores if s.criterion is Criterion.REACHABILITY),
                          likelihood=3, impact=4, basis=H, source="regels",
                          mitigation="Zoek een kanaal waar de doelgroep geconcentreerd is (partner, platform)."))
    if res is None or res.topics_with_facts <= 3:
        n = 0 if res is None else res.topics_with_facts
        risks.append(Risk(title="Onvoldoende bewijs",
                          description=f"Slechts {n}/10 onderzoeksonderwerpen hebben een geverifieerd feit; "
                                      "de analyse rust grotendeels op aannames.",
                          likelihood=4, impact=3, basis=F, source="regels",
                          mitigation="Voer de validatietest uit en voeg bronnen toe (onderzoek --url)."))
    return risks


def validation_test(idea: Idea, checks: list[InvestorCheck]) -> ValidationTest:
    rm = idea.finance.revenue_model if idea.finance else "abonnement_b2b"
    price = idea.finance.scenarios["realistisch"].price_per_month if idea.finance else None
    price_txt = f"€ {price:g}" if price else "de beoogde prijs"
    # Test the most critical open assumption first: without paying customers nothing else matters.
    by_q = {c.question: c for c in checks}
    first = next((q for q in (Q.WILL_PAY, Q.CAC_TOO_HIGH, Q.SATURATED) if by_q[q].alarming), Q.WILL_PAY)
    critical = {
        Q.WILL_PAY: f"{idea.target_customer} betalen {price_txt} per maand voor deze oplossing",
        Q.CAC_TOO_HIGH: f"{idea.target_customer} zijn goedkoop te bereiken en te overtuigen",
        Q.SATURATED: f"{idea.target_customer} kiezen voor deze oplossing boven bestaande alternatieven",
    }[first]
    t = idea.target_customer
    if rm == "abonnement_b2b":
        return ValidationTest(
            critical_assumption=critical,
            method="Probleeminterviews + pre-sale (zonder product)",
            steps=[f"Stel een lijst op van 40 {t} (beroepsfederatie, LinkedIn, lokale netwerken).",
                   "Voer 12 probleeminterviews over hun huidige aanpak, zonder je oplossing te verkopen "
                   "(vraag naar concrete voorbeelden en kosten, niet naar meningen).",
                   f"Maak een eenvoudige landingspagina met de belofte en de prijs ({price_txt}/maand).",
                   "Vraag de geïnteresseerden om een intentieverklaring of een voorafbetaling voor een pilot."],
            success_criteria=["Minstens 8 van 12 herkennen het probleem en noemen concrete tijd- of geldverliezen.",
                              f"Minstens 3 schriftelijke intentieverklaringen of voorafbetalingen aan ≥ {price_txt}/maand.",
                              "Minstens 1 kanaal waarlangs je 10 gesprekken kon plannen zonder betaalde advertenties."],
            kill_criteria=["Minder dan 4 van 12 herkennen het probleem.",
                           "Niemand wil vooraf iets toezeggen of betalen."],
            duration_weeks=4, cost_eur="€ 0 (optioneel ~€ 15 voor een domeinnaam)")
    if rm == "abonnement_b2c":
        return ValidationTest(
            critical_assumption=critical, method="Wachtlijst met prijs ('fake door') via gratis kanalen",
            steps=["Maak een landingspagina met de belofte, de prijs en een inschrijfknop.",
                   f"Deel ze in 5-10 online groepen of fora waar {t} actief zijn (volg hun regels).",
                   "Spreek 10 inschrijvers kort aan over hun probleem en de prijs."],
            success_criteria=["≥ 300 bezoekers en ≥ 8% inschrijvingen op de wachtlijst.",
                              "≥ 5 van 10 gesprekken bevestigen betalingsbereidheid aan de getoonde prijs."],
            kill_criteria=["< 3% inschrijvingen na 300 bezoekers.", "Kanalen laten geen promotie toe en er is geen alternatief."],
            duration_weeks=3, cost_eur="€ 0")
    if rm == "transactie":
        return ValidationTest(
            critical_assumption=f"Beide kanten ({t} en hun tegenpartij) willen via jou handelen en de commissie betalen",
            method="Concierge-MVP: koppelingen handmatig regelen in één regio",
            steps=["Kies één regio en één niche.", "Werf 10 aanbieders en 10 vragers persoonlijk.",
                   "Regel de eerste transacties manueel (e-mail, WhatsApp) en reken de commissie aan."],
            success_criteria=["≥ 5 afgeronde transacties waarbij de commissie betaald is.",
                              "≥ 30% van de deelnemers doet een tweede transactie."],
            kill_criteria=["< 2 transacties na 6 weken.", "Partijen gaan na de eerste koppeling buiten jou om."],
            duration_weeks=6, cost_eur="€ 0")
    return ValidationTest(
        critical_assumption=critical, method="Voorverkoop vóór productie",
        steps=["Maak een productpagina met prijs en levertermijn.", f"Bied ze persoonlijk aan bij 30 {t}.",
               "Vraag een voorafbetaling of reservering."],
        success_criteria=["≥ 10 voorverkopen of reserveringen."], kill_criteria=["< 3 voorverkopen."],
        duration_weeks=4, cost_eur="€ 0")


def review_by_rules(idea: Idea) -> CriticalReview:
    checks = investor_checks(idea)
    risks = sorted(rule_risks(idea, checks), key=lambda r: r.severity, reverse=True)[:5]
    fatal = [f"{r.title}: {r.description}" for r in risks if r.fatal]
    alarming = sum(c.alarming for c in checks)
    summary = (f"{alarming}/6 investeerdersvragen zijn alarmerend of onbeantwoord. "
               + ("Fatale gebreken gevonden." if fatal else "Geen fatale gebreken op basis van bewijs.")
               + (f" Grootste risico: {risks[0].title.lower()}." if risks else ""))
    return CriticalReview(checks=checks, risks=risks, fatal_flaws=fatal, summary=summary,
                          validation_test=validation_test(idea, checks), reviewed_by="regels")
