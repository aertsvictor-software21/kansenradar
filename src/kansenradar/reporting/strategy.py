"""Deterministic pricing and acquisition strategy texts for the report (Dutch)."""

from __future__ import annotations

from kansenradar.domain import Idea
from kansenradar.history.embeddings import normalize


def pricing_strategy(idea: Idea) -> list[str]:
    comp, fin = idea.competition, idea.finance
    lines = []
    if comp and comp.verified_price_count >= 2:
        lines.append(f"Concurrenten vragen € {comp.price_min_month:g} tot € {comp.price_max_month:g} per maand "
                     f"(mediaan € {comp.price_median_month:g}, {comp.verified_price_count} geverifieerde prijzen).")
        lines.append(f"Advies: start rond de mediaan (€ {comp.price_median_month:g}) en positioneer op het "
                     "onderscheidende voordeel in plaats van op de laagste prijs; een prijsoorlog is voor een "
                     "starter niet te winnen.")
    else:
        price = fin.scenarios["realistisch"].price_per_month if fin else None
        lines.append("Er zijn geen geverifieerde concurrentieprijzen; de prijs is een hypothese"
                     + (f" (werkhypothese € {price:g} per maand)." if price else "."))
        lines.append("Advies: toets de prijs expliciet in de validatietest (toon de prijs op de landingspagina "
                     "en vraag om vooruitbetaling), in plaats van achteraf te vragen wat klanten zouden betalen.")
    if fin and fin.revenue_model == "abonnement_b2b":
        lines.append("Overweeg een jaarabonnement met korting (betere cashflow, minder klantverloop) en een "
                     "betaalde pilot in plaats van een gratis proefperiode.")
    elif fin and fin.revenue_model == "transactie":
        lines.append("Bij een commissiemodel: houd de commissie laag genoeg zodat partijen niet buiten het "
                     "platform om gaan, en bied extra waarde (betaling, garantie, administratie).")
    return lines


def acquisition_strategy(idea: Idea) -> list[str]:
    target = normalize(idea.target_customer)
    fin = idea.finance
    b2c = fin is not None and fin.revenue_model in ("abonnement_b2c", "eenmalig") and "verhuurder" not in target
    lines = []
    if fin and fin.revenue_model == "transactie":
        lines.append("Start met de moeilijkste kant van de marktplaats in één regio en één niche; "
                     "werf de eerste deelnemers persoonlijk (concierge).")
    if b2c:
        lines += ["Online gemeenschappen en fora waar de doelgroep actief is (waarde bieden, niet spammen).",
                  "Zoekmachineoptimalisatie op concrete probleemzoekopdrachten.",
                  "Doorverwijzing door tevreden klanten (beloning per aanbreng)."]
    else:
        lines += [f"Rechtstreeks contact met {idea.target_customer} via beroepsfederaties, vakbeurzen en lokale netwerken.",
                  "Partnerships met partijen die de doelgroep al bedienen (boekhouders, groothandels, softwareleveranciers).",
                  "Inhoud die het probleem concreet maakt (rekenvoorbeelden, checklists) voor zoekverkeer en LinkedIn.",
                  "Doorverwijzing: in beroepsgroepen werkt mond-tot-mond sterk."]
    if fin:
        r = fin.scenarios["realistisch"]
        basis = fin.assumption("cac").basis.value
        lines.append(f"Werkhypothese klantenwervingskosten: € {r.cac:g} per klant [{basis}]; "
                     f"terugverdiend in {r.cac_payback_months or '—'} maanden. Meet dit in de eerste campagne.")
    if idea.research and idea.research.coverage:
        from kansenradar.domain import ResearchTopic
        if not idea.research.coverage[ResearchTopic.REACH].has_fact:
            lines.append("De bereikbaarheid via deze kanalen is nog niet met bronnen bevestigd.")
    return lines
