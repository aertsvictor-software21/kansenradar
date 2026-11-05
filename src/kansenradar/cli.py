"""Command-line interface (Dutch).

    python -m kansenradar run --aantal 10 --top 3     (alles in één keer)
    python -m kansenradar genereer --aantal 10
    python -m kansenradar screen
    python -m kansenradar onderzoek [--top 3] [--idee ID --url URL]
    python -m kansenradar ideeen [--alle]
    python -m kansenradar toon <idee-id>
    python -m kansenradar status
"""

from __future__ import annotations

import argparse
import logging
import sys

from kansenradar.config import get_settings
from kansenradar.domain import IdeaStatus, OriginalityStrategy, Sector
from kansenradar.generation import run_generation
from kansenradar.history import get_embedder
from kansenradar.competition import CompetitorAnalyzer, analyse_competition
from kansenradar.finance import analyse_finance
from kansenradar.review import review_ideas
from kansenradar.reporting import write_reports
from kansenradar.scoring import rank, score_ideas
from kansenradar.research import (
    EurostatClient,
    LLMExtractor,
    PageFetcher,
    MarketResearcher,
    RuleExtractor,
    build_search,
    research_ideas,
)
from kansenradar.screening import screen_ideas
from kansenradar.llm import PRESETS, RunBudget, build_llm_client
from kansenradar.storage import Repository, make_engine


def _enum_arg(enum):
    def parse(value: str):
        for member in enum:
            if value.upper() == member.name or value.lower() == member.value.lower():
                return member
        names = ", ".join(m.name for m in enum)
        raise argparse.ArgumentTypeError(f"Onbekende waarde '{value}'. Kies uit: {names}")
    return parse


def _repo(settings) -> Repository:
    return Repository(make_engine(settings.database_url))


def cmd_genereer(args, settings) -> int:
    budget = RunBudget(max_usd=settings.max_run_cost_usd, max_calls=settings.max_llm_calls_per_run)
    llm = None if args.zonder_llm else build_llm_client(settings, budget)
    embedder = get_embedder(settings.embedder)
    print(f"Modus: {'LLM via ' + settings.llm_provider if llm else 'sjablonen (geen LLM)'} | "
          f"embedder: {embedder.name}")
    report = run_generation(
        _repo(settings), embedder, llm, count=args.aantal, per_seed=args.per_invalshoek,
        sectors=args.sector or None, strategies=args.strategie or None, rng_seed=args.seed,
        budget=budget,
    )
    print(report.summary())
    print()
    for idea in report.ideas:
        print(_line(idea))
        if idea.status is not IdeaStatus.GENERATED and idea.dedup_note:
            print(f"      ↳ {idea.dedup_note}")
        elif idea.status is IdeaStatus.REJECTED_GENERIC:
            print(f"      ↳ {'; '.join(idea.quality_flags)}")
    if llm:
        print(f"\nLLM-aanroepen: {len(budget.entries)} | kosten: ${budget.spent_usd:.2f}")
    return 0


def cmd_screen(args, settings) -> int:
    budget = RunBudget(max_usd=settings.max_run_cost_usd, max_calls=settings.max_llm_calls_per_run)
    llm = None if args.zonder_llm else build_llm_client(settings, budget)
    report = screen_ideas(_repo(settings), llm, limit=args.max)
    print(report.summary())
    if report.mode == "regels" and report.ideas:
        print("Let op: regelgebaseerde screening kent marktomvang en concurrentie niet. 'DOOR' betekent\n"
              "hier: geen dealbreaker gevonden in de tekst (lage betrouwbaarheid), niet: kansrijk.")
    ranking = report.ranking()
    if ranking:
        print("\nVolgorde voor marktonderzoek:")
        for n, idea in enumerate(ranking, 1):
            s = idea.screening
            print(f"  {n:2d}. {idea.research_priority:.2f}  {s.decision.value:8s} {idea.id}  {idea.title}")
    rejected = [i for i in report.ideas if i.research_priority is None]
    if rejected:
        print("\nAfgewezen:")
        for idea in rejected:
            print(f"  - {idea.title}: {'; '.join(idea.screening.knockouts)}")
    if llm:
        print(f"\nLLM-aanroepen: {len(budget.entries)} | kosten: ${budget.spent_usd:.2f}")
    return 0


def build_researcher(settings, llm) -> MarketResearcher:
    return MarketResearcher(
        search=build_search(settings),
        extractor=LLMExtractor(llm) if llm else RuleExtractor(),
        opendata=EurostatClient() if settings.use_open_data else None,
        max_age_days=settings.evidence_max_age_days,
    )


def cmd_onderzoek(args, settings) -> int:
    budget = RunBudget(max_usd=settings.max_run_cost_usd, max_calls=settings.max_llm_calls_per_run)
    llm = None if args.zonder_llm else build_llm_client(settings, budget)
    researcher = build_researcher(settings, llm)
    ids = [args.idee] if args.idee else None
    urls = {args.idee: args.url} if args.idee and args.url else None
    if args.url and not args.idee:
        print("--url vereist --idee <idee-id>.")
        return 2
    report = research_ideas(_repo(settings), researcher, top_n=args.top, idea_ids=ids, extra_urls=urls)
    print(report.summary())
    for idea in report.ideas:
        print(f"\n{idea.title}\n  Modus: {idea.research.mode}")
        for topic, cov in idea.research.coverage.items():
            print(f"  {topic.value:32s} feiten {cov.facts}  schattingen {cov.estimates}  hypotheses {cov.hypotheses}")
        if idea.research.open_questions:
            print("  Open vragen (geen geverifieerd feit):")
            for q in idea.research.open_questions:
                print(f"   - {q}")
    return 0


def cmd_concurrentie(args, settings) -> int:
    budget = RunBudget(max_usd=settings.max_run_cost_usd, max_calls=settings.max_llm_calls_per_run)
    llm = None if args.zonder_llm else build_llm_client(settings, budget)
    analyzer = CompetitorAnalyzer(search=build_search(settings), fetcher=PageFetcher(), llm=llm)
    ideas = analyse_competition(_repo(settings), analyzer, [args.idee] if args.idee else None)
    if not ideas:
        print("Geen onderzochte ideeën zonder concurrentieanalyse. Voer eerst 'onderzoek' uit.")
    for idea in ideas:
        c = idea.competition
        print(f"\n{idea.title}\n  Modus: {c.mode}\n  Verzadiging: {c.saturation} — {c.saturation_reason}")
        if c.price_median_month is not None:
            print(f"  Prijzen (geverifieerd, per maand): min € {c.price_min_month:g}, mediaan "
                  f"€ {c.price_median_month:g}, max € {c.price_max_month:g} (n={c.verified_price_count})")
        for comp in c.competitors:
            tag = " [modelkennis, te verifiëren]" if comp.source == "modelkennis" else ""
            print(f"   - {comp.kind:9s} {comp.name} ({comp.domain}){tag}")
        for gap in c.gaps:
            print(f"   ? gat (hypothese): {gap}")
    return 0


def _eur(x: float | None) -> str:
    return "—" if x is None else f"€ {x:,.0f}".replace(",", ".")


def print_finance(idea) -> None:
    f = idea.finance
    print(f"\n{idea.title}\n  Verdienmodel: {f.revenue_model} | schaalbaarheid: {f.scalability} — {f.scalability_reason}")
    print("  Aannames:")
    for a in f.assumptions:
        print(f"   - {a.label}: {a.low:g} / {a.mid:g} / {a.high:g} {a.unit} [{a.basis.value}] {a.source}")
    rows = [
        ("Startkapitaal", lambda r: _eur(r.startup_capital)),
        ("Vaste kosten/maand", lambda r: _eur(r.monthly_fixed_costs)),
        ("Marketing/maand", lambda r: _eur(r.marketing_per_month)),
        ("CAC", lambda r: _eur(r.cac)),
        ("Klanten na 12 / 36 mnd", lambda r: f"{r.customers_m12} / {r.customers_m36}"),
        ("Omzet jaar 1", lambda r: _eur(r.revenue_year1)),
        ("Omzet jaar 3", lambda r: _eur(r.revenue_year3)),
        ("Brutomarge", lambda r: f"{r.gross_margin_pct:.0f}%"),
        ("Netto jaar 3*", lambda r: _eur(r.net_year3)),
        ("Break-even (maand)", lambda r: str(r.operational_break_even_month or "> 36")),
        ("Terugverdiend (maand)", lambda r: str(r.payback_month or "> 36")),
        ("Benodigd kapitaal", lambda r: _eur(r.capital_needed)),
        ("LTV / CAC", lambda r: f"{r.ltv_cac:.1f}"),
    ]
    sc = f.scenarios
    print(f"  {'':24s}{'conservatief':>15s}{'realistisch':>15s}{'optimistisch':>15s}")
    for label, fn in rows:
        print(f"  {label:24s}" + "".join(f"{fn(sc[n]):>15s}" for n in ("conservatief", "realistisch", "optimistisch")))
    for n in f.notes:
        print(f"  * {n}")


def cmd_financieel(args, settings) -> int:
    ideas = analyse_finance(_repo(settings), [args.idee] if args.idee else None, args.marketing)
    if not ideas:
        print("Geen onderzochte ideeën zonder financiële analyse. Voer eerst 'onderzoek' uit.")
    for idea in ideas:
        print_finance(idea)
    return 0


def print_review(idea) -> None:
    r = idea.review
    print(f"\n{idea.title}\n  {r.summary}  (door {r.reviewed_by})")
    for c in r.checks:
        flag = "!" if c.alarming else " "
        print(f"  {flag} {c.question.value:48s} {c.verdict:9s} {c.explanation}")
    print("  Vijf grootste risico's:")
    for k in r.risks:
        fatal = "  FATAAL" if k.fatal else ""
        print(f"   - [{k.likelihood}x{k.impact}={k.severity:2d}] {k.title} [{k.basis.value}]{fatal}: {k.description}")
        print(f"       beperking: {k.mitigation}")
    v = r.validation_test
    print(f"  Eerste validatietest: {v.method} ({v.duration_weeks} weken, {v.cost_eur})")
    print(f"    Te toetsen: {v.critical_assumption}")
    for st in v.steps:
        print(f"    • {st}")
    print("    Geslaagd als: " + " | ".join(v.success_criteria))
    print("    Stoppen als:  " + " | ".join(v.kill_criteria))


def cmd_kritiek(args, settings) -> int:
    budget = RunBudget(max_usd=settings.max_run_cost_usd, max_calls=settings.max_llm_calls_per_run)
    llm = None if args.zonder_llm else build_llm_client(settings, budget)
    ideas = review_ideas(_repo(settings), llm, [args.idee] if args.idee else None)
    if not ideas:
        print("Geen onderzochte ideeën zonder kritische beoordeling. Voer eerst 'onderzoek' uit.")
    for idea in ideas:
        print_review(idea)
    return 0


def print_scorecard(idea) -> None:
    sc = idea.scorecard
    print(f"\n{sc.conclusion.value} — {sc.total:.0f}/100, betrouwbaarheid {sc.confidence:.0f}%\n  {idea.title} ({idea.id})")
    for s in sc.subscores:
        cap = f"  [plafond: {s.capped_by}]" if s.capped_by else ""
        print(f"   {s.criterion:25s}{s.points:5.1f}/{s.max_points:<3d} {s.rationale}{cap}")
    print(f"  Conclusie: {sc.conclusion_reason}")
    print("  Betrouwbaarheid: " + " ".join(sc.confidence_reasons))


def cmd_score(args, settings) -> int:
    ranked = score_ideas(_repo(settings), [args.idee] if args.idee else None, settings.evidence_max_age_days)
    if not ranked:
        print("Geen onderzochte ideeën om te scoren. Voer eerst 'onderzoek' uit.")
    for idea in ranked:
        print_scorecard(idea)
    return 0


def cmd_ranking(args, settings) -> int:
    ranked = rank(_repo(settings).list_ideas())
    if not ranked:
        print("Nog geen gescoorde ideeën.")
    for n, idea in enumerate(ranked, 1):
        sc = idea.scorecard
        print(f"{n:3d}. {sc.conclusion.value:32s} {sc.total:5.1f}/100  betr. {sc.confidence:3.0f}%  {idea.id}  {idea.title}")
    return 0


def cmd_rapport(args, settings) -> int:
    index = write_reports(_repo(settings), args.map)
    print(f"Dashboard en rapporten geschreven: {index.resolve()}")
    print("Open dit bestand in je browser. PDF: Afdrukken -> Opslaan als PDF.")
    return 0


def cmd_run(args, settings) -> int:
    """Complete pipeline: generate -> screen -> research -> competition -> finance -> critique -> score -> report."""
    budget = RunBudget(max_usd=settings.max_run_cost_usd, max_calls=settings.max_llm_calls_per_run)
    llm = None if args.zonder_llm else build_llm_client(settings, budget)
    repo = _repo(settings)
    embedder = get_embedder(settings.embedder)
    print(f"Modus: {'LLM via ' + settings.llm_provider if llm else 'zonder LLM'} | zoeken: "
          f"{build_search(settings).name} | open data: {'aan' if settings.use_open_data else 'uit'}")
    steps = [
        ("1/8 Genereren", lambda: run_generation(repo, embedder, llm, count=args.aantal, rng_seed=args.seed,
                                                 budget=budget).summary()),
        ("2/8 Screenen", lambda: screen_ideas(repo, llm).summary()),
        ("3/8 Marktonderzoek", lambda: research_ideas(repo, build_researcher(settings, llm), top_n=args.top).summary()),
        ("4/8 Concurrentie", lambda: f"{len(analyse_competition(repo, CompetitorAnalyzer(search=build_search(settings), llm=llm)))} idee(ën)"),
        ("5/8 Financieel", lambda: f"{len(analyse_finance(repo))} idee(ën)"),
        ("6/8 Kritische investeerder", lambda: f"{len(review_ideas(repo, llm))} idee(ën)"),
        ("7/8 Scoren", lambda: f"{len(score_ideas(repo, max_age_days=settings.evidence_max_age_days))} idee(ën) gescoord"),
        ("8/8 Rapport", lambda: str(write_reports(repo, args.map).resolve())),
    ]
    for label, step in steps:
        print(f"\n== {label} ==\n{step()}")
    print("\nRangschikking:")
    cmd_ranking(args, settings)
    if llm:
        print(f"\nLLM-aanroepen: {len(budget.entries)} | kosten: ${budget.spent_usd:.2f}")
    return 0


def cmd_kalibreer(args, settings) -> int:
    from pathlib import Path

    from kansenradar.calibration import DEFAULT_CASES, report, run_calibration
    results = run_calibration(Path(args.bestand) if args.bestand else DEFAULT_CASES)
    print(report(results))
    return 0 if all(r.ok for r in results) else 1


def _line(idea) -> str:
    orig = f"orig. {idea.originality.score:>4.1f}" if idea.originality else ""
    prio = f"prio {idea.research_priority:.2f}" if idea.research_priority is not None else ""
    return f"  [{idea.status.value:22s}] {idea.id}  {orig}  {prio:9s} {idea.title}"


def cmd_ideeen(args, settings) -> int:
    statuses = None if args.alle else {IdeaStatus.GENERATED, IdeaStatus.SCREENED_PASS,
                                        IdeaStatus.SCREENED_DOUBT}
    ideas = [i for i in _repo(settings).list_ideas() if statuses is None or i.status in statuses]
    ideas.sort(key=lambda i: i.research_priority or -1, reverse=True)
    if not ideas:
        print("Geen ideeën gevonden.")
    for idea in ideas:
        print(_line(idea))
    return 0


def cmd_toon(args, settings) -> int:
    idea = _repo(settings).get_idea(args.idee_id)
    if idea is None:
        print(f"Idee '{args.idee_id}' niet gevonden.")
        return 1
    rows = [
        ("Naam", idea.title), ("Omschrijving", idea.one_liner), ("Probleem", idea.problem),
        ("Doelgroep", idea.target_customer), ("Sector", idea.sector.value),
        ("Verdienmodel", idea.business_model), ("Waarom nu", idea.why_now),
        ("Onderscheidend", idea.differentiator), ("Inspiratie (hypothese)", idea.inspiration),
        ("Strategie", idea.strategy.value), ("Herkomst", f"{idea.generated_by} / {idea.seed_key}"),
        ("Status", idea.status.value), ("Duplicaatcontrole", idea.dedup_note),
        ("Kwaliteitssignalen", "; ".join(idea.quality_flags) or None),
    ]
    if idea.screening:
        sc = idea.screening
        rows.append(("Screening", f"{sc.decision.value} — gewogen {sc.weighted_score:.2f}/5, "
                                  f"betrouwbaarheid {sc.confidence}, door {sc.screened_by}"))
        for cs in sc.scores:
            mark = "" if cs.known else " (onbekend)"
            rows.append((f"  {cs.criterion.value}", f"{cs.score}/5{mark} — {cs.motivation}"))
        if sc.knockouts:
            rows.append(("  Dealbreakers", "; ".join(sc.knockouts)))
        if sc.key_assumptions:
            rows.append(("  Aannames", "; ".join(sc.key_assumptions)))
        rows.append(("  Onderzoeksvragen", " | ".join(sc.research_questions)))
    if idea.originality:
        rows.append(("Originaliteit", f"{idea.originality.score}/10 ({idea.originality.label}) — "
                                      f"{idea.originality.explanation}"))
    for label, value in rows:
        if value:
            print(f"{label + ':':24s}{value}")
    return 0


def cmd_status(args, settings) -> int:
    import os

    print(f"Database:      {settings.database_url}")
    print(f"LLM-provider:  {settings.llm_provider}")
    preset = PRESETS.get(settings.llm_provider)
    if preset and preset.api_key_env:
        ok = "ingesteld" if os.environ.get(preset.api_key_env) else "ONTBREEKT"
        print(f"API-sleutel:   {preset.api_key_env} {ok}")
    embedder = get_embedder(settings.embedder)
    print(f"Embedder:      {embedder.name}")
    print(f"Webzoeken:     {build_search(settings).name}")
    print(f"Open data:     {'Eurostat' if settings.use_open_data else 'uit'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kansenradar", description="Kansenradar: businessopportuniteiten BE/NL")
    p.add_argument("-v", "--uitgebreid", action="store_true", help="toon logberichten")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="volledige pijplijn: genereren t/m rapport")
    r.add_argument("--aantal", type=int, default=10, help="aantal invalshoeken")
    r.add_argument("--top", type=int, default=3, help="aantal ideeën voor diepgaand onderzoek")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--map", default="rapporten")
    r.add_argument("--zonder-llm", action="store_true")
    r.set_defaults(func=cmd_run)

    g = sub.add_parser("genereer", help="genereer nieuwe ideeën")
    g.add_argument("--aantal", type=int, default=10, help="aantal invalshoeken (standaard 10)")
    g.add_argument("--per-invalshoek", type=int, default=2, help="ideeën per invalshoek (LLM-modus)")
    g.add_argument("--sector", type=_enum_arg(Sector), action="append", help="beperk tot sector (herhaalbaar)")
    g.add_argument("--strategie", type=_enum_arg(OriginalityStrategy), action="append")
    g.add_argument("--seed", type=int, default=0, help="toevalszaad voor reproduceerbaarheid")
    g.add_argument("--zonder-llm", action="store_true", help="forceer sjabloonmodus")
    g.set_defaults(func=cmd_genereer)

    sc = sub.add_parser("screen", help="screen nieuwe ideeën (snelle eerste beoordeling)")
    sc.add_argument("--max", type=int, default=None, help="maximum aantal ideeën")
    sc.add_argument("--zonder-llm", action="store_true", help="forceer regelgebaseerde screening")
    sc.set_defaults(func=cmd_screen)

    o = sub.add_parser("onderzoek", help="marktonderzoek voor de beste gescreende ideeën")
    o.add_argument("--top", type=int, default=3, help="aantal ideeën (standaard 3)")
    o.add_argument("--idee", help="onderzoek één specifiek idee")
    o.add_argument("--url", action="append", help="eigen bron-URL (herhaalbaar, met --idee)")
    o.add_argument("--zonder-llm", action="store_true")
    o.set_defaults(func=cmd_onderzoek)

    k = sub.add_parser("concurrentie", help="concurrentieanalyse voor onderzochte ideeën")
    k.add_argument("--idee", help="één specifiek idee")
    k.add_argument("--zonder-llm", action="store_true")
    k.set_defaults(func=cmd_concurrentie)

    f = sub.add_parser("financieel", help="financiële analyse (3 scenario's)")
    f.add_argument("--idee", help="één specifiek idee (herberekenen)")
    f.add_argument("--marketing", type=float, default=750, help="marketingbudget per maand (EUR)")
    f.set_defaults(func=cmd_financieel)

    kr = sub.add_parser("kritiek", help="kritische investeerder + eerste validatietest")
    kr.add_argument("--idee", help="één specifiek idee (opnieuw beoordelen)")
    kr.add_argument("--zonder-llm", action="store_true")
    kr.set_defaults(func=cmd_kritiek)

    so = sub.add_parser("score", help="scoor onderzochte ideeën (100 punten + betrouwbaarheid)")
    so.add_argument("--idee", help="één specifiek idee")
    so.set_defaults(func=cmd_score)

    rk = sub.add_parser("ranking", help="rangschikking van alle gescoorde ideeën")
    rk.set_defaults(func=cmd_ranking)

    rp = sub.add_parser("rapport", help="schrijf dashboard (index.html) en rapporten per idee")
    rp.add_argument("--map", default="rapporten", help="uitvoermap (standaard: rapporten)")
    rp.set_defaults(func=cmd_rapport)

    kb = sub.add_parser("kalibreer", help="toets de beoordelingslogica op gelabelde gevallen")
    kb.add_argument("--bestand", help="JSON met gevallen (standaard: kalibratie/archetypes.json)")
    kb.set_defaults(func=cmd_kalibreer)

    i = sub.add_parser("ideeen", help="toon ideeën")
    i.add_argument("--alle", action="store_true", help="ook afgewezen en duplicaten")
    i.set_defaults(func=cmd_ideeen)

    t = sub.add_parser("toon", help="toon één idee in detail")
    t.add_argument("idee_id")
    t.set_defaults(func=cmd_toon)

    s = sub.add_parser("status", help="toon configuratie")
    s.set_defaults(func=cmd_status)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.uitgebreid else logging.WARNING,
                        format="%(levelname)s %(message)s")
    return args.func(args, get_settings())


if __name__ == "__main__":
    sys.exit(main())
