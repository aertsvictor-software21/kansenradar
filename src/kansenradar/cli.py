"""Command-line interface (Dutch).

    python -m kansenradar genereer --aantal 10
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


def _line(idea) -> str:
    orig = f"orig. {idea.originality.score:>4.1f}" if idea.originality else ""
    return f"  [{idea.status.value:22s}] {idea.id}  {orig}  {idea.title}"


def cmd_ideeen(args, settings) -> int:
    ideas = _repo(settings).list_ideas(None if args.alle else IdeaStatus.GENERATED)
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
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kansenradar", description="Kansenradar: businessopportuniteiten BE/NL")
    p.add_argument("-v", "--uitgebreid", action="store_true", help="toon logberichten")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("genereer", help="genereer nieuwe ideeën")
    g.add_argument("--aantal", type=int, default=10, help="aantal invalshoeken (standaard 10)")
    g.add_argument("--per-invalshoek", type=int, default=2, help="ideeën per invalshoek (LLM-modus)")
    g.add_argument("--sector", type=_enum_arg(Sector), action="append", help="beperk tot sector (herhaalbaar)")
    g.add_argument("--strategie", type=_enum_arg(OriginalityStrategy), action="append")
    g.add_argument("--seed", type=int, default=0, help="toevalszaad voor reproduceerbaarheid")
    g.add_argument("--zonder-llm", action="store_true", help="forceer sjabloonmodus")
    g.set_defaults(func=cmd_genereer)

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
