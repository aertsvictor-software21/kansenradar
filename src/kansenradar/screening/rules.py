"""Rule-based screener (no LLM).

Honest by design: it only scores what the text of the idea actually reveals
(pain signals, B2B vs. B2C, revenue model, regulation, hardware). Everything
else is 'onbekend' (score 3, known=False) and becomes a research question.
Confidence is always 'laag'.
"""

from __future__ import annotations

from kansenradar.domain import Criterion, CriterionScore, IdeaCandidate, ScreeningResult
from kansenradar.history.embeddings import normalize
from kansenradar.screening.decision import decide

PAIN_SIGNALS = (
    "verlies", "verliezen", "kost", "kosten", "fout", "fouten", "boete", "conflict", "uren", "dagen",
    "avonden", "marge", "omzet", "wachtlijst", "leeg", "lege", "retour", "onbetaald", "lopen vast",
    "vastlopen", "vast in", "risico", "weggooien", "verspil", "missen", "mislopen", "trage", "traag",
    " duur ", "te duur",
)
REGULATED = (
    "diagnos", "radiolog", "klinische beslissing", "medisch hulpmiddel", "medical device", "geneesmiddel", "voorschrift", "krediet", "lening", "belegg",
    "verzeker", "crypto", "kansspel", "gokken", "betalingsdienst", "bankrekening", "alcohol",
    "tabak", "financiering", "patientengegevens", "medische gegevens",
)
# Special-category personal data (GDPR art. 9): not a knock-out, but a real barrier.
SENSITIVE_DATA = ("patient", "zorg", "verpleegkundig", "therapeut", "kinesi", "artsen", "medisch",
                  "gezondheid", "huisarts", "tandarts", "dossier")
HARDWARE = ("hardware", "sensor", "robot", "drone", "fabricage", "productielijn", "toestel", "apparaat")
B2C_TARGETS = (
    "ouders", "consument", "gezinnen", "studenten", "mantelzorgers", "kopers", "patienten", "huurders",
    "eigenaars van", "baasjes", "particulieren", "senioren", "jongeren",
)
B2B_TARGETS = (
    "kantor", "praktijk", "aannemer", "installateur", "bedrijv", "zaken", "groothandel", "webshop",
    "producent", "bakkerij", "advoca", "syndic", "beheerder", "verhuurder", "werkgever", "kmo",
    "technici", "verpleegkundig", "therapeut", "kinesi", "landbouw", "tuinbouw", "horeca",
    "boekhoud", "artsen", "vervoerder", "handel", "starters in een vrij beroep", "vakmens", "kapper",
    "garage", "restaurant", "cafe", "renovatie", "loodgieter", "schilder", "elektricien",
)
REVENUE_WORDS = ("abonnement", "prijs", "commissie", "licentie", "per ", "betaal", "fee", "verkoop",
                 "aankoop", "huur", "marge op")
MARKETPLACE_WORDS = ("commissie", "marktplaats", "koppelt", "koppelen", "platform dat")


def _has(text: str, words) -> list[str]:
    return [w for w in words if w in text]


def _has_word_start(text: str, words) -> list[str]:
    """Match only at the start of a word: 'lening' must not match 'dienstverlening'."""
    import re
    return [w for w in words if re.search(r"(?<![a-z0-9])" + re.escape(w), text)]


def screen_by_rules(c: IdeaCandidate) -> ScreeningResult:
    problem = " " + normalize(c.problem) + " "
    target = normalize(c.target_customer)
    model = normalize(c.business_model) + " "
    whole = normalize(" ".join([c.title, c.one_liner, c.problem, c.business_model, c.why_now or ""]))
    assumptions: list[str] = []
    questions = [
        f"Hoeveel {c.target_customer} zijn er in België en Nederland?",
        f"Hoe lossen {c.target_customer} dit vandaag op, en met welke tools of dienstverleners?",
        "Welke prijzen vragen bestaande alternatieven?",
    ]

    pain_hits = _has(problem, PAIN_SIGNALS)
    pain = CriterionScore(
        criterion=Criterion.PAIN, score=4 if len(pain_hits) >= 2 else 3 if pain_hits else 2,
        known=bool(pain_hits),
        motivation=(f"Probleem bevat concrete pijnsignalen ({', '.join(pain_hits[:4])})." if pain_hits
                    else "Probleem noemt geen concrete kosten, tijdverlies of risico."),
    )

    is_b2c = bool(_has(target, B2C_TARGETS)) and not _has(target, ("verhuurder", "beheerder"))
    is_b2b = bool(_has(target, B2B_TARGETS)) and not is_b2c
    revenue = _has(model, REVENUE_WORDS)
    if not revenue:
        wtp = CriterionScore(criterion=Criterion.WILLINGNESS_TO_PAY, score=1, known=True,
                             motivation="Geen duidelijk verdienmodel opgegeven.")
    elif is_b2b:
        wtp = CriterionScore(criterion=Criterion.WILLINGNESS_TO_PAY, score=4, known=True,
                             motivation="Zakelijke doelgroep met een terugkerend of duidelijk verdienmodel.")
        assumptions.append(f"{c.target_customer} willen betalen via: {c.business_model}.")
    else:
        wtp = CriterionScore(criterion=Criterion.WILLINGNESS_TO_PAY, score=3, known=False,
                             motivation="Betalingsbereidheid onbekend" + (" (consumenten)." if is_b2c else "."))
        assumptions.append(f"{c.target_customer} zijn bereid te betalen ({c.business_model}).")

    marketplace = bool(_has(model, MARKETPLACE_WORDS)) or "marktplaats" in whole
    reach_score = 4 if is_b2b else 2 if is_b2c else 3
    reach_reason = ("Afgebakende beroepsgroep, bereikbaar via sectorkanalen (hypothese)." if is_b2b
                    else "Consumenten bereiken is doorgaans duur (hypothese)." if is_b2c
                    else "Bereikbaarheid onbekend.")
    if marketplace:
        reach_score = max(1, reach_score - 1)
        reach_reason += " Tweezijdige marktplaats: kip-en-ei-probleem bij de start."
        assumptions.append("Beide kanten van de marktplaats kunnen tegelijk opgebouwd worden.")
        questions.append("Welke kant van de marktplaats is het moeilijkst te werven, en hoe?")
    reach = CriterionScore(criterion=Criterion.REACHABILITY, score=reach_score,
                           known=is_b2b or is_b2c, motivation=reach_reason)

    size = CriterionScore(criterion=Criterion.MARKET_SIZE, score=3, known=False,
                          motivation="Marktomvang onbekend zonder data; te onderzoeken.")
    competition = CriterionScore(criterion=Criterion.COMPETITION, score=3, known=False,
                                 motivation="Concurrentie onbekend zonder onderzoek.")

    regulated = _has_word_start(whole, REGULATED)
    hardware = _has_word_start(whole, HARDWARE)
    if regulated:
        feas = CriterionScore(criterion=Criterion.FEASIBILITY, score=2, known=True,
                              motivation=f"Mogelijk vergunning of toezicht nodig ({', '.join(regulated[:3])}).")
        questions.append("Welke vergunningen, certificeringen of toezichthouders zijn van toepassing?")
    elif _has_word_start(whole, SENSITIVE_DATA):
        feas = CriterionScore(criterion=Criterion.FEASIBILITY, score=3, known=True,
                              motivation="Verwerkt waarschijnlijk gezondheidsgegevens: strengere AVG-eisen "
                                         "en mogelijk koppelingen met zorgsystemen.")
        questions.append("Welke AVG-eisen en koppelingen met zorgsystemen zijn nodig, en wat kost dat?")
    elif hardware:
        feas = CriterionScore(criterion=Criterion.FEASIBILITY, score=3, known=True,
                              motivation="Hardware vraagt meer startkapitaal en doorlooptijd.")
    else:
        feas = CriterionScore(criterion=Criterion.FEASIBILITY, score=4, known=True,
                              motivation="Softwaregedreven; beperkt startkapitaal (hypothese).")

    scores = [pain, wtp, reach, size, competition, feas]
    unknown = [s.criterion.value for s in scores if not s.known]
    summary = ("Regelgebaseerde screening op de tekst van het idee. Onbekend: "
               + (", ".join(unknown) if unknown else "niets") + ".")
    return decide(scores, screened_by="regels", key_assumptions=assumptions,
                  research_questions=questions, summary=summary)
