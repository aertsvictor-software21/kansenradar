"""Deterministic filter for generic ideas (works without an LLM).

An idea is rejected as generic when its target customer is a catch-all
("bedrijven", "consumenten"), or when it piles up vague signals: buzzwords,
"Uber voor X" phrasing, or a problem described too briefly to be concrete.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kansenradar.domain import IdeaCandidate
from kansenradar.history.embeddings import normalize

BROAD_TARGETS = {
    "iedereen", "bedrijven", "consumenten", "kmo s", "kmos", "kmo", "ondernemers", "particulieren",
    "mensen", "klanten", "gebruikers", "bedrijven en particulieren", "b2b", "b2c", "het grote publiek",
    "alle bedrijven", "zelfstandigen", "organisaties", "mkb", "mkb s", "mkbs", "mkb ers", "mkb bedrijven",
    "kmo s en mkb s", "mkb s en kmo s", "kleine bedrijven", "ondernemingen", "kmo bedrijven",
}
# Words that do not make a target customer more specific.
FILLER = {"in", "en", "de", "het", "van", "voor", "uit", "nederland", "nederlandse", "belgie", "belgische",
          "benelux", "vlaanderen", "vlaamse", "wallonie", "brussel", "europa", "alle", "kleine", "middelgrote",
          "grote", "lokale"}
BUZZWORDS = (
    "uber voor", "airbnb voor", "netflix voor", "tinder voor", "revolutionair", "alles in een",
    "one stop shop", "innovatief platform", "slimme oplossing", "disruptief", "blockchain",
    "ai gestuurd platform", "game changer", "voor iedereen",
)
MIN_PROBLEM_WORDS = 8


@dataclass
class QualityCheck:
    flags: list[str] = field(default_factory=list)
    is_generic: bool = False


def check_quality(c: IdeaCandidate) -> QualityCheck:
    flags: list[str] = []
    target = normalize(c.target_customer)
    core = " ".join(w for w in target.split() if w not in FILLER)
    if target in BROAD_TARGETS or core in BROAD_TARGETS or not core:
        flags.append(f"doelgroep te breed ('{c.target_customer}')")
    text = " " + normalize(f"{c.title} {c.one_liner} {c.differentiator or ''}") + " "
    for bw in BUZZWORDS:
        if f" {bw} " in text:
            flags.append(f"modewoord/clichéformule: '{bw}'")
    if len(normalize(c.problem).split()) < MIN_PROBLEM_WORDS:
        flags.append("probleem te vaag of te kort omschreven")
    if not ({"BE", "NL"} & {m.upper() for m in c.markets}):
        flags.append("niet gericht op België of Nederland")
    if c.generated_by == "llm" and not (c.differentiator and c.differentiator.strip()):
        flags.append("geen onderscheidend vermogen opgegeven")

    hard = any(f.startswith(("doelgroep te breed", "niet gericht")) for f in flags)
    return QualityCheck(flags=flags, is_generic=hard or len(flags) >= 2)
