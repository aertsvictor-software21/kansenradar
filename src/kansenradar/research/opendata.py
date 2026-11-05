"""Official open data (free, no key): Eurostat structural business statistics.

Dataset sbs_ovw_act, indicator ENT_NR = number of enterprises, per NACE Rev. 2
activity and country (BE, NL), years 2021-2024 (status 2026-09).

The value is read directly from the API, so it is recorded as a verified FEIT with
the API URL as source. The mapping from target customer to NACE code is an
approximation, which is recorded separately as a SCHATTING.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product

import httpx

from kansenradar.domain import Evidence, EvidenceType, IdeaCandidate, ResearchTopic, SourceQuality
from kansenradar.history.embeddings import normalize

log = logging.getLogger(__name__)

EUROSTAT_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/sbs_ovw_act"


@dataclass(frozen=True)
class NaceMapping:
    keywords: tuple[str, ...]
    codes: tuple[str, ...]
    label: str
    caveat: str = ""


NACE_MAP: tuple[NaceMapping, ...] = (
    NaceMapping(("installateur",), ("F432",), "bouwinstallatie (elektro, sanitair, HVAC)"),
    NaceMapping(("aannemer", "bouwbedrijf"), ("F41", "F43"),
                "bouw van gebouwen en gespecialiseerde bouwwerkzaamheden"),
    NaceMapping(("boekhoud",), ("M692",), "boekhouding, accountancy en belastingadvies"),
    NaceMapping(("advoca",), ("M691",), "rechtskundige dienstverlening"),
    NaceMapping(("huisarts", "tandarts", "artsenpraktijk"), ("Q862",), "medische en tandheelkundige praktijken"),
    NaceMapping(("kinesi", "fysio", "therapeut", "verpleegkundig"), ("Q869",), "overige menselijke gezondheidszorg"),
    NaceMapping(("horeca", "restaurant", "cafe"), ("I56",), "eet- en drinkgelegenheden"),
    NaceMapping(("transportbedrij", "vervoerder", "vrachtwagen"), ("H494",), "goederenvervoer over de weg"),
    NaceMapping(("groothandel",), ("G46",), "groothandel"),
    NaceMapping(("webshop",), ("G4791",), "detailhandel via postorder of internet"),
    NaceMapping(("bakkerij",), ("C1071",), "vervaardiging van brood en vers banketbakkerswerk"),
    NaceMapping(("voedingsproducent",), ("C10",), "vervaardiging van voedingsmiddelen"),
    NaceMapping(("syndic", "vme", "vve"), ("L6832",), "beheer van onroerend goed voor derden"),
    NaceMapping(("verhuurder",), ("L682",), "verhuur van eigen onroerend goed",
                "particuliere verhuurders (natuurlijke personen) worden hier niet geteld"),
)


def nace_for(candidate: IdeaCandidate) -> NaceMapping | None:
    target = normalize(candidate.target_customer)
    return next((m for m in NACE_MAP if any(k in target for k in m.keywords)), None)


def parse_jsonstat(data: dict) -> dict[tuple[str, ...], float]:
    """Flatten a JSON-stat 2.0 dataset into {(code per dimension...): value}."""
    dims = data["id"]
    sizes = data["size"]
    index = {d: data["dimension"][d]["category"]["index"] for d in dims}
    by_pos = {d: {pos: code for code, pos in (index[d].items() if isinstance(index[d], dict)
                                              else ((c, i) for i, c in enumerate(index[d])))}
              for d in dims}
    values = data.get("value", {})
    if isinstance(values, list):
        values = {str(i): v for i, v in enumerate(values) if v is not None}
    strides = [1] * len(dims)
    for i in range(len(dims) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    out = {}
    for combo in product(*(range(s) for s in sizes)):
        flat = sum(p * st for p, st in zip(combo, strides))
        v = values.get(str(flat))
        if v is not None:
            out[tuple(by_pos[d][p] for d, p in zip(dims, combo))] = float(v)
    return out


class EurostatClient:
    def __init__(self, http: httpx.Client | None = None) -> None:
        self.http = http or httpx.Client(timeout=30)

    def enterprise_count(self, country: str, nace: str) -> tuple[int, str, str] | None:
        """Latest available (count, year, url) or None."""
        params = {"format": "JSON", "lang": "en", "geo": country, "nace_r2": nace,
                  "indic_sbs": "ENT_NR", "freq": "A"}
        try:
            r = self.http.get(EUROSTAT_URL, params=params)
            r.raise_for_status()
            flat = parse_jsonstat(r.json())
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            log.warning("Eurostat niet beschikbaar voor %s/%s: %s", country, nace, exc)
            return None
        time_pos = r.json()["id"].index("time")
        by_year = {k[time_pos]: v for k, v in flat.items()}
        if not by_year:
            return None
        year = max(by_year)
        return int(by_year[year]), year, str(r.url)


def market_size_evidence(candidate: IdeaCandidate, client: EurostatClient,
                         countries: tuple[str, ...] = ("BE", "NL")) -> list[Evidence]:
    mapping = nace_for(candidate)
    if mapping is None:
        return []
    now = datetime.now(UTC)
    evidence: list[Evidence] = []
    totals = {}
    for country in countries:
        for code in mapping.codes:
            result = client.enterprise_count(country, code)
            if result is None:
                continue
            count, year, url = result
            totals[country] = totals.get(country, 0) + count
            quote = f"ENT_NR {country} {code} {year} = {count}"
            evidence.append(Evidence(
                claim=f"Volgens Eurostat telde {'België' if country == 'BE' else 'Nederland'} in {year} "
                      f"{count:,} ondernemingen in NACE {code} ({mapping.label}), alle groottes.".replace(",", "."),
                type=EvidenceType.FACT, topic=ResearchTopic.MARKET_SIZE.value,
                source_url=url, source_title="Eurostat sbs_ovw_act (Enterprises - number)",
                source_quality=SourceQuality.OFFICIAL, quote=quote, verified=True, retrieved_at=now,
            ))
    if evidence:
        caveat = f" Let op: {mapping.caveat}." if mapping.caveat else ""
        evidence.append(Evidence(
            claim=f"De doelgroep '{candidate.target_customer}' is benaderd met NACE "
                  f"{', '.join(mapping.codes)} ({mapping.label}); het werkelijke aantal potentiële "
                  f"klanten is waarschijnlijk kleiner (grootteklasse, specialisatie).{caveat}",
            type=EvidenceType.ESTIMATE, topic=ResearchTopic.CUSTOMERS.value,
            reasoning=f"Afgeleid van officiële aantallen: {totals}. Afbakening op trefwoorden "
                      f"{', '.join(mapping.keywords)}.",
        ))
    return evidence
