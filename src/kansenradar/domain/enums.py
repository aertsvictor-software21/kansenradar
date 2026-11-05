"""Domain enums. Values are the Dutch labels shown to users."""

from __future__ import annotations

from enum import StrEnum


class EvidenceType(StrEnum):
    FACT = "FEIT"
    ESTIMATE = "SCHATTING"
    HYPOTHESIS = "HYPOTHESE"


class SourceQuality(StrEnum):
    """Reliability class of a source, used in the confidence score."""

    OFFICIAL = "officieel"        # government statistics, regulators, registries
    INDUSTRY = "sector"           # industry reports, trade associations
    COMPANY = "bedrijf"           # a company's own website / pricing page
    MEDIA = "media"               # news, trade press
    COMMUNITY = "gemeenschap"     # forums, reviews, Reddit
    UNKNOWN = "onbekend"


class Conclusion(StrEnum):
    REJECT = "AFWIJZEN"
    MORE_RESEARCH = "MEER ONDERZOEK NODIG"
    EXPERIMENT = "EXPERIMENTEREN"
    PROMISING = "KANSRIJK"
    VERY_INTERESTING = "ZEER INTERESSANTE OPPORTUNITEIT"


class Sector(StrEnum):
    SAAS = "Software en SaaS"
    AI = "Kunstmatige intelligentie"
    ECOMMERCE = "E-commerce"
    DIGITAL_PRODUCTS = "Digitale producten"
    MARKETPLACES = "Online marktplaatsen"
    PROPTECH = "Vastgoedtechnologie"
    HEALTHTECH = "Gezondheidszorgtechnologie"
    LOGISTICS = "Logistiek"
    EDUCATION = "Onderwijs"
    PROFESSIONAL_SERVICES = "Professionele dienstverlening"
    CONSUMER_PRODUCTS = "Consumentenproducten"
    TRADITIONAL = "Traditionele sector (technologische vernieuwing)"


class IdeaStatus(StrEnum):
    GENERATED = "gegenereerd"
    DUPLICATE = "duplicaat"
    SCREENED_OUT = "afgewezen bij screening"
    IN_RESEARCH = "in onderzoek"
    ANALYSED = "geanalyseerd"
