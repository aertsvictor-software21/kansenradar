"""Curated seed catalog for the originality engine (Dutch content).

IMPORTANT: every entry is a HYPOTHESIS used as a starting point for idea
generation. Nothing here counts as evidence. Market research (phase 4) must
verify any claim before it can become a FEIT; regulatory triggers in particular
must be checked for exact scope and dates.
"""

from __future__ import annotations

from dataclasses import dataclass

from kansenradar.domain import Sector


@dataclass(frozen=True)
class ProblemSeed:
    key: str
    sector: Sector
    target_customer: str
    problem: str
    typical_model: str
    deliverable: str = ""            # concrete output, used in template one-liners
    tech_fit: tuple[str, ...] = ()   # TechLens keys that make sense for this problem


@dataclass(frozen=True)
class TechLens:
    key: str
    capability: str
    solution_pattern: str  # "{verb}" phrase used by the template generator


@dataclass(frozen=True)
class InternationalPattern:
    key: str
    region: str
    pattern: str


@dataclass(frozen=True)
class RegulatoryTrigger:
    key: str
    markets: tuple[str, ...]
    description: str


S = Sector
PROBLEM_SEEDS: tuple[ProblemSeed, ...] = (
    ProblemSeed("saas-boekhouder-documenten", S.SAAS, "kleine boekhoudkantoren (1-10 medewerkers)",
                "Boekhouders verliezen veel tijd met het opvragen en opvolgen van ontbrekende documenten bij klanten.",
                "abonnement per kantoor",
                deliverable="ontbrekende documenten en herinneringen naar klanten", tech_fit=('whatsapp', 'documentextractie', 'agent')),
    ProblemSeed("saas-vve-beheer", S.SAAS, "vrijwillige syndici en kleine VME/VvE-beheerders",
                "Kleine mede-eigendommen worden beheerd met e-mail en Excel, met trage besluitvorming en onduidelijke kosten.",
                "abonnement per gebouw",
                deliverable="beslissingen, kostenverdeling en communicatie met eigenaars", tech_fit=('whatsapp', 'documentextractie', 'peppol')),
    ProblemSeed("ai-tenders", S.AI, "kmo's die inschrijven op overheidsopdrachten",
                "Het lezen van lastenboeken en opstellen van offertes voor overheidsopdrachten kost kmo's dagen werk per dossier.",
                "abonnement plus prijs per dossier",
                deliverable="een samenvatting van het lastenboek en een eerste offerteversie", tech_fit=('documentextractie', 'agent')),
    ProblemSeed("ai-zorgverslag", S.AI, "zelfstandige thuisverpleegkundigen",
                "Thuisverpleegkundigen besteden een groot deel van hun dag aan verslaggeving en administratie in plaats van zorg.",
                "maandabonnement per verpleegkundige",
                deliverable="zorgverslagen en administratie", tech_fit=('spraak', 'agent')),
    ProblemSeed("ecom-retouren", S.ECOMMERCE, "kleine en middelgrote webshops in mode en schoenen",
                "Retouren door verkeerde maten drukken de marge van kleine webshops sterk.",
                "prijs per bestelling of per vermeden retour",
                deliverable="persoonlijk maatadvies vóór de aankoop", tech_fit=('voorspelling', 'camera')),
    ProblemSeed("ecom-b2b-herbestellen", S.ECOMMERCE, "groothandels met vaste klanten in horeca en retail",
                "Vaste B2B-klanten bestellen nog via telefoon, e-mail of WhatsApp, met fouten en handmatig overtypen tot gevolg.",
                "abonnement per groothandel",
                deliverable="correcte bestellingen in het systeem van de groothandel", tech_fit=('whatsapp', 'documentextractie', 'voorspelling')),
    ProblemSeed("dig-sjablonen-zelfstandigen", S.DIGITAL_PRODUCTS, "starters in een vrij beroep",
                "Starters in vrije beroepen weten niet welke contracten, voorwaarden en administratie ze juridisch correct nodig hebben.",
                "eenmalige aankoop plus updates",
                deliverable="juridisch correcte contracten en voorwaarden op maat", tech_fit=('agent', 'documentextractie')),
    ProblemSeed("mkt-reststromen", S.MARKETPLACES, "voedingsproducenten en bakkerijen",
                "Producenten gooien bruikbare reststromen weg omdat er geen eenvoudige lokale afnemer te vinden is.",
                "commissie per transactie",
                deliverable="afspraken met lokale afnemers van reststromen", tech_fit=('whatsapp', 'planning', 'voorspelling')),
    ProblemSeed("mkt-vervangers-zorg", S.MARKETPLACES, "huisartsenpraktijken en tandartspraktijken",
                "Praktijken vinden moeilijk vervangers bij ziekte of vakantie en zoeken via informele netwerken.",
                "abonnement voor praktijken, gratis voor vervangers",
                deliverable="geschikte vervangers voor open periodes", tech_fit=('planning', 'whatsapp')),
    ProblemSeed("prop-plaatsbeschrijving", S.PROPTECH, "particuliere verhuurders met 1-10 panden",
                "Plaatsbeschrijvingen en huurwaarborgdiscussies kosten verhuurders tijd en leiden tot conflicten.",
                "prijs per plaatsbeschrijving",
                deliverable="een volledige plaatsbeschrijving met handtekening", tech_fit=('camera', 'spraak')),
    ProblemSeed("prop-renovatieplicht", S.PROPTECH, "kopers van oudere woningen",
                "Kopers van een oude woning weten niet wat een verplichte of wenselijke energierenovatie concreet kost en in welke volgorde.",
                "betaald rapport plus commissie van aannemers",
                deliverable="een gefaseerd renovatieplan met kostenraming", tech_fit=('camera', 'documentextractie', 'voorspelling')),
    ProblemSeed("health-wachtlijst-kine", S.HEALTHTECH, "kinesitherapeuten en fysiotherapeuten",
                "Lege plekken door late annuleringen blijven onbenut terwijl er wachtlijsten zijn.",
                "abonnement per praktijk",
                deliverable="opgevulde afspraken uit de wachtlijst", tech_fit=('planning', 'voorspelling', 'whatsapp')),
    ProblemSeed("health-mantelzorg", S.HEALTHTECH, "werkende mantelzorgers",
                "Mantelzorgers coördineren zorg, afspraken en premies versnipperd en missen tegemoetkomingen waar ze recht op hebben.",
                "abonnement of aanbieden via werkgever/ziekenfonds",
                deliverable="een overzicht van afspraken, taken en tegemoetkomingen", tech_fit=('agent', 'whatsapp')),
    ProblemSeed("log-retourritten", S.LOGISTICS, "kleine transportbedrijven met 1-15 vrachtwagens",
                "Kleine vervoerders rijden vaak leeg terug omdat ze geen toegang hebben tot geschikte retourvrachten.",
                "commissie per rit",
                deliverable="passende retourvrachten", tech_fit=('planning', 'voorspelling')),
    ProblemSeed("log-bouwlevering", S.LOGISTICS, "aannemers en bouwmaterialenhandels",
                "Leveringen op werven lopen mis door slechte afstemming van tijdstip, toegang en losplaats.",
                "abonnement voor handelaars",
                deliverable="afgestemde leveringsmomenten op de werf", tech_fit=('planning', 'whatsapp')),
    ProblemSeed("edu-bijscholing-ambacht", S.EDUCATION, "installateurs en technici",
                "Technici moeten bijscholen voor nieuwe technieken (warmtepompen, laadpalen) maar opleidingen zijn duur en ver weg.",
                "abonnement per technicus, betaald door werkgever",
                deliverable="korte praktijkgerichte bijscholingsmodules", tech_fit=('camera', 'spraak')),
    ProblemSeed("edu-nt2-werkvloer", S.EDUCATION, "werkgevers met anderstalige medewerkers",
                "Anderstalige medewerkers leren de taal te traag voor veiligheid en communicatie op de werkvloer.",
                "abonnement per medewerker",
                deliverable="taaloefeningen op basis van de eigen werkvloer", tech_fit=('spraak', 'whatsapp')),
    ProblemSeed("prof-advocaat-intake", S.PROFESSIONAL_SERVICES, "kleine advocatenkantoren",
                "Advocaten besteden veel onbetaalde tijd aan intakegesprekken met potentiële cliënten die geen dossier worden.",
                "abonnement per kantoor",
                deliverable="een gestructureerd intakedossier met eerste inschatting", tech_fit=('spraak', 'documentextractie', 'agent')),
    ProblemSeed("prof-installateur-offerte", S.PROFESSIONAL_SERVICES, "installateurs van zonnepanelen en warmtepompen",
                "Offertes opmaken vraagt een plaatsbezoek en veel rekenwerk, waardoor installateurs veel tijd steken in offertes die niet doorgaan.",
                "abonnement per installateur",
                deliverable="een offerte op basis van foto's en verbruiksgegevens", tech_fit=('camera', 'documentextractie')),
    ProblemSeed("cons-huisdier-voeding", S.CONSUMER_PRODUCTS, "eigenaars van honden met allergieën of overgewicht",
                "Baasjes vinden moeilijk passende voeding en verspillen geld aan proberen.",
                "abonnement op voeding",
                deliverable="een voedingsplan en passende voeding", tech_fit=('voorspelling', 'camera')),
    ProblemSeed("cons-babyspullen-huur", S.CONSUMER_PRODUCTS, "jonge ouders",
                "Babyspullen worden maar enkele maanden gebruikt maar wel duur gekocht.",
                "huurabonnement",
                deliverable="tijdige wissels van babyspullen per leeftijdsfase", tech_fit=('planning', 'voorspelling')),
    ProblemSeed("trad-werfverslag", S.TRADITIONAL, "aannemers met 1-20 werknemers",
                "Kleine aannemers verliezen avonden aan werfverslagen, meerwerk en facturatie.",
                "abonnement per gebruiker",
                deliverable="werfverslagen en meerwerkfacturen", tech_fit=('spraak', 'camera', 'peppol')),
    ProblemSeed("trad-landbouw-administratie", S.TRADITIONAL, "land- en tuinbouwbedrijven",
                "Land- en tuinbouwers lopen vast in steeds zwaardere registratie- en aangifteverplichtingen.",
                "jaarabonnement per bedrijf",
                deliverable="registraties en aangiftes", tech_fit=('documentextractie', 'agent')),
    ProblemSeed("trad-horeca-personeel", S.TRADITIONAL, "zelfstandige horecazaken",
                "Horecazaken plannen personeel en flexi-jobs nog handmatig en lopen fouten op in aangiftes.",
                "abonnement per zaak",
                deliverable="personeelsplanning en correcte aangiftes", tech_fit=('planning', 'agent')),
)

TECH_LENSES: tuple[TechLens, ...] = (
    TechLens("spraak", "spraakherkenning en samenvatting", "zet ingesproken notities automatisch om in"),
    TechLens("documentextractie", "documentextractie met taalmodellen", "haalt gegevens uit documenten en maakt automatisch"),
    TechLens("camera", "computervisie via de smartphonecamera", "maakt op basis van smartphonefoto's automatisch"),
    TechLens("whatsapp", "WhatsApp als interface", "regelt volledig via WhatsApp"),
    TechLens("planning", "automatische planning en optimalisatie", "plant en optimaliseert automatisch"),
    TechLens("agent", "AI-agent die formulieren en portalen invult", "handelt via een AI-agent automatisch af:"),
    TechLens("voorspelling", "voorspellende modellen op eigen data", "voorspelt op basis van eigen gegevens"),
    TechLens("peppol", "gestructureerde e-facturatie (Peppol)", "koppelt e-facturen automatisch aan"),
)

INTERNATIONAL_PATTERNS: tuple[InternationalPattern, ...] = (
    InternationalPattern("verticale-saas-ambacht", "Verenigde Staten",
                         "Verticale software voor één specifiek ambacht (planning, offertes, facturatie en betalingen in één)"),
    InternationalPattern("embedded-finance", "Verenigde Staten",
                         "Verticale software die verdient aan geïntegreerde betalingen, financiering of verzekering"),
    InternationalPattern("managed-marketplace", "Verenigd Koninkrijk / Verenigde Staten",
                         "Een beheerde marktplaats die kwaliteit garandeert in plaats van enkel vraag en aanbod te koppelen"),
    InternationalPattern("roll-up-software", "Verenigde Staten",
                         "Overname en digitalisering van kleine traditionele dienstverleners met eigen software"),
    InternationalPattern("abonnement-onderhoud", "Scandinavië",
                         "Onderhoud en service als abonnement in plaats van losse interventies"),
    InternationalPattern("b2b-inkoop", "Duitsland",
                         "Digitaal inkoopplatform dat versnipperde B2B-bestellingen van kleine afnemers bundelt"),
)

REGULATORY_TRIGGERS: tuple[RegulatoryTrigger, ...] = (
    RegulatoryTrigger("be-peppol", ("BE",),
                      "Verplichte gestructureerde B2B-e-facturatie in België (Peppol), gefaseerd ingevoerd vanaf 2026"),
    RegulatoryTrigger("vl-renovatieplicht", ("BE",),
                      "Renovatieverplichting na aankoop van energetisch slechte woningen en gebouwen in Vlaanderen"),
    RegulatoryTrigger("eu-ai-act", ("BE", "NL"),
                      "Gefaseerde verplichtingen uit de Europese AI-verordening (AI Act) voor aanbieders en gebruikers van AI"),
    RegulatoryTrigger("eu-nis2", ("BE", "NL"),
                      "Cyberbeveiligingsverplichtingen uit NIS2 voor middelgrote organisaties in kritieke sectoren"),
    RegulatoryTrigger("eu-csrd-keten", ("BE", "NL"),
                      "Duurzaamheidsrapportering bij grote bedrijven (CSRD) die gegevens opvragen bij kleinere leveranciers"),
    RegulatoryTrigger("nl-wet-dba", ("NL",),
                      "Strengere handhaving op schijnzelfstandigheid in Nederland (Wet DBA)"),
)
