import httpx

from kansenradar.competition import CompetitorAnalyzer, extract_prices
from kansenradar.competition.analyzer import _LLMCompetitor, _LLMCompetitors, _LLMKnown, _LLMKnownList
from kansenradar.domain import EvidenceType, Idea, IdeaStatus, OriginalityStrategy, ResearchTopic, Sector
from kansenradar.llm import FakeBackend, LLMClient, RunBudget
from kansenradar.research import PageFetcher, TavilySearch


def idea() -> Idea:
    return Idea(
        title="Werfverslag uit spraak",
        one_liner="App die ingesproken notities omzet in werfverslagen en meerwerkfacturen.",
        problem="Kleine aannemers verliezen avonden aan werfverslagen en administratie.",
        target_customer="aannemers met 1-20 werknemers", sector=Sector.TRADITIONAL,
        business_model="abonnement per gebruiker", strategy=OriginalityStrategy.NEW_TECH_APPLICATION,
        status=IdeaStatus.RESEARCHED,
    )


def page(title, body):
    return httpx.Response(200, text=f"<html><head><title>{title}</title></head><body>{body}</body></html>",
                          headers={"content-type": "text/html"})


def web(results):
    routes = {
        "https://api.tavily.com/search": httpx.Response(200, json={"results": results}),
        "https://werfapp.be/robots.txt": httpx.Response(404),
        "https://werfapp.be/": page("WerfApp | Werfverslagen voor aannemers",
            "<p>WerfApp maakt werfverslagen en meerwerkfacturen voor aannemers, gewoon vanaf de werf.</p>"
            "<p>Starter € 29 per maand, Pro € 59 per maand per gebruiker.</p>"),
        "https://bouwtool.nl/robots.txt": httpx.Response(404),
        "https://bouwtool.nl/": page("Bouwtool - administratie voor aannemers",
            "<p>Bouwtool regelt werfverslagen, administratie en facturen voor aannemers en bouwbedrijven.</p>"),
        "https://bouwtool.nl/prijzen": page("Prijzen", "<p>Bouwtool kost 348 euro per jaar per bedrijf.</p>"),
        "https://excel.example.com/robots.txt": httpx.Response(404),
        "https://excel.example.com/": page("Spreadsheet", "<p>Een algemene rekenblad-toepassing voor iedereen die cijfers verwerkt.</p>"),
    }

    def handler(request):
        u = str(request.url)
        for prefix, r in routes.items():
            if u == prefix or (u.startswith(prefix) and not prefix.endswith("/")):
                return r
        return httpx.Response(404)
    return httpx.Client(transport=httpx.MockTransport(handler))


RESULTS = [
    {"title": "WerfApp", "url": "https://werfapp.be/", "content": ""},
    {"title": "Bouwtool", "url": "https://bouwtool.nl/", "content": ""},
    {"title": "Review", "url": "https://www.capterra.com/x", "content": ""},
    {"title": "Nieuws", "url": "https://www.tijd.be/artikel", "content": ""},
    {"title": "Spreadsheet", "url": "https://excel.example.com/", "content": ""},
]


def analyzer(llm=None, results=RESULTS):
    http = web(results)
    return CompetitorAnalyzer(search=TavilySearch("k", http=http), fetcher=PageFetcher(http=http), llm=llm)


def test_price_extraction_formats():
    pts = extract_prices("Starter € 29 per maand. Pro €49,95/mnd. Jaar: 290 euro per jaar. Licentie EUR 1.250 eenmalig.", "u")
    assert [(p.amount_eur, p.period) for p in pts] == [(29, "maand"), (49.95, "maand"), (290, "jaar"), (1250, "eenmalig")]
    assert pts[2].monthly_eur == 24.17 and pts[3].monthly_eur is None
    assert extract_prices("In 2024 groeide het met 3 procent.", "u") == []


def test_web_analysis_excludes_media_and_reviews_and_reads_prices():
    analysis, evidence = analyzer().analyse(idea())
    domains = {c.domain for c in analysis.competitors}
    assert domains == {"werfapp.be", "bouwtool.nl", "excel.example.com"}
    kinds = {c.domain: c.kind for c in analysis.competitors}
    assert kinds["werfapp.be"] == "direct" and kinds["bouwtool.nl"] == "direct"
    assert kinds["excel.example.com"] == "indirect"
    assert analysis.saturation == "gemiddeld"
    # prices: 29, 59 (werfapp) + 348/yr = 29/mo (bouwtool, from /prijzen)
    assert analysis.verified_price_count == 3
    assert analysis.price_min_month == 29 and analysis.price_max_month == 59
    price_facts = [e for e in evidence if e.topic == ResearchTopic.PRICING.value]
    assert price_facts and all(e.type is EvidenceType.FACT and e.quote for e in price_facts)


def test_no_search_means_unknown_saturation_not_low():
    analysis, evidence = CompetitorAnalyzer().analyse(idea())
    assert analysis.saturation == "onbekend" and analysis.competitors == [] and evidence == []


def test_llm_classification_can_drop_irrelevant_and_adds_gaps():
    verdicts = _LLMCompetitors(competitors=[
        _LLMCompetitor(index=0, kind="direct", description="Werfverslagen-app.", strengths=["eenvoudig"], weaknesses=["geen spraak"]),
        _LLMCompetitor(index=1, kind="indirect", description="Algemene bouwadministratie.", strengths=[], weaknesses=[]),
        _LLMCompetitor(index=2, kind="geen", description="Rekenblad.", strengths=[], weaknesses=[]),
    ], gaps=["Geen enkele concurrent ondersteunt spraakinvoer (hypothese)."])
    llm = LLMClient(FakeBackend({"competitors": verdicts}), RunBudget(max_calls=5), free=True)
    analysis, _ = analyzer(llm).analyse(idea())
    assert {c.domain for c in analysis.competitors} == {"werfapp.be", "bouwtool.nl"}
    assert analysis.saturation == "laag"  # one direct competitor
    assert analysis.gaps and "llm-beoordeling" in analysis.mode
    assert analysis.competitors[0].weaknesses == ["geen spraak"]


def test_offline_llm_known_competitors_are_hypotheses_only():
    known = _LLMKnownList(competitors=[_LLMKnown(name="X", domain="x.be", description="d", kind="direct")])
    llm = LLMClient(FakeBackend({"known_competitors": known}), RunBudget(max_calls=5), free=True)
    analysis, evidence = CompetitorAnalyzer(llm=llm).analyse(idea())
    assert analysis.saturation == "onbekend"
    assert analysis.price_median_month is None
    assert [e.type for e in evidence] == [EvidenceType.HYPOTHESIS]
