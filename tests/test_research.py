import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from kansenradar.domain import (
    EvidenceType,
    Idea,
    IdeaStatus,
    OriginalityStrategy,
    ResearchTopic,
    Sector,
    SourceQuality,
)
from kansenradar.llm import FakeBackend, LLMClient, RunBudget
from kansenradar.research import (
    EurostatClient,
    FetchSkipped,
    LLMExtractor,
    MarketResearcher,
    PageFetcher,
    RuleExtractor,
    SearxngSearch,
    TavilySearch,
    classify_source,
    html_to_text,
    key_terms,
    market_size_evidence,
    nace_for,
    parse_jsonstat,
    research_ideas,
    verify_quote,
)
from kansenradar.research.extract import _LLMClaim, _LLMClaims
from kansenradar.research.fetch import FetchedPage
from kansenradar.storage import Repository, make_engine

EUROSTAT = json.loads((Path(__file__).parent / "fixtures" / "eurostat_sample.json").read_text())

PAGE_HTML = """<html><head><title>Bouwunie: administratie bij aannemers</title>
<meta property="article:published_time" content="2026-03-01T10:00:00Z"></head>
<body><nav>Menu Home Contact</nav><script>var x = 1;</script>
<p>Uit een bevraging blijkt dat kleine aannemers gemiddeld 6 uur per week besteden aan administratie en werfverslagen.</p>
<p>Veel aannemers gebruiken nog papier of Excel voor hun werfverslagen en meerwerk.</p>
<p>Dit staat los van alles.</p></body></html>"""


def idea(**kw) -> Idea:
    data = dict(
        title="Werfverslag uit spraak",
        one_liner="App die ingesproken notities omzet in werfverslagen en meerwerkfacturen.",
        problem="Kleine aannemers verliezen avonden aan werfverslagen en administratie.",
        target_customer="aannemers met 1-20 werknemers", sector=Sector.TRADITIONAL,
        business_model="abonnement per gebruiker", strategy=OriginalityStrategy.NEW_TECH_APPLICATION,
        status=IdeaStatus.SCREENED_PASS, research_priority=0.7,
    )
    data.update(kw)
    return Idea(**data)


def mock_web(routes: dict[str, httpx.Response], seen: list | None = None) -> httpx.Client:
    def handler(request: httpx.Request):
        if seen is not None:
            seen.append(str(request.url))
        for prefix, response in routes.items():
            if str(request.url).startswith(prefix):
                return response
        return httpx.Response(404)
    return httpx.Client(transport=httpx.MockTransport(handler))


def html(body: str = PAGE_HTML) -> httpx.Response:
    return httpx.Response(200, text=body, headers={"content-type": "text/html; charset=utf-8"})


# --- quote verification -------------------------------------------------------------

PAGE_TEXT = "In 2025 telde de sector 12.000 kleine aannemers. Zij besteden gemiddeld 6 uur per week aan administratie."


def test_verify_exact_and_case_accent_insensitive():
    assert verify_quote("telde de sector 12.000 kleine aannemers", PAGE_TEXT).found
    assert verify_quote("TELDE de séctor 12.000  kleine aannemers", PAGE_TEXT).found


def test_verify_near_exact_but_numbers_must_match():
    ok = verify_quote("Zij besteden gemiddeld 6 uur per week aan de administratie.", PAGE_TEXT)
    assert ok.found and ok.ratio < 1
    wrong = verify_quote("In 2025 telde de sector 20.000 kleine aannemers.", PAGE_TEXT)
    assert not wrong.found


def test_verify_rejects_invented_and_too_short_quotes():
    assert not verify_quote("Aannemers betalen graag 99 euro per maand voor software.", PAGE_TEXT).found
    assert verify_quote("aannemers", PAGE_TEXT).reason == "citaat te kort om te verifiëren"


# --- fetching -------------------------------------------------------------------------

def test_html_to_text_strips_noise_and_reads_meta():
    title, text, meta = html_to_text(PAGE_HTML)
    assert title.startswith("Bouwunie")
    assert "var x" not in text and "Menu Home" not in text
    assert "6 uur per week" in text
    assert meta["article:published_time"].startswith("2026")


def test_fetcher_respects_robots_and_content_type():
    http = mock_web({
        "https://geblokkeerd.be/robots.txt": httpx.Response(200, text="User-agent: *\nDisallow: /"),
        "https://open.be/robots.txt": httpx.Response(404),
        "https://open.be/pdf": httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"}),
        "https://open.be/pagina": html(),
    })
    f = PageFetcher(http=http)
    with pytest.raises(FetchSkipped, match="robots"):
        f.fetch("https://geblokkeerd.be/artikel")
    with pytest.raises(FetchSkipped, match="geen HTML"):
        f.fetch("https://open.be/pdf")
    page = f.fetch("https://open.be/pagina")
    assert page.published_at == datetime(2026, 3, 1, 10, tzinfo=UTC)
    assert f.fetch("https://open.be/pagina") is page  # cached


# --- open data -------------------------------------------------------------------------

def test_parse_jsonstat_and_latest_year():
    flat = parse_jsonstat(EUROSTAT)
    assert flat[("A", "ENT_NR", "F43", "BE", "2023")] == 83456
    assert ("A", "ENT_NR", "F43", "BE", "2024") not in flat  # missing value


def test_eurostat_evidence_is_verified_official_fact_plus_estimate():
    http = mock_web({"https://ec.europa.eu/eurostat": httpx.Response(200, json=EUROSTAT)})
    ev = market_size_evidence(idea(), EurostatClient(http=http), countries=("BE",))
    facts = [e for e in ev if e.type is EvidenceType.FACT]
    assert len(facts) == 2  # F41 and F43 (mock returns the same sample)
    assert facts[0].source_quality is SourceQuality.OFFICIAL and facts[0].verified
    assert "83.456 ondernemingen" in facts[0].claim and "2023" in facts[0].claim
    estimate = [e for e in ev if e.type is EvidenceType.ESTIMATE][0]
    assert estimate.topic == ResearchTopic.CUSTOMERS.value and "benaderd" in estimate.claim


def test_nace_mapping_and_caveat():
    assert nace_for(idea(target_customer="kleine boekhoudkantoren")).codes == ("M692",)
    assert "particuliere" in nace_for(idea(target_customer="particuliere verhuurders")).caveat
    assert nace_for(idea(target_customer="jonge ouders")) is None


def test_eurostat_down_gives_no_evidence():
    http = mock_web({"https://ec.europa.eu/eurostat": httpx.Response(503)})
    assert market_size_evidence(idea(), EurostatClient(http=http)) == []


# --- extraction ----------------------------------------------------------------------------

def page() -> FetchedPage:
    title, text, _ = html_to_text(PAGE_HTML)
    return FetchedPage(url="https://www.bouwunie.be/nieuws", title=title, text=text,
                       fetched_at=datetime(2026, 11, 1, tzinfo=UTC))


def test_rule_extractor_yields_verified_facts_about_the_question():
    r = RuleExtractor().extract(page(), ResearchTopic.PROBLEM,
                                "Hoeveel tijd kost administratie van werfverslagen aannemers?")
    assert r.evidence and all(e.type is EvidenceType.FACT for e in r.evidence)
    assert "6 uur per week" in r.evidence[0].quote  # number sentence ranked first
    assert all("los van alles" not in e.quote for e in r.evidence)


def test_llm_extractor_drops_invented_quotes():
    claims = _LLMClaims(claims=[
        _LLMClaim(claim="Aannemers besteden 6 uur per week aan administratie.",
                  quote="kleine aannemers gemiddeld 6 uur per week besteden aan administratie", relevant=True),
        _LLMClaim(claim="Aannemers willen 50 euro betalen.", quote="aannemers betalen graag 50 euro per maand",
                  relevant=True),
        _LLMClaim(claim="Irrelevant", quote="Veel aannemers gebruiken nog papier of Excel", relevant=False),
    ])
    llm = LLMClient(FakeBackend({"extract_claims": claims}), RunBudget(max_calls=5), free=True)
    r = LLMExtractor(llm).extract(page(), ResearchTopic.PROBLEM,
                                  "Hoeveel tijd kost administratie van werfverslagen voor aannemers?")
    assert len(r.evidence) == 1 and r.rejected_quotes == 1
    assert r.evidence[0].type is EvidenceType.FACT and r.evidence[0].source_quality is SourceQuality.INDUSTRY


def test_classify_source():
    assert classify_source("https://statbel.fgov.be/nl/x") is SourceQuality.OFFICIAL
    assert classify_source("https://www.reddit.com/r/belgium") is SourceQuality.COMMUNITY
    assert classify_source("https://www.tijd.be/artikel") is SourceQuality.MEDIA
    assert classify_source("https://www.voorbeeldsoftware.be") is SourceQuality.COMPANY


# --- search providers ---------------------------------------------------------------------

def test_search_providers_parse_results():
    sx = SearxngSearch("http://localhost:8888", http=mock_web({"http://localhost:8888/search": httpx.Response(
        200, json={"results": [{"title": "A", "url": "https://a.be", "content": "x"}, {"title": "geen url"}]})}))
    assert [h.url for h in sx.search("q")] == ["https://a.be"]
    tv = TavilySearch("k", http=mock_web({"https://api.tavily.com/search": httpx.Response(
        200, json={"results": [{"title": "B", "url": "https://b.nl", "content": "y"}]})}))
    assert tv.search("q")[0].title == "B"
    down = SearxngSearch("http://localhost:8888", http=mock_web({}))
    assert down.search("q") == []


# --- researcher & pipeline -----------------------------------------------------------------

def test_key_terms_exclude_target_words():
    kt = key_terms(idea())
    assert "aannemers" not in kt and "werfverslagen" in kt


def test_end_to_end_research_with_mocked_web():
    seen = []
    http = mock_web({
        "https://api.tavily.com/search": httpx.Response(200, json={"results": [
            {"title": "Bouwunie", "url": "https://www.bouwunie.be/nieuws", "content": ""},
            {"title": "Verboden", "url": "https://geblokkeerd.be/x", "content": ""}]}),
        "https://www.bouwunie.be/robots.txt": httpx.Response(404),
        "https://www.bouwunie.be/nieuws": html(),
        "https://geblokkeerd.be/robots.txt": httpx.Response(200, text="User-agent: *\nDisallow: /"),
        "https://ec.europa.eu/eurostat": httpx.Response(200, json=EUROSTAT),
    }, seen)
    researcher = MarketResearcher(search=TavilySearch("k", http=http), fetcher=PageFetcher(http=http),
                                  opendata=EurostatClient(http=http), pages_per_query=2)
    repo = Repository(make_engine("sqlite:///:memory:"))
    first, second = idea(), idea(title="Lagere prioriteit", research_priority=0.2)
    for i in (first, second):
        repo.save_idea(i)

    report = research_ideas(repo, researcher, top_n=1)
    assert [i.id for i in report.ideas] == [first.id]
    stored = repo.get_idea(first.id)
    assert stored.status is IdeaStatus.RESEARCHED
    r = stored.research
    assert r.queries_run == 10 and r.pages_fetched == 1 and r.pages_skipped == 1
    assert r.coverage[ResearchTopic.MARKET_SIZE].facts >= 2
    assert r.topics_with_facts >= 2
    assert any("concurrenten" not in q and "?" in q for q in r.open_questions)
    assert "zoeken: tavily" in r.mode and "Eurostat" in r.mode
    # robots.txt-blocked page itself was never requested
    assert "https://geblokkeerd.be/x" not in seen
    facts = [e for e in repo.evidence_for(first.id) if e.type is EvidenceType.FACT]
    assert all(e.source_url and e.quote and e.verified for e in facts)
    assert "geen gescreende" not in report.summary()


def test_offline_research_with_user_url_only():
    http = mock_web({"https://mijnbron.be/robots.txt": httpx.Response(404), "https://mijnbron.be/a": html()})
    researcher = MarketResearcher(fetcher=PageFetcher(http=http))
    summary, evidence = researcher.research(idea(), extra_urls=["https://mijnbron.be/a"])
    assert summary.queries_run == 0 and summary.pages_fetched == 1
    assert "geen webzoekfunctie" in summary.mode and "1 eigen bron" in summary.mode
    assert evidence


def test_llm_extractor_skips_unrelated_pages_without_calling_the_model():
    backend = FakeBackend({"extract_claims": _LLMClaims(claims=[])})
    llm = LLMClient(backend, RunBudget(max_calls=5), free=True)
    r = LLMExtractor(llm).extract(page(), ResearchTopic.PRICING, "Wat kost een hondenkapper in Gent?")
    assert r.evidence == [] and backend.calls == []



def test_interrupted_research_resumes_where_it_stopped():
    class FlakySearch:
        name = "test"

        def __init__(self, fail_at):
            self.fail_at, self.queries = fail_at, []

        def search(self, query, max_results=5):
            self.queries.append(query)
            if len(self.queries) == self.fail_at:
                raise KeyboardInterrupt  # the Chromebook switches off
            return [type("H", (), {"url": "https://mijnbron.be/a", "title": "", "snippet": ""})()]

    http = mock_web({"https://mijnbron.be/robots.txt": httpx.Response(404), "https://mijnbron.be/a": html()})
    repo = Repository(make_engine("sqlite:///:memory:"))
    first = idea()
    repo.save_idea(first)
    flaky = FlakySearch(fail_at=4)
    with pytest.raises(KeyboardInterrupt):
        research_ideas(repo, MarketResearcher(search=flaky, fetcher=PageFetcher(http=http)), top_n=1)
    saved = repo.get_idea(first.id)
    assert saved.research is None and len(saved.research_done) == 3
    evidence_before = len(repo.evidence_for(first.id))
    assert evidence_before > 0

    resumed = FlakySearch(fail_at=999)
    report = research_ideas(repo, MarketResearcher(search=resumed, fetcher=PageFetcher(http=http)), top_n=1)
    assert len(resumed.queries) == 7  # only the 7 remaining topics
    done = repo.get_idea(first.id)
    assert done.status is IdeaStatus.RESEARCHED and report.ideas[0].id == first.id
    assert done.research.topics_with_facts >= 1
