import re

from kansenradar.domain import IdeaStatus
from kansenradar.reporting import render_dashboard, render_idea_report, write_reports
from kansenradar.scoring import score_idea
from kansenradar.storage import Repository, make_engine
from tests.test_scoring import ResearchTopic, build, fact

SECTIONS = ["Naam van het concept", "Korte beschrijving", "Het opgeloste probleem", "Doelgroep",
            "Marktpotentieel", "Concurrenten", "Verdienmodel", "Prijsstrategie", "Financiële prognose",
            "Klantenwervingsstrategie", "Concurrentievoordeel", "Risico's", "Eerste validatietest",
            "Eindbeoordeling"]


def analysed(**kw):
    ev = [fact(t) for t in list(ResearchTopic)[:5]]
    idea, ev = build(ev, saturation="gemiddeld", prices=3, **kw)
    idea.scorecard = score_idea(idea, ev)
    idea.status = IdeaStatus.ANALYSED
    return idea, ev


def test_report_has_all_14_sections_in_order_and_sources():
    idea, ev = analysed()
    html = render_idea_report(idea, ev)
    headings = re.findall(r"<h2>(\d+)\. ([^<]+)</h2>", html)
    assert [int(n) for n, _ in headings] == list(range(1, 15))
    assert [t.replace("&#x27;", "'") for _, t in headings] == SECTIONS
    assert "Bronnen en bewijs" in html and "geen beleggingsadvies" in html
    assert 'class="badge FEIT"' in html and "https://bron.be" in html
    assert "conservatief" in html and "€ 45" in html  # scenario table and pricing advice


def test_report_escapes_untrusted_text():
    idea, ev = analysed(title="<script>alert(1)</script> Idee",
                        differentiator='<img src=x onerror="alert(1)">')
    html = render_idea_report(idea, ev)
    assert "<script>alert" not in html and "<img src=x" not in html
    assert "&lt;script&gt;" in html


def test_dashboard_funnel_ranking_and_filters():
    idea, _ = analysed()
    dup, _ = build(title="Dubbel idee", status=IdeaStatus.DUPLICATE)
    html = render_dashboard([idea, dup], [])
    assert "Trechter" in html and "Rangschikking" in html
    assert f"ideeen/{idea.id}.html" in html
    assert 'id="q"' in html and 'id="fc"' in html
    assert "Nog geen runs" in html


def test_write_reports_creates_files(tmp_path):
    repo = Repository(make_engine("sqlite:///:memory:"))
    idea, ev = analysed()
    repo.save_idea(idea)
    repo.add_evidence(idea.id, ev)
    index = write_reports(repo, tmp_path)
    assert index.exists() and (tmp_path / "ideeen" / f"{idea.id}.html").exists()


def test_cli_full_pipeline_offline(tmp_path, monkeypatch, capsys):
    from kansenradar.cli import main
    from kansenradar.config import get_settings
    monkeypatch.setenv("KR_DATABASE_URL", f"sqlite:///{tmp_path / 'kr.db'}")
    monkeypatch.setenv("KR_EMBEDDER", "hashing")
    monkeypatch.setenv("KR_LLM_PROVIDER", "none")
    monkeypatch.setenv("KR_SEARCH_PROVIDER", "none")
    monkeypatch.setenv("KR_USE_OPEN_DATA", "false")
    get_settings.cache_clear()
    try:
        assert main(["run", "--aantal", "5", "--top", "2", "--map", str(tmp_path / "rap")]) == 0
        out = capsys.readouterr().out
        assert "== 8/8 Rapport ==" in out and "MEER ONDERZOEK NODIG" in out
        assert len(list((tmp_path / "rap" / "ideeen").glob("*.html"))) == 2
    finally:
        get_settings.cache_clear()


def test_cli_reanalyse_wipes_old_evidence(tmp_path, monkeypatch, capsys):
    from kansenradar.cli import main
    from kansenradar.config import get_settings
    from kansenradar.storage import Repository, make_engine
    db = tmp_path / "kr.db"
    for k, v in {"KR_DATABASE_URL": f"sqlite:///{db}", "KR_EMBEDDER": "hashing", "KR_LLM_PROVIDER": "none",
                 "KR_SEARCH_PROVIDER": "none", "KR_USE_OPEN_DATA": "false"}.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    try:
        main(["run", "--aantal", "3", "--top", "1", "--map", str(tmp_path / "r")])
        repo = Repository(make_engine(f"sqlite:///{db}"))
        idea = next(i for i in repo.list_ideas() if i.status is IdeaStatus.ANALYSED)
        from kansenradar.domain import Evidence, EvidenceType
        repo.add_evidence(idea.id, [Evidence(claim="Oude ruis uit een blog", type=EvidenceType.ESTIMATE,
                                             topic="concurrenten", reasoning="oud")])
        capsys.readouterr()
        assert main(["heranalyseer", "--map", str(tmp_path / "r")]) == 0
        out = capsys.readouterr().out
        assert "Gewist:" in out and "Rangschikking" in out
        topics = [e.topic for e in repo.evidence_for(idea.id)]
        assert "concurrenten" not in topics and any(t.startswith("screening:") for t in topics)
        assert repo.get_idea(idea.id).status is IdeaStatus.ANALYSED
    finally:
        get_settings.cache_clear()



def test_pdf_export_one_file_per_analysed_idea(tmp_path):
    import pytest
    pytest.importorskip("weasyprint")
    from kansenradar.reporting.pdf import write_pdfs
    repo = Repository(make_engine("sqlite:///:memory:"))
    idea, ev = analysed(title="Google-profiel onderhoud voor kappers")
    other, _ = build(title="Niet geanalyseerd")
    repo.save_idea(idea)
    repo.add_evidence(idea.id, ev)
    repo.save_idea(other)
    paths = write_pdfs(repo, tmp_path)
    assert len(paths) == 1 and paths[0].read_bytes().startswith(b"%PDF")
    assert "google-profiel-onderhoud-voor-kappers" in paths[0].name
    assert paths[0].stat().st_size > 5_000
