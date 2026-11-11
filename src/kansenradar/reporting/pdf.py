"""PDF export of the per-idea reports (A4), via WeasyPrint (free, optional: pip install -e ".[pdf]")."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from kansenradar.domain import IdeaStatus
from kansenradar.reporting.html import render_idea_report

PRINT_CSS = """
@page { size: A4; margin: 16mm 14mm 18mm;
        @bottom-center { content: "Kansenradar · pagina " counter(page) " van " counter(pages);
                         font-size: 8pt; color: #777; } }
:root { --bg:#fff; --card:#fff; --fg:#1d1d1b; --muted:#5f5f5a; --line:#ddd; --accent:#2f5d8a;
        --fact:#1f7a4d; --est:#9a6a00; --hyp:#8a3b3b; --good:#1f7a4d; --mid:#9a6a00; --bad:#a33a3a; }
body { background:#fff; font-size: 10pt; }
main { max-width: none; padding: 0; }
section { padding: 10px 12px; margin: 8px 0; }
h1 { font-size: 18pt; } h2 { font-size: 12pt; }
table { font-size: 8.5pt; }
a { color: var(--accent); text-decoration: none; }
p.muted.small:first-child { display: none; }  /* 'back to overview' link is useless on paper */
"""


class PdfUnavailable(RuntimeError):
    pass


def slug(text: str, max_len: int = 60) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return text[:max_len].rstrip("-") or "idee"


def html_to_pdf(html: str, path: Path) -> None:
    try:
        from weasyprint import CSS, HTML
    except (ImportError, OSError) as exc:  # package or system libraries missing
        raise PdfUnavailable(str(exc)) from exc
    HTML(string=html).write_pdf(path, stylesheets=[CSS(string=PRINT_CSS)])


def write_pdfs(repo, out_dir: str | Path, idea_ids: list[str] | None = None) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if idea_ids:
        ideas = [i for i in (repo.get_idea(x) for x in idea_ids) if i is not None]
    else:
        ideas = [i for i in repo.list_ideas() if i.status is IdeaStatus.ANALYSED]
    written = []
    for idea in ideas:
        conclusion = slug(idea.scorecard.conclusion.value, 25) if idea.scorecard else "niet-gescoord"
        path = out / f"{conclusion}_{slug(idea.title)}_{idea.id[-6:]}.pdf"
        html_to_pdf(render_idea_report(idea, repo.evidence_for(idea.id)), path)
        written.append(path)
    return written
