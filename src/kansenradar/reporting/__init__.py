"""Write the dashboard and per-idea reports to a folder."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from kansenradar.domain import IdeaStatus
from kansenradar.reporting.dashboard import render_dashboard
from kansenradar.reporting.html import render_idea_report
from kansenradar.storage import Repository
from kansenradar.storage.models import RunRow


def _runs(repo: Repository) -> list[dict]:
    with repo.session() as s:
        rows = s.scalars(select(RunRow).order_by(RunRow.started_at.desc())).all()
        return [dict(id=r.id, started=f"{r.started_at:%d-%m-%Y %H:%M}", status=r.status,
                     mode=(r.params or {}).get("mode", ""), spent=r.spent_usd or 0.0) for r in rows]


def write_reports(repo: Repository, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    (out / "ideeen").mkdir(parents=True, exist_ok=True)
    ideas = repo.list_ideas()
    for idea in ideas:
        if idea.status is IdeaStatus.ANALYSED:
            (out / "ideeen" / f"{idea.id}.html").write_text(
                render_idea_report(idea, repo.evidence_for(idea.id)), encoding="utf-8")
    index = out / "index.html"
    index.write_text(render_dashboard(ideas, _runs(repo)), encoding="utf-8")
    return index


__all__ = ["render_dashboard", "render_idea_report", "write_reports"]
