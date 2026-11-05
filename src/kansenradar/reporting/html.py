"""HTML report per idea (14 sections + sources). Self-contained: no external files,
works offline. All dynamic text is escaped. Print to PDF from the browser."""

from __future__ import annotations

from datetime import UTC, datetime
from html import escape

from kansenradar.domain import Conclusion, Evidence, EvidenceType, Idea, ResearchTopic
from kansenradar.reporting.strategy import acquisition_strategy, pricing_strategy

CSS = """
:root{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1b;--muted:#5f5f5a;--line:#e3e2dd;--accent:#2f5d8a;
--fact:#1f7a4d;--est:#9a6a00;--hyp:#8a3b3b;--good:#1f7a4d;--mid:#9a6a00;--bad:#a33a3a}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161615;--card:#1f1f1d;--fg:#ecebe6;
--muted:#a6a59f;--line:#34332f;--accent:#7fb0e0;--fact:#5cc58f;--est:#e0b450;--hyp:#e08a8a;--good:#5cc58f;
--mid:#e0b450;--bad:#e08a8a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:960px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.7rem;margin:.2em 0}h2{font-size:1.15rem;margin:0 0 .6em;color:var(--accent)}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin:14px 0}
.muted{color:var(--muted)}table{width:100%;border-collapse:collapse;font-size:.92rem}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.badge{display:inline-block;font-size:.72rem;font-weight:700;letter-spacing:.03em;padding:1px 7px;
border-radius:999px;border:1px solid currentColor;white-space:nowrap}
.FEIT{color:var(--fact)}.SCHATTING{color:var(--est)}.HYPOTHESE{color:var(--hyp)}
.verdict{font-size:1.25rem;font-weight:700}.c0,.c1{color:var(--good)}.c2,.c3{color:var(--mid)}.c4{color:var(--bad)}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden;min-width:80px}
.bar>span{display:block;height:100%;background:var(--accent)}ul{margin:.3em 0;padding-left:1.2em}
a{color:var(--accent)}.wrap{overflow-x:auto}.small{font-size:.85rem}
@media print{body{background:#fff}section{break-inside:avoid;border-color:#ccc}}
"""

CONCLUSION_CLASS = {Conclusion.VERY_INTERESTING: "c0", Conclusion.PROMISING: "c1", Conclusion.EXPERIMENT: "c2",
                    Conclusion.MORE_RESEARCH: "c3", Conclusion.REJECT: "c4"}


def e(x) -> str:
    return escape("" if x is None else str(x))


def eur(x) -> str:
    return "—" if x is None else "€ " + f"{x:,.0f}".replace(",", ".")


def num(x) -> str:
    return f"{x:,.0f}".replace(",", ".")


def badge(kind: EvidenceType | str) -> str:
    v = kind.value if isinstance(kind, EvidenceType) else kind
    return f'<span class="badge {e(v)}">{e(v)}</span>'


def page(title: str, body: str) -> str:
    return (f'<!doctype html><html lang="nl"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{e(title)}</title><style>{CSS}</style></head><body><main>{body}</main></body></html>")


def _section(n: int, title: str, inner: str) -> str:
    return f"<section><h2>{n}. {e(title)}</h2>{inner}</section>"


def _ul(items) -> str:
    items = [i for i in items if i]
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>" if items else '<p class="muted">—</p>'


def _evidence_items(evidence: list[Evidence], topics: tuple[ResearchTopic, ...], limit: int = 6) -> str:
    values = {t.value for t in topics}
    rows = [ev for ev in evidence if ev.topic in values]
    rows.sort(key=lambda ev: {EvidenceType.FACT: 0, EvidenceType.ESTIMATE: 1}.get(ev.type, 2))
    items = []
    for ev in rows[:limit]:
        src = f' — <a href="{e(ev.source_url)}">{e(ev.source_title or ev.source_url)}</a>' if ev.source_url else ""
        items.append(f"{badge(ev.type)} {e(ev.claim)}{src}")
    return _ul(items) if items else '<p class="muted">Geen onderbouwing gevonden; dit onderwerp blijft open.</p>'


def render_idea_report(idea: Idea, evidence: list[Evidence]) -> str:
    sc, fin, comp, rev, res = idea.scorecard, idea.finance, idea.competition, idea.review, idea.research
    T = ResearchTopic
    parts = [f'<p class="muted small"><a href="../index.html">← Overzicht</a> · {e(idea.id)} · '
             f"{e(idea.sector.value)} · strategie: {e(idea.strategy.value)} · herkomst: {e(idea.generated_by)}</p>"]

    # 1. Name
    verdict = (f'<p class="verdict {CONCLUSION_CLASS[sc.conclusion]}">{e(sc.conclusion.value)}</p>'
               f"<p>{sc.total:.0f}/100 · betrouwbaarheid {sc.confidence:.0f}%</p>") if sc else ""
    parts.append(f"<section><h2>1. Naam van het concept</h2><h1>{e(idea.title)}</h1>{verdict}</section>")
    parts.append(_section(2, "Korte beschrijving", f"<p>{e(idea.one_liner)}</p>"
                          + (f'<p class="muted">Waarom nu: {e(idea.why_now)}</p>' if idea.why_now else "")))
    parts.append(_section(3, "Het opgeloste probleem", f"<p>{e(idea.problem)}</p>"
                          + _evidence_items(evidence, (T.PROBLEM, T.DEMAND, T.CURRENT_SOLUTION))))
    parts.append(_section(4, "Doelgroep", f"<p>{e(idea.target_customer)}</p>"
                          + _evidence_items(evidence, (T.CUSTOMERS,))))

    # 5. Market potential
    inner = _evidence_items(evidence, (T.MARKET_SIZE, T.CUSTOMERS, T.TRENDS), limit=8)
    if fin:
        m, p = fin.assumption("market"), fin.assumption("penetration")
        inner += (f"<p>Werkhypothese: {num(m.low)} – {num(m.high)} potentiële klanten {badge(m.basis)}, "
                  f"maximaal {p.low:g}–{p.high:g}% bereikbaar binnen 3 jaar {badge(p.basis)}.</p>")
    if res:
        inner += (f'<p class="muted small">Onderzoek: {e(res.mode)} · {res.topics_with_facts}/10 onderwerpen met '
                  f"geverifieerd feit · {res.pages_fetched} pagina's · {res.quotes_rejected} verworpen citaten.</p>")
    parts.append(_section(5, "Marktpotentieel", inner))

    # 6. Competitors
    if comp:
        rows = "".join(
            f"<tr><td>{e(c.name)}<br><a class='small' href='{e(c.url)}'>{e(c.domain)}</a>"
            f"{' ' + badge('HYPOTHESE') if c.source == 'modelkennis' else ''}</td><td>{e(c.kind)}</td>"
            f"<td>{e(c.description)}</td><td class='num'>"
            + (", ".join(f"€ {pp.amount_eur:g}/{pp.period}" for pp in c.prices[:3]) or "—") + "</td></tr>"
            for c in comp.competitors)
        inner = (f"<p>Verzadiging: <strong>{e(comp.saturation)}</strong> — {e(comp.saturation_reason)}</p>"
                 + (f'<div class="wrap"><table><tr><th>Concurrent</th><th>Type</th><th>Aanbod</th>'
                    f'<th class="num">Prijzen</th></tr>{rows}</table></div>' if rows else
                    '<p class="muted">Geen concurrenten gevonden of onderzocht.</p>')
                 + (("<p>Mogelijke gaten in de markt " + badge("HYPOTHESE") + ":</p>" + _ul(e(g) for g in comp.gaps))
                    if comp.gaps else ""))
    else:
        inner = '<p class="muted">Concurrentieanalyse niet uitgevoerd.</p>'
    parts.append(_section(6, "Concurrenten", inner))

    parts.append(_section(7, "Verdienmodel", f"<p>{e(idea.business_model)}</p>"
                          + (f'<p class="muted">Type: {e(fin.revenue_model)}</p>' if fin else "")))
    parts.append(_section(8, "Prijsstrategie", _ul(e(x) for x in pricing_strategy(idea))))

    # 9. Financial forecast
    if fin:
        sc_names = ("conservatief", "realistisch", "optimistisch")
        rows_spec = [
            ("Startkapitaal", lambda r: eur(r.startup_capital)), ("Maandelijkse vaste kosten", lambda r: eur(r.monthly_fixed_costs)),
            ("Marketing per maand", lambda r: eur(r.marketing_per_month)), ("Klantenwervingskosten (CAC)", lambda r: eur(r.cac)),
            ("Klanten na 12 / 36 maanden", lambda r: f"{r.customers_m12} / {r.customers_m36}"),
            ("Omzet jaar 1", lambda r: eur(r.revenue_year1)), ("Omzet jaar 3", lambda r: eur(r.revenue_year3)),
            ("Brutomarge", lambda r: f"{r.gross_margin_pct:.0f}%"), ("Nettoresultaat jaar 3*", lambda r: eur(r.net_year3)),
            ("Break-even (maand)", lambda r: str(r.operational_break_even_month or "> 36")),
            ("Terugverdiend (maand)", lambda r: str(r.payback_month or "> 36")),
            ("Benodigd kapitaal", lambda r: eur(r.capital_needed)), ("LTV / CAC", lambda r: f"{r.ltv_cac:.1f}"),
        ]
        table = ("<tr><th></th>" + "".join(f'<th class="num">{n}</th>' for n in sc_names) + "</tr>"
                 + "".join(f"<tr><td>{e(label)}</td>" + "".join(f'<td class="num">{e(fn(fin.scenarios[n]))}</td>'
                                                             for n in sc_names) + "</tr>" for label, fn in rows_spec))
        assumptions = "".join(
            f"<tr><td>{e(a.label)}</td><td class='num'>{a.low:g} / {a.mid:g} / {a.high:g} {e(a.unit)}</td>"
            f"<td>{badge(a.basis)}</td><td class='small'>{e(a.source)}</td></tr>" for a in fin.assumptions)
        inner = (f'<div class="wrap"><table>{table}</table></div>'
                 f"<p>Schaalbaarheid: <strong>{e(fin.scalability)}</strong> — {e(fin.scalability_reason)}</p>"
                 f'<h3 class="small">Aannames (laag / midden / hoog)</h3><div class="wrap"><table>{assumptions}</table></div>'
                 + _ul(e(n) for n in fin.notes))
    else:
        inner = '<p class="muted">Financiële analyse niet uitgevoerd.</p>'
    parts.append(_section(9, "Financiële prognose", inner))

    parts.append(_section(10, "Klantenwervingsstrategie", _ul(e(x) for x in acquisition_strategy(idea))
                          + _evidence_items(evidence, (T.REACH,))))

    # 11. Competitive advantage
    inner = f"<p>{e(idea.differentiator) or '—'}</p>"
    if rev:
        inner += _ul(f"{e(c.question.value)} <strong>{e(c.verdict)}</strong> {badge(c.basis)} — {e(c.explanation)}"
                     for c in rev.checks)
    parts.append(_section(11, "Concurrentievoordeel", inner))

    # 12. Risks
    if rev:
        rows = "".join(f"<tr><td>{e(k.title)}{' <strong>(FATAAL)</strong>' if k.fatal else ''}</td>"
                       f"<td class='num'>{k.likelihood}×{k.impact}={k.severity}</td><td>{e(k.description)}</td>"
                       f"<td>{e(k.mitigation)}</td><td>{badge(k.basis)}</td></tr>" for k in rev.risks)
        inner = (f"<p>{e(rev.summary)}</p><div class='wrap'><table><tr><th>Risico</th><th class='num'>Kans×impact</th>"
                 f"<th>Toelichting</th><th>Beperking</th><th>Basis</th></tr>{rows}</table></div>")
    else:
        inner = '<p class="muted">Kritische beoordeling niet uitgevoerd.</p>'
    parts.append(_section(12, "Risico's", inner))

    # 13. Validation test
    if rev:
        v = rev.validation_test
        inner = (f"<p><strong>{e(v.method)}</strong> · {v.duration_weeks} weken · {e(v.cost_eur)}</p>"
                 f"<p>Te toetsen aanname: <em>{e(v.critical_assumption)}</em></p>"
                 f"<ol>{''.join(f'<li>{e(s)}</li>' for s in v.steps)}</ol>"
                 f"<p><strong>Geslaagd als</strong></p>{_ul(e(s) for s in v.success_criteria)}"
                 f"<p><strong>Stoppen als</strong></p>{_ul(e(s) for s in v.kill_criteria)}")
    else:
        inner = '<p class="muted">—</p>'
    parts.append(_section(13, "Eerste validatietest", inner))

    # 14. Final assessment
    if sc:
        rows = "".join(
            f"<tr><td>{e(s.criterion)}</td><td class='num'>{s.points:.1f} / {s.max_points}</td>"
            f"<td><div class='bar'><span style='width:{100 * s.points / s.max_points:.0f}%'></span></div></td>"
            f"<td class='small'>{e(s.rationale)}" + (f"<br><em>Plafond: {e(s.capped_by)}</em>" if s.capped_by else "")
            + "</td></tr>" for s in sc.subscores)
        inner = (f'<p class="verdict {CONCLUSION_CLASS[sc.conclusion]}">{e(sc.conclusion.value)}</p>'
                 f"<p>{e(sc.conclusion_reason)}</p><div class='wrap'><table>{rows}"
                 f"<tr><th>Totaal</th><th class='num'>{sc.total:.1f} / 100</th><th></th><th></th></tr></table></div>"
                 f"<p>Betrouwbaarheid: <strong>{sc.confidence:.0f}%</strong></p>{_ul(e(r) for r in sc.confidence_reasons)}")
    else:
        inner = '<p class="muted">Nog niet gescoord.</p>'
    parts.append(_section(14, "Eindbeoordeling", inner))

    # Sources
    sourced = [ev for ev in evidence if ev.source_url]
    rows = "".join(f"<tr><td>{badge(ev.type)}</td><td>{e(ev.topic)}</td><td>{e(ev.claim)}"
                   + (f"<br><span class='muted small'>“{e(ev.quote)}”</span>" if ev.quote else "")
                   + f"</td><td><a href='{e(ev.source_url)}'>{e(ev.source_title or ev.source_url)}</a>"
                   f"<br><span class='muted small'>{e(ev.source_quality.value)}</span></td></tr>" for ev in sourced)
    counts = {t: sum(ev.type is t for ev in evidence) for t in EvidenceType}
    parts.append("<section><h2>Bronnen en bewijs</h2>"
                 f"<p>{counts[EvidenceType.FACT]} feiten · {counts[EvidenceType.ESTIMATE]} schattingen · "
                 f"{counts[EvidenceType.HYPOTHESIS]} hypotheses.</p>"
                 + (f"<div class='wrap'><table>{rows}</table></div>" if rows else '<p class="muted">Geen externe bronnen.</p>')
                 + "</section>")
    parts.append(f'<p class="muted small">Gegenereerd door Kansenradar op {datetime.now(UTC):%d-%m-%Y}. '
                 "Dit rapport ondersteunt een beslissing en is geen beleggingsadvies of due diligence. "
                 "FEIT = geverifieerd in de bron; SCHATTING = afgeleid; HYPOTHESE = te toetsen aanname.</p>")
    return page(idea.title, "".join(parts))
