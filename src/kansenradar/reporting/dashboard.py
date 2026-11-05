"""Static dashboard (index.html): funnel, ranking with filters/sorting, research queue, runs.
Self-contained and offline; small inline script for filtering and sorting."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

from kansenradar.domain import Conclusion, Idea, IdeaStatus
from kansenradar.reporting.html import CONCLUSION_CLASS, e, page
from kansenradar.scoring import rank

S = IdeaStatus
FUNNEL = [
    ("Gegenereerd", None),
    ("Generiek afgewezen", {S.REJECTED_GENERIC}),
    ("Duplicaat", {S.DUPLICATE}),
    ("Afgewezen bij screening", {S.SCREENED_OUT}),
    ("Wacht op onderzoek", {S.SCREENED_PASS, S.SCREENED_DOUBT, S.GENERATED}),
    ("Onderzocht", {S.RESEARCHED}),
    ("Volledig geanalyseerd", {S.ANALYSED}),
]

SCRIPT = """
const q=document.getElementById('q'),fc=document.getElementById('fc'),fs=document.getElementById('fs');
function apply(){const t=q.value.toLowerCase();document.querySelectorAll('#rank tbody tr').forEach(r=>{
const ok=(!t||r.dataset.text.includes(t))&&(!fc.value||r.dataset.c===fc.value)&&(!fs.value||r.dataset.s===fs.value);
r.style.display=ok?'':'none';});}
[q,fc,fs].forEach(x=>x&&x.addEventListener('input',apply));
document.querySelectorAll('#rank th[data-k]').forEach((th,i)=>th.addEventListener('click',()=>{
const tb=document.querySelector('#rank tbody'),rows=[...tb.rows],k=th.dataset.k,dir=th.dataset.dir==='a'?'d':'a';
th.dataset.dir=dir;rows.sort((a,b)=>{const x=a.dataset[k],y=b.dataset[k];const n=parseFloat(x)-parseFloat(y);
return (isNaN(n)?x.localeCompare(y):n)*(dir==='a'?1:-1);});rows.forEach(r=>tb.appendChild(r));}));
"""

EXTRA_CSS = """
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.tile b{display:block;font-size:1.5rem}.controls{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}
input,select{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
th[data-k]{cursor:pointer}th[data-k]:after{content:' ↕';color:var(--muted)}
"""


def render_dashboard(ideas: list[Idea], runs: list[dict]) -> str:
    status_counts = Counter(i.status for i in ideas)
    tiles = []
    for label, statuses in FUNNEL:
        n = len(ideas) if statuses is None else sum(status_counts[s] for s in statuses)
        tiles.append(f'<div class="tile"><span class="muted small">{e(label)}</span><b>{n}</b></div>')

    ranked = rank(ideas)
    conc_counts = Counter(i.scorecard.conclusion for i in ranked)
    conc_tiles = "".join(f'<div class="tile"><span class="small {CONCLUSION_CLASS[c]}">{e(c.value)}</span>'
                         f"<b>{conc_counts[c]}</b></div>" for c in Conclusion)
    rows = []
    for n, i in enumerate(ranked, 1):
        sc = i.scorecard
        text = f"{i.title} {i.target_customer} {i.problem}".lower()
        rows.append(
            f'<tr data-text="{e(text)}" data-c="{e(sc.conclusion.value)}" data-s="{e(i.sector.value)}" '
            f'data-n="{n}" data-t="{sc.total}" data-b="{sc.confidence}" data-o="{i.originality.score if i.originality else 0}">'
            f"<td class='num'>{n}</td><td><a href='ideeen/{e(i.id)}.html'>{e(i.title)}</a><br>"
            f"<span class='muted small'>{e(i.target_customer)}</span></td>"
            f"<td class='{CONCLUSION_CLASS[sc.conclusion]}'><strong>{e(sc.conclusion.value)}</strong></td>"
            f"<td class='num'>{sc.total:.0f}</td><td class='num'>{sc.confidence:.0f}%</td>"
            f"<td class='num'>{i.originality.score if i.originality else '—'}</td><td class='small'>{e(i.sector.value)}</td></tr>")
    sectors = sorted({i.sector.value for i in ranked})
    ranking = (
        '<div class="controls"><input id="q" placeholder="Zoeken…" aria-label="Zoeken">'
        '<select id="fc" aria-label="Conclusie"><option value="">Alle conclusies</option>'
        + "".join(f"<option>{e(c.value)}</option>" for c in Conclusion)
        + '</select><select id="fs" aria-label="Sector"><option value="">Alle sectoren</option>'
        + "".join(f"<option>{e(s)}</option>" for s in sectors) + "</select></div>"
        '<div class="wrap"><table id="rank"><thead><tr><th data-k="n" class="num">#</th><th>Idee</th>'
        '<th>Conclusie</th><th data-k="t" class="num">Score</th><th data-k="b" class="num">Betrouwb.</th>'
        '<th data-k="o" class="num">Orig.</th><th>Sector</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table></div>"
    ) if rows else '<p class="muted">Nog geen volledig geanalyseerde ideeën.</p>'

    queue = sorted([i for i in ideas if i.status in (S.SCREENED_PASS, S.SCREENED_DOUBT)],
                   key=lambda i: i.research_priority or 0, reverse=True)[:15]
    queue_html = ("<div class='wrap'><table><tr><th class='num'>Prioriteit</th><th>Idee</th><th>Screening</th></tr>"
                  + "".join(f"<tr><td class='num'>{i.research_priority:.2f}</td><td>{e(i.title)}<br>"
                            f"<span class='muted small'>{e(i.id)}</span></td><td>{e(i.screening.decision.value)} "
                            f"({i.screening.weighted_score:.2f}/5)</td></tr>" for i in queue)
                  + "</table></div>") if queue else '<p class="muted">Geen ideeën in de wachtrij.</p>'

    runs_html = ("<div class='wrap'><table><tr><th>Run</th><th>Gestart</th><th>Status</th><th>Modus</th>"
                 "<th class='num'>Kosten</th></tr>"
                 + "".join(f"<tr><td class='small'>{e(r['id'])}</td><td>{e(r['started'])}</td><td>{e(r['status'])}</td>"
                           f"<td>{e(r.get('mode', ''))}</td><td class='num'>$ {r['spent']:.2f}</td></tr>" for r in runs[:20])
                 + "</table></div>") if runs else '<p class="muted">Nog geen runs.</p>'

    body = (f"<h1>Kansenradar</h1><p class='muted'>Businessopportuniteiten België en Nederland · "
            f"bijgewerkt {datetime.now(UTC):%d-%m-%Y %H:%M} UTC</p>"
            f"<section><h2>Trechter</h2><div class='grid'>{''.join(tiles)}</div></section>"
            f"<section><h2>Conclusies</h2><div class='grid'>{conc_tiles}</div></section>"
            f"<section><h2>Rangschikking</h2>{ranking}</section>"
            f"<section><h2>Wachtrij voor marktonderzoek</h2>{queue_html}</section>"
            f"<section><h2>Runs</h2>{runs_html}</section>"
            "<p class='muted small'>Score = commerciële score (100). Betrouwbaarheid = onderbouwing met "
            "geverifieerde bronnen. Originaliteit telt niet mee in de score. Geen beleggingsadvies.</p>"
            f"<script>{SCRIPT}</script>")
    html = page("Kansenradar", body)
    return html.replace("</style>", EXTRA_CSS + "</style>", 1)
