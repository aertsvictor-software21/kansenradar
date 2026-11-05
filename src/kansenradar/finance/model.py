"""Deterministic 36-month simulation per scenario. No LLM involved.

Per month t:
  ramp       = min(1, t / RAMP_MONTHS)                 (acquisition ramps up)
  capacity   = market * penetration - customers_prev   (market ceiling)
  new        = min(ramp * (marketing / CAC + organic), capacity)
  customers  = customers_prev * (1 - churn) + new
  revenue    = customers * price
  net        = revenue * margin - fixed - marketing
  cumulative = -startup + sum(net)
"""

from __future__ import annotations

from kansenradar.domain import Assumption, Evidence, FinancialModel, Idea, ScenarioResult
from kansenradar.finance.assumptions import build_assumptions

SCENARIOS = ("conservatief", "realistisch", "optimistisch")
RAMP_MONTHS = 6
HORIZON = 36


def simulate(values: dict[str, float], scenario: str, months: int = HORIZON) -> ScenarioResult:
    price, cac, churn = values["price"], values["cac"], values["churn"] / 100
    margin, fixed, marketing = values["margin"] / 100, values["fixed"], values["marketing"]
    ceiling = values["market"] * values["penetration"] / 100
    customers, cumulative = 0.0, -values["startup"]
    revenues, nets, counts, cumul = [], [], [], []
    cap_month = None
    for t in range(1, months + 1):
        ramp = min(1.0, t / RAMP_MONTHS)
        retained = customers * (1 - churn)
        capacity = max(0.0, ceiling - retained)
        wanted = ramp * (marketing / cac + values["organic"])
        new = min(wanted, capacity)
        if cap_month is None and wanted > capacity:
            cap_month = t
        customers = retained + new
        revenue = customers * price
        net = revenue * margin - fixed - marketing
        cumulative += net
        revenues.append(revenue)
        nets.append(net)
        counts.append(customers)
        cumul.append(cumulative)

    op_be = next((t + 1 for t in range(months) if all(n >= 0 for n in nets[t:])), None)
    payback = next((t + 1 for t, c in enumerate(cumul) if c >= 0), None)
    capital_needed = max(0.0, -min([-values["startup"]] + cumul))
    ltv = price * margin / max(churn, 0.01)
    unit_margin = price * margin
    return ScenarioResult(
        scenario=scenario, startup_capital=values["startup"], monthly_fixed_costs=fixed,
        marketing_per_month=marketing, cac=cac, price_per_month=price, gross_margin_pct=values["margin"],
        customers_m12=round(counts[11]), customers_m36=round(counts[-1]),
        revenue_m12=round(revenues[11], 2), revenue_m36=round(revenues[-1], 2),
        revenue_year1=round(sum(revenues[:12]), 2), revenue_year3=round(sum(revenues[24:36]), 2),
        net_month36=round(nets[-1], 2), net_year3=round(sum(nets[24:36]), 2),
        operational_break_even_month=op_be, payback_month=payback,
        capital_needed=round(capital_needed, 2), ltv=round(ltv, 2),
        ltv_cac=round(ltv / cac, 2) if cac else 0.0,
        cac_payback_months=round(cac / unit_margin, 1) if unit_margin > 0 and churn < 1 else None,
        market_cap_reached_month=cap_month,
    )


def scalability(model_results: dict[str, ScenarioResult], revenue_model: str) -> tuple[str, str]:
    r = model_results["realistisch"]
    if r.gross_margin_pct < 50:
        return "laag", f"Brutomarge van {r.gross_margin_pct:.0f}% laat weinig ruimte om te groeien."
    if r.market_cap_reached_month and r.market_cap_reached_month <= 24:
        return "laag", f"Het bereikbare marktplafond wordt al in maand {r.market_cap_reached_month} geraakt."
    if revenue_model == "eenmalig":
        return "gemiddeld", "Eenmalige verkopen: omzet groeit enkel mee met de wervingsinspanning."
    if r.gross_margin_pct >= 75 and r.ltv_cac >= 3:
        return "hoog", (f"Hoge brutomarge ({r.gross_margin_pct:.0f}%) en LTV/CAC {r.ltv_cac:.1f}: "
                        "extra klanten kosten weinig extra.")
    return "gemiddeld", f"Brutomarge {r.gross_margin_pct:.0f}%, LTV/CAC {r.ltv_cac:.1f}."


def build_financial_model(idea: Idea, evidence: list[Evidence],
                          marketing_per_month: float = 750) -> FinancialModel:
    rm, assumptions, notes = build_assumptions(idea, evidence, marketing_per_month)
    results = {}
    for sc in SCENARIOS:
        values = {a.key: a.for_scenario(sc) for a in assumptions}
        results[sc] = simulate(values, sc)
    label, reason = scalability(results, rm)
    from kansenradar.domain import EvidenceType
    core = [a for a in assumptions if a.key in ("price", "cac", "churn", "margin")]
    if all(a.basis is EvidenceType.HYPOTHESIS for a in core):
        reason += " Let op: gebaseerd op niet-gevalideerde vuistregels."
    if results["conservatief"].operational_break_even_month is None:
        notes.append("In het conservatieve scenario wordt binnen 36 maanden geen break-even gehaald.")
    return FinancialModel(revenue_model=rm, horizon_months=HORIZON, assumptions=assumptions,
                          scenarios=results, scalability=label, scalability_reason=reason, notes=notes)


def assumption_values(assumptions: list[Assumption], scenario: str) -> dict[str, float]:
    return {a.key: a.for_scenario(scenario) for a in assumptions}
