"""Price extraction from page text (Dutch/English notations), normalised to per month."""

from __future__ import annotations

import re

from kansenradar.domain import PricePoint

_AMOUNT = r"(\d{1,5}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)"
_PRICE = re.compile(
    rf"(?:€|eur\b|euro\b)\s*{_AMOUNT}|{_AMOUNT}\s*(?:€|eur\b|euro\b)", re.IGNORECASE)
_PERIODS = (
    ("maand", re.compile(r"(per|/|p\.?)\s*(maand|mnd|month|mo)\b|maandelijks|monthly", re.I)),
    ("jaar", re.compile(r"(per|/|p\.?)\s*(jaar|jr|year|yr)\b|jaarlijks|yearly|annually", re.I)),
    ("eenmalig", re.compile(r"eenmalig|one[- ]time|levenslang|lifetime", re.I)),
)


def _to_float(raw: str) -> float:
    """'1.250' -> 1250, '49,95' -> 49.95, '1.250,50' -> 1250.5 (last separator + 1-2 digits = decimals)."""
    raw = raw.strip()
    m = re.match(r"^(.*?)[.,](\d{1,2})$", raw)
    if m:
        return float(re.sub(r"[.,]", "", m.group(1)) + "." + m.group(2))
    return float(re.sub(r"[.,]", "", raw))


def extract_prices(text: str, url: str, max_points: int = 6) -> list[PricePoint]:
    points: list[PricePoint] = []
    for m in _PRICE.finditer(text):
        amount = _to_float(m.group(1) or m.group(2))
        if not 1 <= amount <= 50_000:
            continue
        start, end = max(0, m.start() - 60), min(len(text), m.end() + 60)
        context = text[start:end]
        after = text[m.end(): m.end() + 40]
        period = next((name for name, rx in _PERIODS if rx.search(after)), None) \
            or next((name for name, rx in _PERIODS if rx.search(context)), "onbekend")
        monthly = amount if period == "maand" else round(amount / 12, 2) if period == "jaar" else None
        quote = re.sub(r"\s+", " ", context).strip()
        points.append(PricePoint(amount_eur=amount, period=period, monthly_eur=monthly,
                                 quote=quote, url=url))
        if len(points) >= max_points:
            break
    return points
