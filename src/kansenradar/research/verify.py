"""Quote verification: is the quote really in the source?

A quote passes when, after normalising case, accents and whitespace, it occurs in
the page text exactly, or near-exactly (similarity >= MIN_RATIO over a window of
the same length). Numbers are strict: every number in the quote must literally
appear in the matched passage, so a paraphrase that changes "12.000" into
"20.000" never passes.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

MIN_RATIO = 0.92
MIN_QUOTE_CHARS = 12
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def _numbers(text: str) -> set[str]:
    return {re.sub(r"[.,]", "", n) for n in _NUM.findall(text)}


@dataclass(frozen=True)
class QuoteCheck:
    found: bool
    ratio: float
    reason: str


def verify_quote(quote: str, page_text: str) -> QuoteCheck:
    q, page = _norm(quote), _norm(page_text)
    if len(q) < MIN_QUOTE_CHARS:
        return QuoteCheck(False, 0.0, "citaat te kort om te verifiëren")
    if q in page:
        return QuoteCheck(True, 1.0, "letterlijk gevonden")

    # Near-exact: compare against windows starting at occurrences of the quote's first word.
    first = q.split(" ")[0]
    best_ratio, best_window = 0.0, ""
    for m in re.finditer(re.escape(first), page):
        window = page[m.start(): m.start() + len(q) + 10]
        ratio = SequenceMatcher(None, q, window[: len(q)]).ratio()
        if ratio > best_ratio:
            best_ratio, best_window = ratio, window
    if best_ratio >= MIN_RATIO:
        missing = _numbers(q) - _numbers(best_window)
        if missing:
            return QuoteCheck(False, best_ratio, f"cijfers komen niet overeen ({', '.join(sorted(missing))})")
        return QuoteCheck(True, best_ratio, "nagenoeg letterlijk gevonden")
    return QuoteCheck(False, best_ratio, "niet gevonden in de bron")
