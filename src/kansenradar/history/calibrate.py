"""Calibrate dedup thresholds for an embedder on labelled Dutch idea pairs.

Usage (on your own machine):
    python -m kansenradar.history.calibrate                 # auto: semantic model if available
    python -m kansenradar.history.calibrate --embedder hashing
    python -m kansenradar.history.calibrate --pairs mijn_paren.json

Prints overall and per-field similarity per label, then the end-to-end
result of the dedup decision with the embedder's current thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kansenradar.domain import IdeaCandidate, OriginalityStrategy, Sector
from kansenradar.history.dedup import DedupVerdict, DuplicateChecker
from kansenradar.history.embeddings import get_embedder
from kansenradar.storage import Repository, cosine, make_engine

DEFAULT_PAIRS = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "dedup_pairs.json"
EXPECTED = {"duplicate": DedupVerdict.DUPLICATE, "similar": DedupVerdict.RELATED,
            "different": DedupVerdict.UNIQUE}


def to_candidate(d: dict, title: str) -> IdeaCandidate:
    return IdeaCandidate(
        title=title, one_liner=d["one_liner"], problem=d["problem"],
        target_customer=d["target_customer"], business_model=d["business_model"],
        sector=Sector.SAAS, strategy=OriginalityStrategy.PROBLEM_DRIVEN,
    )


def evaluate(pairs: list[dict], embedder) -> list[dict]:
    rows = []
    for i, p in enumerate(pairs):
        a, b = to_candidate(p["a"], f"Idee A{i}"), to_candidate(p["b"], f"Idee B{i}")
        va, vb = embedder.embed([a.fingerprint_text(), b.fingerprint_text()])
        repo = Repository(make_engine("sqlite:///:memory:"))
        checker = DuplicateChecker(repo, embedder)
        checker.register(a)
        _, check = checker.register(b)
        rows.append(dict(label=p["label"], note=p.get("note", ""), overall=cosine(va, vb),
                         verdict=check.verdict, expected=EXPECTED[p["label"]]))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--embedder", default="auto", choices=["auto", "fastembed", "hashing"])
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    args = ap.parse_args(argv)

    embedder = get_embedder(args.embedder)
    pairs = json.loads(args.pairs.read_text(encoding="utf-8"))
    rows = evaluate(pairs, embedder)
    t = embedder.thresholds
    print(f"Embedder: {embedder.name}")
    print(f"Drempels: duplicaat >= {t.duplicate}, grijze zone >= {t.similar}\n")
    for label in EXPECTED:
        sims = [r["overall"] for r in rows if r["label"] == label]
        if sims:
            print(f"{label:10s} gelijkenis min {min(sims):.2f}  max {max(sims):.2f}  (n={len(sims)})")
    correct = sum(r["verdict"] == r["expected"] for r in rows)
    print(f"\nBeslissingen correct: {correct}/{len(rows)}")
    for r in rows:
        mark = "OK " if r["verdict"] == r["expected"] else "FOUT"
        print(f"  {mark} {r['label']:9s} {r['overall']:.2f} -> {r['verdict'].value:9s} {r['note']}")
    lost = [r for r in rows if r["verdict"] is DedupVerdict.DUPLICATE and r["label"] != "duplicate"]
    if lost:
        print(f"\nWAARSCHUWING: {len(lost)} niet-duplicaten onterecht als duplicaat verworpen. Verhoog de drempels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
