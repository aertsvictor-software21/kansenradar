"""Text embeddings for duplicate detection.

Two implementations, both free and local:

- FastEmbedEmbedder: semantic multilingual model (ONNX via fastembed, ~220 MB,
  CPU only, no PyTorch). Recognises paraphrases ("boekhouding" ~ "administratie").
  Downloads the model once from HuggingFace on first use.
- HashingEmbedder: lexical fallback with zero downloads and zero dependencies
  beyond numpy. Character n-grams make it robust to Dutch compounds and
  inflection ("werfdagboek" ~ "werfdagboeken"), but it does not understand synonyms.

Each embedder carries its own similarity thresholds, because the scales differ.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Thresholds:
    duplicate: float  # overall similarity >= this: same idea
    similar: float    # >= this (and < duplicate): grey zone, needs a judge
    # Per-field minimums used by the heuristic judge in the grey zone.
    field_target: float
    field_problem: float
    field_solution: float


class Embedder(Protocol):
    name: str
    thresholds: Thresholds

    def embed(self, texts: list[str]) -> list[list[float]]: ...


# Short list of Dutch function words that carry no meaning for idea similarity.
_STOPWORDS = frozenset(
    """de het een en of van voor in op aan met door bij naar om te tot uit als dat die dit
    deze is zijn wordt worden kan kunnen hun hen ze zij wij we je jij u hij er niet geen ook
    nog wel maar dan zo per via over onder tussen alle elke meer minder zeer veel weinig
    probleem doelgroep oplossing verdienmodel""".split()
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return [t for t in normalize(text).split() if t not in _STOPWORDS and len(t) > 1]


class HashingEmbedder:
    """Lexical embedding: hashed word unigrams + character 3..5-grams, L2-normalised."""

    # Calibrated on tests/fixtures/dedup_pairs.json (see history/calibrate.py):
    # unrelated pairs <= 0.14, duplicates 0.46-0.87, related ideas 0.21-0.62.
    # Lexical similarity cannot separate duplicates from related ideas on its own,
    # so the wide grey zone is resolved by the judge.
    thresholds = Thresholds(
        duplicate=0.80, similar=0.18, field_target=0.55, field_problem=0.45, field_solution=0.45
    )

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.name = f"hashing-v1-{dim}"

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        value = int.from_bytes(h, "little")
        sign = 1.0 if value & 1 else -1.0
        return (value >> 1) % self.dim, sign

    def _features(self, text: str) -> dict[str, float]:
        feats: dict[str, float] = {}
        for tok in tokens(text):
            feats["w:" + tok] = feats.get("w:" + tok, 0.0) + 1.0
            padded = f"<{tok}>"
            for n in (3, 4, 5):
                for i in range(len(padded) - n + 1):
                    key = f"c{n}:" + padded[i : i + n]
                    feats[key] = feats.get(key, 0.0) + 0.5
        return feats

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float64)
            for feat, weight in self._features(text).items():
                idx, sign = self._bucket(feat)
                vec[idx] += sign * np.log1p(weight)  # sublinear term frequency
            norm = np.linalg.norm(vec)
            out.append((vec / norm if norm else vec).round(6).tolist())
        return out


class FastEmbedEmbedder:
    """Semantic multilingual embeddings via fastembed (local ONNX inference)."""

    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # PROVISIONAL values for this model (not yet calibrated: the model could not be
    # downloaded in the build environment). Run `python -m kansenradar.history.calibrate`
    # on your own machine and adjust.
    thresholds = Thresholds(
        duplicate=0.88, similar=0.55, field_target=0.75, field_problem=0.70, field_solution=0.70
    )

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: str | None = None) -> None:
        from fastembed import TextEmbedding  # optional dependency

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.name = f"fastembed:{model_name}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [np.asarray(v, dtype=np.float64).round(6).tolist() for v in self._model.embed(texts)]


def get_embedder(kind: str = "auto", **kwargs) -> Embedder:
    """kind: 'auto' (semantic if available, else lexical), 'fastembed' or 'hashing'."""
    if kind == "hashing":
        return HashingEmbedder()
    if kind not in ("auto", "fastembed"):
        raise ValueError(f"Onbekende embedder '{kind}'.")
    try:
        return FastEmbedEmbedder(**kwargs)
    except Exception as exc:  # missing package, no network for first download, ...
        if kind == "fastembed":
            raise
        log.warning(
            "Semantisch embeddingmodel niet beschikbaar (%s); val terug op lexicale embedder.", exc
        )
        return HashingEmbedder()
