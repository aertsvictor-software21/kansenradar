import json
from pathlib import Path

import numpy as np
import pytest

from kansenradar.config import Settings
from kansenradar.domain import IdeaCandidate, IdeaStatus, OriginalityStrategy, Sector
from kansenradar.history import (
    DedupVerdict,
    DuplicateChecker,
    HashingEmbedder,
    HeuristicJudge,
    LLMJudge,
    get_embedder,
    reindex,
)
from kansenradar.history import embeddings as emb_module
from kansenradar.history.calibrate import EXPECTED, evaluate
from kansenradar.history.dedup import _LLMJudgement
from kansenradar.llm import FakeBackend, LLMClient, RunBudget
from kansenradar.storage import Repository, make_engine
from kansenradar.storage.models import IdeaRow

PAIRS = json.loads((Path(__file__).parent / "fixtures" / "dedup_pairs.json").read_text("utf-8"))


def cand(d: dict, title="Testidee") -> IdeaCandidate:
    return IdeaCandidate(title=title, sector=Sector.SAAS,
                         strategy=OriginalityStrategy.PROBLEM_DRIVEN, **d)


def pair(label: str, index: int = 0):
    p = [p for p in PAIRS if p["label"] == label][index]
    return cand(p["a"], "Idee A"), cand(p["b"], "Idee B")


@pytest.fixture
def repo():
    return Repository(make_engine("sqlite:///:memory:"))


@pytest.fixture
def embedder():
    return HashingEmbedder()


# --- embedder -----------------------------------------------------------------

def test_hashing_embedder_is_deterministic_and_normalised(embedder):
    a, b = embedder.embed(["Werfdagboek voor aannemers", "Werfdagboek voor aannemers"])
    assert a == b
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)


def test_hashing_embedder_handles_accents_and_inflection(embedder):
    from kansenradar.storage import cosine
    a, b, c = embedder.embed(["werfdagboeken voor aannemers", "werfdagboek voor aannémer",
                              "bijles wiskunde voor leerlingen"])
    assert cosine(a, b) > 0.6 > cosine(a, c)


def test_get_embedder_falls_back_without_semantic_model(monkeypatch):
    def broken(*a, **k):
        raise OSError("geen netwerk")
    monkeypatch.setattr(emb_module, "FastEmbedEmbedder", broken)
    assert isinstance(get_embedder("auto"), HashingEmbedder)
    with pytest.raises(OSError):
        get_embedder("fastembed")


def test_fastembed_wrapper_uses_model(monkeypatch):
    class FakeTextEmbedding:
        def __init__(self, model_name, cache_dir=None):
            self.model_name = model_name
        def embed(self, texts):
            return (np.ones(4) / 2 for _ in texts)
    fastembed = pytest.importorskip("fastembed")
    monkeypatch.setattr(fastembed, "TextEmbedding", FakeTextEmbedding)
    e = emb_module.FastEmbedEmbedder()
    assert e.name.startswith("fastembed:")
    assert e.embed(["a", "b"]) == [[0.5] * 4, [0.5] * 4]


# --- decisions on the labelled set ----------------------------------------------

def test_calibration_set_never_discards_non_duplicates(embedder):
    rows = evaluate(PAIRS, embedder)
    wrongly_discarded = [r for r in rows if r["verdict"] is DedupVerdict.DUPLICATE
                         and r["label"] != "duplicate"]
    assert wrongly_discarded == []
    unrelated_flagged = [r for r in rows if r["label"] == "different"
                         and r["verdict"] is not DedupVerdict.UNIQUE]
    assert unrelated_flagged == []
    accuracy = sum(r["verdict"] == r["expected"] for r in rows) / len(rows)
    assert accuracy >= 0.85


def test_exact_duplicate_is_detected(repo, embedder):
    checker = DuplicateChecker(repo, embedder)
    a, _ = pair("duplicate")
    checker.register(a)
    idea, check = checker.register(a.model_copy(update={"title": "Andere titel"}))
    assert check.verdict is DedupVerdict.DUPLICATE and check.judged_by == "exact"
    assert idea.status is IdeaStatus.DUPLICATE and idea.duplicate_of


def test_related_idea_is_kept_with_link(repo, embedder):
    checker = DuplicateChecker(repo, embedder)
    a, b = pair("similar", 1)  # same target customer, different problem
    first, _ = checker.register(a)
    idea, check = checker.register(b)
    assert check.verdict is DedupVerdict.RELATED
    assert idea.status is IdeaStatus.GENERATED
    assert idea.similar_to == [first.id]
    assert "Verschilt op" in idea.dedup_note


def test_batch_is_deduplicated_internally(repo, embedder):
    a, _ = pair("duplicate", 2)
    results = DuplicateChecker(repo, embedder).register_many([a, a.model_copy()])
    assert [c.verdict for _, c in results] == [DedupVerdict.UNIQUE, DedupVerdict.DUPLICATE]


def test_earlier_rejection_is_reported(repo, embedder):
    checker = DuplicateChecker(repo, embedder)
    a, b = pair("duplicate", 1)
    first, _ = checker.register(a)
    first.status = IdeaStatus.SCREENED_OUT
    repo.save_idea(first)
    with repo.session() as s, s.begin():
        s.get(IdeaRow, first.id).conclusion = "AFWIJZEN"
    idea, check = checker.register(b)
    assert check.verdict is DedupVerdict.DUPLICATE
    assert check.earlier_outcome == "eerdere conclusie: AFWIJZEN"
    assert "AFWIJZEN" in idea.dedup_note


def test_duplicates_are_not_used_as_reference(repo, embedder):
    checker = DuplicateChecker(repo, embedder)
    a, _ = pair("duplicate", 2)
    original, _ = checker.register(a)
    checker.register(a.model_copy())
    _, check = checker.register(a.model_copy())
    assert check.match_id == original.id


# --- judges ---------------------------------------------------------------------

def make_llm(handler):
    return LLMClient(FakeBackend({"dedup_judge": handler}), RunBudget(1.0), Settings())


def test_llm_judge_decides_grey_zone(repo, embedder):
    a, b = pair("duplicate", 0)  # synonyms: heuristic says related, LLM recognises it
    llm = make_llm(_LLMJudgement(same_idea=True, explanation="Zelfde doelgroep en probleem."))
    checker = DuplicateChecker(repo, embedder, LLMJudge(llm, HeuristicJudge(embedder)))
    checker.register(a)
    _, check = checker.register(b)
    assert check.verdict is DedupVerdict.DUPLICATE and check.judged_by == "llm"


def test_llm_judge_falls_back_to_heuristic_when_provider_fails(repo, embedder):
    def unavailable(_prompt):
        raise ConnectionError("provider offline")
    checker = DuplicateChecker(repo, embedder,
                               LLMJudge(make_llm(unavailable), HeuristicJudge(embedder)))
    a, b = pair("similar", 1)
    checker.register(a)
    _, check = checker.register(b)
    assert check.judged_by == "heuristiek" and check.verdict is DedupVerdict.RELATED


# --- reindex ----------------------------------------------------------------------

def test_reindex_after_switching_embedder(repo, embedder):
    a, b = pair("different")
    DuplicateChecker(repo, embedder).register_many([a, b])
    new = HashingEmbedder(dim=256)
    assert reindex(repo, new) == 2
    assert reindex(repo, new) == 0
    # Old embeddings are no longer compared with the new model.
    check, _ = DuplicateChecker(repo, new).check(a)
    assert check.verdict is DedupVerdict.DUPLICATE and check.judged_by == "exact"


def test_expected_mapping_covers_all_labels():
    assert {p["label"] for p in PAIRS} == set(EXPECTED)
