from datetime import UTC, datetime, timedelta

import pytest

from kansenradar.domain import Evidence, EvidenceType

NOW = datetime(2026, 11, 1, tzinfo=UTC)


def verified_fact(**overrides):
    data = dict(
        claim="België telt meer dan 1,1 miljoen btw-plichtige ondernemingen.",
        type=EvidenceType.FACT,
        topic="marktgrootte",
        source_url="https://statbel.fgov.be/voorbeeld",
        quote="1.1 miljoen btw-plichtige ondernemingen",
        verified=True,
        retrieved_at=NOW,
    )
    data.update(overrides)
    return Evidence(**data)


def test_verified_fact_is_accepted():
    assert verified_fact().type is EvidenceType.FACT


@pytest.mark.parametrize("missing", ["source_url", "quote", "retrieved_at"])
def test_fact_without_requirement_is_rejected(missing):
    with pytest.raises(ValueError):
        verified_fact(**{missing: None})


def test_unverified_fact_is_rejected_by_constructor():
    with pytest.raises(ValueError):
        verified_fact(verified=False)


def test_create_downgrades_fact_without_source_to_hypothesis():
    ev = Evidence.create(claim="Kmo's willen hiervoor betalen.", type=EvidenceType.FACT, topic="prijs")
    assert ev.type is EvidenceType.HYPOTHESIS
    assert "bron-URL" in ev.downgrade_reason


def test_create_downgrades_unverified_sourced_fact_to_estimate():
    ev = Evidence.create(
        claim="Concurrent X vraagt 49 euro per maand.", type=EvidenceType.FACT, topic="prijs",
        source_url="https://example.com/pricing", quote="€49 / maand",
    )
    assert ev.type is EvidenceType.ESTIMATE
    assert "verificatie" in ev.downgrade_reason


def test_estimate_requires_reasoning_or_source():
    with pytest.raises(ValueError):
        Evidence(claim="Ongeveer 5000 klanten.", type=EvidenceType.ESTIMATE, topic="markt")
    ok = Evidence(
        claim="Ongeveer 5000 klanten.", type=EvidenceType.ESTIMATE, topic="markt",
        reasoning="50.000 bedrijven x 10% met dit probleem.",
    )
    assert ok.type is EvidenceType.ESTIMATE


def test_failed_verification_downgrades_fact():
    ev = verified_fact().mark_verified(quote_found=False, checked_at=NOW)
    assert ev.type is EvidenceType.ESTIMATE
    assert not ev.verified
    assert "niet teruggevonden" in ev.downgrade_reason


def test_outdated_fact_counts_as_estimate():
    old = verified_fact(published_at=NOW - timedelta(days=800))
    assert old.effective_type(max_age_days=365, now=NOW) is EvidenceType.ESTIMATE
    fresh = verified_fact(published_at=NOW - timedelta(days=30))
    assert fresh.effective_type(max_age_days=365, now=NOW) is EvidenceType.FACT
