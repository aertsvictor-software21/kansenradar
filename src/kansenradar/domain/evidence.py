"""Evidence model: every claim the platform makes is an Evidence record.

Core rule (enforced in code, not left to the model):
    A claim may only be a FEIT (fact) if it has a source URL, a verbatim
    quote, has been verified against the source, and is not outdated.
    Anything else is downgraded to SCHATTING or HYPOTHESE.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from kansenradar.domain.enums import EvidenceType, SourceQuality


def _now() -> datetime:
    return datetime.now(UTC)


class EvidenceError(ValueError):
    """Raised when an Evidence record violates the evidence rules."""


class Evidence(BaseModel):
    id: str = Field(default_factory=lambda: f"ev_{uuid4().hex[:12]}")
    claim: str = Field(min_length=3, description="De bewering, in het Nederlands.")
    type: EvidenceType
    topic: str = Field(description="Onderzoeksvraag/criterium, bv. 'marktvraag', 'prijs'.")
    source_url: str | None = None
    source_title: str | None = None
    source_quality: SourceQuality = SourceQuality.UNKNOWN
    quote: str | None = Field(default=None, description="Letterlijk citaat uit de bron.")
    published_at: datetime | None = None
    retrieved_at: datetime | None = None
    verified: bool = False
    reasoning: str | None = Field(
        default=None, description="Voor SCHATTING/HYPOTHESE: hoe is dit afgeleid?"
    )
    downgrade_reason: str | None = None

    @model_validator(mode="after")
    def _enforce_fact_rules(self) -> Evidence:
        if self.type is EvidenceType.FACT:
            missing = self.fact_requirements_missing()
            if missing:
                raise EvidenceError(
                    "Een FEIT vereist: " + ", ".join(missing)
                    + ". Gebruik Evidence.create() om automatisch te degraderen."
                )
        if self.type is not EvidenceType.FACT and not (self.reasoning or self.source_url):
            raise EvidenceError(
                f"Een {self.type.value} vereist een onderbouwing (reasoning) of een bron."
            )
        return self

    def fact_requirements_missing(self) -> list[str]:
        missing = []
        if not self.source_url:
            missing.append("bron-URL")
        if not self.quote:
            missing.append("letterlijk citaat")
        if not self.verified:
            missing.append("verificatie tegen de bron")
        if not self.retrieved_at:
            missing.append("ophaaldatum")
        return missing

    @classmethod
    def create(cls, **data) -> Evidence:
        """Build an Evidence record, downgrading a FEIT that doesn't meet the rules.

        A FEIT with a source but missing verification becomes SCHATTING;
        a FEIT without any source becomes HYPOTHESE.
        """
        requested = EvidenceType(data.get("type", EvidenceType.HYPOTHESIS))
        if requested is EvidenceType.FACT:
            probe = cls.model_construct(**{**data, "type": EvidenceType.FACT})
            missing = probe.fact_requirements_missing()
            if missing:
                data["type"] = (
                    EvidenceType.ESTIMATE if data.get("source_url") else EvidenceType.HYPOTHESIS
                )
                data["downgrade_reason"] = "Gedegradeerd van FEIT: ontbreekt " + ", ".join(missing)
                data.setdefault("reasoning", "Bewering uit bron, nog niet geverifieerd.")
        return cls(**data)

    def is_outdated(self, max_age_days: int, now: datetime | None = None) -> bool:
        reference = self.published_at or self.retrieved_at
        if reference is None:
            return False
        now = now or _now()
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=UTC)
        return now - reference > timedelta(days=max_age_days)

    def effective_type(self, max_age_days: int, now: datetime | None = None) -> EvidenceType:
        """A verified fact that has become outdated counts as an estimate."""
        if self.type is EvidenceType.FACT and self.is_outdated(max_age_days, now):
            return EvidenceType.ESTIMATE
        return self.type

    def mark_verified(self, quote_found: bool, checked_at: datetime | None = None) -> Evidence:
        """Return a copy with the verification result applied.

        If the quote was not found in the source, a FEIT is downgraded.
        """
        checked_at = checked_at or _now()
        if quote_found:
            return self.model_copy(update={"verified": True, "retrieved_at": checked_at})
        update: dict = {"verified": False, "retrieved_at": checked_at}
        if self.type is EvidenceType.FACT:
            update["type"] = EvidenceType.ESTIMATE
            update["downgrade_reason"] = "Citaat niet teruggevonden in de bron bij verificatie."
            update["reasoning"] = self.reasoning or "Bron bevestigt het citaat niet."
        return self.model_copy(update=update)
