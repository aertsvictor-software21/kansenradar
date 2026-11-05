"""Repository: the only place that reads/writes the database."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from kansenradar.domain import Evidence, Idea, IdeaStatus
from kansenradar.llm.budget import CostEntry
from kansenradar.storage.models import Base, CostRow, EvidenceRow, IdeaRow, RunRow


def make_engine(url: str) -> Engine:
    return create_engine(url, future=True)


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embeddings hebben verschillende dimensies.")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


@dataclass(frozen=True)
class SimilarIdea:
    idea_id: str
    title: str
    similarity: float
    status: str
    conclusion: str | None
    created_at: datetime


class Repository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        Base.metadata.create_all(engine)
        self._session = sessionmaker(engine, expire_on_commit=False)

    def session(self) -> Session:
        return self._session()

    # --- runs -------------------------------------------------------------
    def start_run(self, max_cost_usd: float, params: dict | None = None) -> str:
        run_id = f"run_{uuid4().hex[:12]}"
        with self.session() as s, s.begin():
            s.add(RunRow(id=run_id, max_cost_usd=max_cost_usd, params=params or {}))
        return run_id

    def finish_run(self, run_id: str, status: str, cost_entries: list[CostEntry]) -> None:
        with self.session() as s, s.begin():
            run = s.get(RunRow, run_id)
            if run is None:
                raise KeyError(run_id)
            for e in cost_entries:
                s.add(CostRow(
                    run_id=run_id, task=e.task, model=e.model, cost_usd=e.cost_usd,
                    usage=e.usage.model_dump(), at=e.at,
                ))
            run.spent_usd = sum(e.cost_usd for e in cost_entries)
            run.status = status
            run.finished_at = datetime.now(UTC)

    # --- ideas ------------------------------------------------------------
    def save_idea(
        self,
        idea: Idea,
        embedding: list[float] | None = None,
        embedding_model: str | None = None,
    ) -> None:
        with self.session() as s, s.begin():
            row = s.get(IdeaRow, idea.id) or IdeaRow(id=idea.id)
            row.run_id = idea.run_id
            row.created_at = idea.created_at
            row.title = idea.title
            row.sector = idea.sector.value
            row.status = idea.status.value
            row.duplicate_of = idea.duplicate_of
            row.data = idea.model_dump(mode="json")
            if embedding is not None:
                row.embedding = embedding
                row.embedding_model = embedding_model
            s.add(row)

    def get_idea(self, idea_id: str) -> Idea | None:
        with self.session() as s:
            row = s.get(IdeaRow, idea_id)
            return Idea.model_validate(row.data) if row else None

    def list_ideas(self, status: IdeaStatus | None = None) -> list[Idea]:
        with self.session() as s:
            q = select(IdeaRow).order_by(IdeaRow.created_at)
            if status:
                q = q.where(IdeaRow.status == status.value)
            return [Idea.model_validate(r.data) for r in s.scalars(q)]

    def used_seed_keys(self) -> set[str]:
        """Seed keys of the originality engine that earlier runs already used."""
        with self.session() as s:
            return {d.get("seed_key") for d in s.scalars(select(IdeaRow.data)) if d.get("seed_key")}

    def most_similar(
        self, embedding: list[float], embedding_model: str, limit: int = 5
    ) -> list[SimilarIdea]:
        """Closest stored ideas (excluding duplicates) embedded with the same model."""
        with self.session() as s:
            rows = s.scalars(
                select(IdeaRow).where(
                    IdeaRow.embedding_model == embedding_model,
                    IdeaRow.status != IdeaStatus.DUPLICATE.value,
                )
            ).all()
            scored = [
                SimilarIdea(r.id, r.title, cosine(embedding, r.embedding), r.status,
                            r.conclusion, r.created_at)
                for r in rows if r.embedding
            ]
        return sorted(scored, key=lambda t: t.similarity, reverse=True)[:limit]

    def ideas_without_embedding(self, embedding_model: str) -> list[Idea]:
        """Ideas not (yet) embedded with this model, e.g. after switching embedder."""
        with self.session() as s:
            rows = s.scalars(
                select(IdeaRow).where(
                    (IdeaRow.embedding_model != embedding_model) | (IdeaRow.embedding_model.is_(None))
                )
            ).all()
            return [Idea.model_validate(r.data) for r in rows]

    def set_embedding(self, idea_id: str, embedding: list[float], embedding_model: str) -> None:
        with self.session() as s, s.begin():
            row = s.get(IdeaRow, idea_id)
            if row is None:
                raise KeyError(idea_id)
            row.embedding = embedding
            row.embedding_model = embedding_model

    # --- evidence ---------------------------------------------------------
    def add_evidence(self, idea_id: str, items: list[Evidence]) -> None:
        with self.session() as s, s.begin():
            for ev in items:
                s.merge(EvidenceRow(
                    id=ev.id, idea_id=idea_id, type=ev.type.value, topic=ev.topic,
                    claim=ev.claim, source_url=ev.source_url, verified=ev.verified,
                    data=ev.model_dump(mode="json"),
                ))

    def evidence_for(self, idea_id: str) -> list[Evidence]:
        with self.session() as s:
            rows = s.scalars(select(EvidenceRow).where(EvidenceRow.idea_id == idea_id)).all()
            return [Evidence.model_validate(r.data) for r in rows]
