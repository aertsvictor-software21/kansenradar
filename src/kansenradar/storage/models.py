"""SQLAlchemy ORM tables for the historical database.

Portable between SQLite (dev/tests) and PostgreSQL (production). Embeddings are
stored as JSON arrays; at the expected scale (thousands of ideas) in-process
cosine similarity is fast enough. pgvector can replace this later without
changing the repository interface.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class RunRow(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(30), default="bezig")
    max_cost_usd: Mapped[float] = mapped_column(Float)
    spent_usd: Mapped[float] = mapped_column(Float, default=0.0)
    params: Mapped[dict] = mapped_column(JSON, default=dict)


class IdeaRow(Base):
    __tablename__ = "ideas"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    title: Mapped[str] = mapped_column(String(200))
    sector: Mapped[str] = mapped_column(String(80), index=True)
    status: Mapped[str] = mapped_column(String(40), index=True)
    duplicate_of: Mapped[str | None] = mapped_column(String(40))
    data: Mapped[dict] = mapped_column(JSON)  # full Idea model dump
    embedding: Mapped[list | None] = mapped_column(JSON)
    embedding_model: Mapped[str | None] = mapped_column(String(80))
    score_total: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)
    conclusion: Mapped[str | None] = mapped_column(String(60))

    evidence: Mapped[list[EvidenceRow]] = relationship(
        back_populates="idea", cascade="all, delete-orphan"
    )


class EvidenceRow(Base):
    __tablename__ = "evidence"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    idea_id: Mapped[str] = mapped_column(ForeignKey("ideas.id"), index=True)
    type: Mapped[str] = mapped_column(String(20), index=True)
    topic: Mapped[str] = mapped_column(String(80), index=True)
    claim: Mapped[str] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(Text)
    verified: Mapped[bool] = mapped_column(default=False)
    data: Mapped[dict] = mapped_column(JSON)  # full Evidence model dump

    idea: Mapped[IdeaRow] = relationship(back_populates="evidence")


class CostRow(Base):
    __tablename__ = "costs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    task: Mapped[str] = mapped_column(String(80))
    model: Mapped[str] = mapped_column(String(80))
    cost_usd: Mapped[float] = mapped_column(Float)
    usage: Mapped[dict] = mapped_column(JSON)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
