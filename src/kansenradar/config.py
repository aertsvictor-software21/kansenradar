"""Central configuration, read from environment variables (prefix KR_) or .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KR_", env_file=".env", extra="ignore")

    # Target markets: Belgium and the Netherlands.
    target_markets: tuple[str, ...] = ("BE", "NL")

    database_url: str = "sqlite:///kansenradar.db"

    # Model per tier. Analysis/red team need the strongest model; screening is bulk work.
    model_analysis: str = "claude-opus-5-5"
    model_extraction: str = "claude-sonnet-5-5"
    model_screening: str = "claude-haiku-5-5"

    # Hard cost ceiling per pipeline run (USD). The run stops when it is reached.
    max_run_cost_usd: float = 30.0

    # Facts older than this are flagged as outdated and no longer count as FEIT.
    evidence_max_age_days: int = 365

    # Embedder for duplicate detection: "auto" (local semantic model if available,
    # otherwise lexical), "fastembed" or "hashing". Thresholds live on each embedder
    # (see history/embeddings.py) because their similarity scales differ.
    embedder: str = "auto"


@lru_cache
def get_settings() -> Settings:
    return Settings()
