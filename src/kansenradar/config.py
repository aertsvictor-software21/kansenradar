"""Central configuration, read from environment variables (prefix KR_) or .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KR_", env_file=".env", extra="ignore")

    # Target markets: Belgium and the Netherlands.
    target_markets: tuple[str, ...] = ("BE", "NL")

    database_url: str = "sqlite:///kansenradar.db"

    # LLM provider: none | groq | gemini | openrouter | ollama | anthropic (paid).
    # Default "none": the platform works without any LLM (non-LLM fallbacks).
    llm_provider: str = "none"
    # Optional model overrides per tier; empty = provider default (llm/providers.py).
    llm_model_analysis: str | None = None
    llm_model_extraction: str | None = None
    llm_model_screening: str | None = None

    # Hard ceilings per pipeline run. Money only applies to paid providers (budget: 0).
    max_run_cost_usd: float = 0.0
    # Free tiers have daily request limits; this caps calls per run.
    max_llm_calls_per_run: int = 300

    # Facts older than this are flagged as outdated and no longer count as FEIT.
    evidence_max_age_days: int = 365

    # Embedder for duplicate detection: "auto" (local semantic model if available,
    # otherwise lexical), "fastembed" or "hashing". Thresholds live on each embedder
    # (see history/embeddings.py) because their similarity scales differ.
    embedder: str = "auto"

    # Web search for market research: none | searxng | tavily (free tier, TAVILY_API_KEY).
    search_provider: str = "none"
    searxng_url: str = "http://localhost:8888"
    # Official open data (Eurostat, free, no key). Off = fully offline.
    use_open_data: bool = True


def load_env_file(path: str = ".env") -> None:
    """Make keys from .env (e.g. GROQ_API_KEY, TAVILY_API_KEY) visible as environment
    variables. Existing environment variables always win."""
    import os
    from pathlib import Path

    file = Path(path)
    if not file.is_file():
        return
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@lru_cache
def get_settings() -> Settings:
    load_env_file()
    return Settings()
