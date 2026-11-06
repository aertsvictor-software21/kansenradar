"""LLM provider presets and the factory that builds an LLMClient from settings.

All providers except 'anthropic' speak the OpenAI-compatible chat API, so one
backend (openai_compat.py) covers them. Model names change often: they are
defaults only and can be overridden with KR_LLM_MODEL_ANALYSIS / _EXTRACTION /
_SCREENING. Check the provider's model list if a call returns 'model not found'.

Free tiers (status as known in 2026; verify yourself):
  - groq:       free tier with per-minute/day limits; needs a free API key.
  - gemini:     Google AI Studio free tier; needs a free API key. Input on the
                free tier may be used by Google to improve its products.
  - openrouter: ':free' models with daily limits; needs a free API key.
  - ollama:     fully local and offline; no key; slow on a CPU-only Chromebook.
  - anthropic:  paid; optional.
  - none:       no LLM; the platform uses its non-LLM fallbacks.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from kansenradar.config import Settings
from kansenradar.llm.budget import RunBudget

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderPreset:
    name: str
    base_url: str | None
    api_key_env: str | None
    models: dict[str, str]  # tier value -> model name
    free: bool
    min_interval_s: float = 0.0  # spacing between requests, for free-tier rate limits
    timeout_s: float = 120.0
    extra_headers: dict[str, str] = field(default_factory=dict)


PRESETS: dict[str, ProviderPreset] = {
    "groq": ProviderPreset(
        name="groq", base_url="https://api.groq.com/openai/v1", api_key_env="GROQ_API_KEY",
        # Llama 3.x models were shut down for free tiers on 2026-08-16; Groq's replacements:
        models={"analysis": "openai/gpt-oss-120b", "extraction": "openai/gpt-oss-120b",
                "screening": "openai/gpt-oss-20b"},
        free=True, min_interval_s=2.5,
    ),
    "gemini": ProviderPreset(
        name="gemini", base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
        models={"analysis": "gemini-2.5-flash", "extraction": "gemini-2.5-flash",
                "screening": "gemini-2.5-flash-lite"},
        free=True, min_interval_s=6.0,
    ),
    "openrouter": ProviderPreset(
        name="openrouter", base_url="https://openrouter.ai/api/v1", api_key_env="OPENROUTER_API_KEY",
        models={t: "meta-llama/llama-3.3-70b-instruct:free"
                for t in ("analysis", "extraction", "screening")},
        free=True, min_interval_s=4.0, extra_headers={"X-Title": "Kansenradar"},
    ),
    "ollama": ProviderPreset(
        name="ollama", base_url="http://localhost:11434/v1", api_key_env=None,
        models={t: "qwen2.5:7b" for t in ("analysis", "extraction", "screening")},
        free=True, timeout_s=900.0,
    ),
    "anthropic": ProviderPreset(
        name="anthropic", base_url=None, api_key_env="ANTHROPIC_API_KEY",
        models={"analysis": "claude-opus-5-5", "extraction": "claude-sonnet-5-5",
                "screening": "claude-haiku-5-5"},
        free=False,
    ),
}


def resolve_models(settings: Settings, preset: ProviderPreset) -> dict[str, str]:
    overrides = {
        "analysis": settings.llm_model_analysis,
        "extraction": settings.llm_model_extraction,
        "screening": settings.llm_model_screening,
    }
    return {tier: overrides[tier] or preset.models[tier] for tier in preset.models}


def build_llm_client(settings: Settings, budget: RunBudget):
    """Return an LLMClient for the configured provider, or None if no LLM is usable.

    Never raises for a missing key or provider: the platform must keep working.
    """
    from kansenradar.llm.client import AnthropicBackend, LLMClient
    from kansenradar.llm.openai_compat import OpenAICompatBackend

    name = settings.llm_provider.lower()
    if name == "none":
        return None
    preset = PRESETS.get(name)
    if preset is None:
        log.warning("Onbekende LLM-provider '%s'; verder zonder LLM.", name)
        return None
    key = os.environ.get(preset.api_key_env) if preset.api_key_env else None
    if preset.api_key_env and not key:
        log.warning("Geen %s ingesteld; verder zonder LLM.", preset.api_key_env)
        return None
    if name == "anthropic":
        try:
            backend = AnthropicBackend()
        except ImportError:
            log.warning("Pakket 'anthropic' niet geïnstalleerd; verder zonder LLM.")
            return None
    else:
        backend = OpenAICompatBackend(preset, api_key=key)
    return LLMClient(backend, budget, models=resolve_models(settings, preset), free=preset.free)
