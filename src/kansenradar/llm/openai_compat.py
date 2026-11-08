"""Backend for OpenAI-compatible chat APIs (Groq, Gemini, OpenRouter, Ollama).

Structured output: the JSON schema is put in the system message, JSON mode is
requested, and the answer is validated with Pydantic. On invalid output the
model gets one repair attempt with the validation error. Free tiers rate-limit
aggressively, so requests are spaced and 429/5xx responses are retried with backoff.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from kansenradar.llm.client import LLMResponse
from kansenradar.llm.pricing import TokenUsage
from kansenradar.llm.providers import ProviderPreset

log = logging.getLogger(__name__)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


MIN_REASONING_TOKENS = 8_000


def _is_reasoning(model: str) -> bool:
    return "gpt-oss" in model


class ProviderError(ConnectionError):
    """Provider unreachable or keeps failing; callers fall back to non-LLM paths."""


def extract_json(text: str) -> Any:
    text = _FENCE.sub("", text.strip()).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def schema_instruction(schema: type[BaseModel]) -> str:
    return (
        "Antwoord uitsluitend met één geldig JSON-object (geen tekst ervoor of erna, geen "
        "markdown) dat voldoet aan dit JSON-schema:\n"
        + json.dumps(schema.model_json_schema(), ensure_ascii=False)
    )


class OpenAICompatBackend:
    def __init__(
        self,
        preset: ProviderPreset,
        api_key: str | None = None,
        http: httpx.Client | None = None,
        max_retries: int = 4,
        sleep=time.sleep,
    ) -> None:
        self.preset = preset
        headers = {"Content-Type": "application/json", **preset.extra_headers}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.http = http or httpx.Client(timeout=preset.timeout_s)
        self.headers = headers
        self.max_retries = max_retries
        self._sleep = sleep
        self._last_request = 0.0

    # --- HTTP with spacing and retries ------------------------------------
    def _post(self, body: dict) -> dict:
        url = f"{self.preset.base_url}/chat/completions"
        for attempt in range(self.max_retries + 1):
            wait = self.preset.min_interval_s - (time.monotonic() - self._last_request)
            if wait > 0:
                self._sleep(wait)
            self._last_request = time.monotonic()
            try:
                r = self.http.post(url, json=body, headers=self.headers)
            except httpx.HTTPError as exc:
                if attempt == self.max_retries:
                    raise ProviderError(f"{self.preset.name} onbereikbaar: {exc}") from exc
                self._sleep(2 ** attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == self.max_retries:
                    raise ProviderError(f"{self.preset.name}: HTTP {r.status_code} na herhaalde pogingen")
                retry_after = r.headers.get("retry-after")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** (attempt + 1)
                from kansenradar.progress import progress
                progress(f"{self.preset.name}: limiet van de gratis versie bereikt, even wachten "
                         f"({min(delay, 60):.0f} s)" if r.status_code == 429 else
                         f"{self.preset.name}: HTTP {r.status_code}, opnieuw over {min(delay, 60):.0f} s")
                self._sleep(min(delay, 60))
                continue
            if r.status_code >= 400:
                raise ProviderError(f"{self.preset.name}: HTTP {r.status_code}: {r.text[:300]}")
            return r.json()
        raise ProviderError("onbereikbaar")  # pragma: no cover

    # --- structured call ----------------------------------------------------
    def parse(self, *, task, model, system, user, schema, max_tokens, effort, tools) -> LLMResponse:
        if tools:
            raise NotImplementedError("Server-side tools zijn niet beschikbaar bij deze provider.")
        messages = [
            {"role": "system", "content": system + "\n\n" + schema_instruction(schema)},
            {"role": "user", "content": user},
        ]
        usage = TokenUsage()
        parsed = None
        stop_reason = None
        for attempt in range(2):  # first try + one repair
            body = {
                "model": model, "messages": messages,
                # Reasoning models spend tokens on thinking first; keep enough room for the answer.
                "max_tokens": max(max_tokens, MIN_REASONING_TOKENS) if _is_reasoning(model) else max_tokens,
                "response_format": {"type": "json_object"},
            }
            if _is_reasoning(model):
                body["reasoning_effort"] = {"low": "low", "medium": "medium"}.get(effort, "high")
            data = self._post(body)
            u = data.get("usage") or {}
            usage.input_tokens += u.get("prompt_tokens", 0)
            usage.output_tokens += u.get("completion_tokens", 0)
            choice = (data.get("choices") or [{}])[0]
            finish = choice.get("finish_reason")
            stop_reason = {"length": "max_tokens", "content_filter": "refusal"}.get(finish, "end_turn")
            content = (choice.get("message") or {}).get("content") or ""
            if stop_reason != "end_turn":
                break
            try:
                parsed = schema.model_validate(extract_json(content))
                break
            except (json.JSONDecodeError, ValidationError) as exc:
                log.info("%s: ongeldige JSON voor taak %s (poging %d)", self.preset.name, task, attempt + 1)
                messages += [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": (
                        f"Je antwoord voldeed niet aan het schema: {str(exc)[:800]}\n"
                        "Geef enkel het gecorrigeerde JSON-object."
                    )},
                ]
        return LLMResponse(parsed=parsed, usage=usage, stop_reason=stop_reason, model=model)
