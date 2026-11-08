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
from collections import defaultdict, deque
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
        max_retries: int = 6,
        sleep=time.sleep,
        clock=time.monotonic,
    ) -> None:
        self.preset = preset
        headers = {"Content-Type": "application/json", **preset.extra_headers}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.http = http or httpx.Client(timeout=preset.timeout_s)
        self.headers = headers
        self.max_retries = max_retries
        self._sleep = sleep
        self._clock = clock
        self._last_request = 0.0
        self._token_log: dict[str, deque] = defaultdict(deque)  # model -> (time, tokens)

    # --- free-tier token budget ---------------------------------------------
    def _reserve_tokens(self, model: str, tokens: int) -> None:
        """Wait until this request fits in the model's tokens-per-minute budget."""
        limit = self.preset.tpm_limit
        if not limit:
            return
        log_ = self._token_log[model]
        while True:
            now = self._clock()
            while log_ and now - log_[0][0] >= 60:
                log_.popleft()
            used = sum(t for _, t in log_)
            if used + tokens <= limit or not log_:
                log_.append((now, tokens))
                return
            wait = 60 - (now - log_[0][0]) + 0.5
            from kansenradar.progress import progress
            progress(f"{self.preset.name}: gratis limiet per minuut bijna op, even wachten ({wait:.0f} s)")
            self._sleep(wait)

    # --- HTTP with spacing and retries ------------------------------------
    def _post(self, body: dict, est_tokens: int = 0) -> dict:
        url = f"{self.preset.base_url}/chat/completions"
        self._reserve_tokens(body["model"], est_tokens)  # once per request; retries back off instead
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
                try:
                    retry_after = float(r.headers.get("retry-after", ""))
                except ValueError:
                    retry_after = 0.0
                # Free-tier limits are per minute: back off generously.
                delay = max(retry_after, 10.0 * (attempt + 1)) if r.status_code == 429 else 2 ** (attempt + 1)
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

    def _output_budget(self, model: str, requested: int, prompt_tokens: int) -> int:
        cap = self.preset.max_output_tokens
        if _is_reasoning(model):
            # Reasoning models think before answering; without a cap give them plenty of room.
            out = cap if cap else max(requested, MIN_REASONING_TOKENS)
        else:
            out = min(requested, cap) if cap else requested
        if self.preset.tpm_limit:
            out = min(out, int(self.preset.tpm_limit * 0.95) - prompt_tokens)
            if out < 400:
                raise ProviderError(f"{self.preset.name}: aanvraag te groot voor de gratis limiet per minuut")
        return out

    def _reasoning_effort(self, effort: str) -> str:
        if self.preset.tpm_limit:  # tight free-tier budget: think briefly
            return "medium" if effort in ("high", "xhigh", "max") else "low"
        return {"low": "low", "medium": "medium"}.get(effort, "high")

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
            prompt_tokens = sum(len(m["content"]) for m in messages) // 3 + 50  # conservative estimate
            out_tokens = self._output_budget(model, max_tokens, prompt_tokens)
            body = {
                "model": model, "messages": messages, "max_tokens": out_tokens,
                "response_format": {"type": "json_object"},
            }
            if _is_reasoning(model):
                body["reasoning_effort"] = self._reasoning_effort(effort)
            try:
                data = self._post(body, prompt_tokens + out_tokens)
            except ProviderError as exc:
                # The provider rejected malformed model output: give it one more chance.
                if any(code in str(exc) for code in ("json_validate_failed", "output_parse_failed")):
                    log.info("%s: model gaf geen geldig JSON voor taak %s (poging %d)",
                             self.preset.name, task, attempt + 1)
                    if attempt == 0:
                        continue
                    break
                raise
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
