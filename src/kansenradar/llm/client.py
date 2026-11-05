"""LLM access layer.

All modules call LLMClient.structured(); none talk to the SDK directly. This gives
one place for: model choice per tier, structured (Pydantic-validated) output,
prompt caching of the shared system prompt, refusal fallbacks, stop-reason
handling, and cost accounting against the run budget.

The backend is swappable so tests run on recorded/fake responses without API calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Protocol, TypeVar

from pydantic import BaseModel

from kansenradar.config import Settings, get_settings
from kansenradar.llm.budget import RunBudget
from kansenradar.llm.pricing import TokenUsage
from kansenradar.llm.prompts import BASE_SYSTEM_PROMPT

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Models with server-side refusal fallback support (Claude API).
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class Tier(StrEnum):
    ANALYSIS = "analysis"      # deep research, red team, scoring rationale
    EXTRACTION = "extraction"  # claim extraction, structuring scraped pages
    SCREENING = "screening"    # bulk first-pass filtering


class LLMError(RuntimeError):
    pass


class LLMRefusalError(LLMError):
    pass


class LLMTruncatedError(LLMError):
    pass


class LLMOutputError(LLMError):
    pass


@dataclass
class LLMResponse:
    parsed: BaseModel | None
    usage: TokenUsage
    stop_reason: str | None
    model: str


class LLMBackend(Protocol):
    def parse(
        self,
        *,
        task: str,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
        max_tokens: int,
        effort: str,
        tools: list[dict[str, Any]] | None,
    ) -> LLMResponse: ...


class AnthropicBackend:
    """Real backend on the Anthropic SDK (beta namespace, for refusal fallbacks)."""

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client

    def parse(self, *, task, model, system, user, schema, max_tokens, effort, tools) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_format=schema,
            output_config={"effort": effort},
        )
        if tools:
            kwargs["tools"] = tools
        if model in _FALLBACK_MODELS:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        response = self._client.beta.messages.parse(**kwargs)
        u = response.usage
        server_tool_use = getattr(u, "server_tool_use", None)
        usage = TokenUsage(
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            web_search_requests=getattr(server_tool_use, "web_search_requests", 0) or 0,
        )
        return LLMResponse(
            parsed=getattr(response, "parsed_output", None),
            usage=usage,
            stop_reason=response.stop_reason,
            model=response.model or model,
        )


class FakeBackend:
    """Test backend: returns canned outputs per task name.

    handlers maps a task name to either a BaseModel instance or a callable
    (user_prompt) -> BaseModel. Every call is recorded in self.calls.
    """

    def __init__(
        self,
        handlers: dict[str, BaseModel | Callable[[str], BaseModel]],
        usage: TokenUsage | None = None,
        stop_reason: str = "end_turn",
    ) -> None:
        self.handlers = handlers
        self.usage = usage or TokenUsage(input_tokens=1000, output_tokens=500)
        self.stop_reason = stop_reason
        self.calls: list[dict[str, Any]] = []

    def parse(self, *, task, model, system, user, schema, max_tokens, effort, tools) -> LLMResponse:
        self.calls.append(dict(task=task, model=model, user=user, effort=effort))
        handler = self.handlers.get(task)
        if handler is None:
            raise KeyError(f"FakeBackend: geen handler voor taak '{task}'")
        parsed = handler(user) if callable(handler) else handler
        if not isinstance(parsed, schema):
            parsed = schema.model_validate(parsed.model_dump())
        return LLMResponse(parsed=parsed, usage=self.usage, stop_reason=self.stop_reason, model=model)


class LLMClient:
    def __init__(
        self,
        backend: LLMBackend,
        budget: RunBudget,
        settings: Settings | None = None,
    ) -> None:
        self.backend = backend
        self.budget = budget
        self.settings = settings or get_settings()

    def model_for(self, tier: Tier) -> str:
        return {
            Tier.ANALYSIS: self.settings.model_analysis,
            Tier.EXTRACTION: self.settings.model_extraction,
            Tier.SCREENING: self.settings.model_screening,
        }[tier]

    def structured(
        self,
        *,
        task: str,
        tier: Tier,
        prompt: str,
        schema: type[T],
        system_extra: str = "",
        effort: str | None = None,
        max_tokens: int = 16_000,
        tools: list[dict[str, Any]] | None = None,
    ) -> T:
        model = self.model_for(tier)
        effort = effort or {Tier.ANALYSIS: "high", Tier.EXTRACTION: "medium", Tier.SCREENING: "low"}[tier]
        self.budget.ensure_can_afford(model, max_tokens)

        # The stable base prompt comes first so the cached prefix is shared across tasks.
        system = BASE_SYSTEM_PROMPT + ("\n\n" + system_extra if system_extra else "")

        response = self.backend.parse(
            task=task, model=model, system=system, user=prompt, schema=schema,
            max_tokens=max_tokens, effort=effort, tools=tools,
        )
        entry = self.budget.record(task, response.model, response.usage)
        log.info("llm task=%s model=%s cost=$%.4f", task, response.model, entry.cost_usd)

        if response.stop_reason == "refusal":
            raise LLMRefusalError(f"Het model weigerde taak '{task}'.")
        if response.stop_reason == "max_tokens":
            raise LLMTruncatedError(
                f"Antwoord voor taak '{task}' afgekapt bij max_tokens={max_tokens}."
            )
        if response.parsed is None:
            raise LLMOutputError(f"Geen geldig gestructureerd antwoord voor taak '{task}'.")
        return response.parsed  # type: ignore[return-value]
