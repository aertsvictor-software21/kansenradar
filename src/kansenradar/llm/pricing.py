"""Price table (USD per 1M tokens) and cost calculation.

Source: Anthropic first-party API prices (checked 2026-10). Cache writes are
billed at 1.25x input (5-minute TTL). Web search price is an ASSUMPTION
(USD 10 per 1,000 searches) that must be checked against the live pricing page.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

WEB_SEARCH_USD_PER_REQUEST = 10.0 / 1000  # assumption, verify
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass(frozen=True)
class ModelPrice:
    input: float
    output: float
    cache_read: float
    # Some models charge more for long prompts.
    long_prompt_threshold: int | None = None
    long_input: float | None = None
    long_output: float | None = None


PRICES: dict[str, ModelPrice] = {
    "claude-opus-5-5": ModelPrice(input=4.00, output=20.00, cache_read=0.20),
    "claude-sonnet-5-5": ModelPrice(input=2.00, output=10.00, cache_read=0.20),
    "claude-haiku-5-5": ModelPrice(
        input=0.10, output=0.50, cache_read=0.01,  # cache_read assumed at 0.1x input
        long_prompt_threshold=100_000, long_input=0.50, long_output=2.50,
    ),
}


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    web_search_requests: int = 0

    @property
    def total_prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens


class UnknownModelError(KeyError):
    pass


def price_for(model: str) -> ModelPrice:
    try:
        return PRICES[model]
    except KeyError as exc:
        raise UnknownModelError(f"Geen prijs bekend voor model '{model}'.") from exc


def cost_usd(model: str, usage: TokenUsage) -> float:
    p = price_for(model)
    input_rate, output_rate = p.input, p.output
    if p.long_prompt_threshold and usage.total_prompt_tokens > p.long_prompt_threshold:
        input_rate = p.long_input or input_rate
        output_rate = p.long_output or output_rate
    per_m = 1_000_000
    return (
        usage.input_tokens * input_rate / per_m
        + usage.cache_creation_input_tokens * input_rate * CACHE_WRITE_MULTIPLIER / per_m
        + usage.cache_read_input_tokens * p.cache_read / per_m
        + usage.output_tokens * output_rate / per_m
        + usage.web_search_requests * WEB_SEARCH_USD_PER_REQUEST
    )


def worst_case_output_cost(model: str, max_tokens: int) -> float:
    p = price_for(model)
    rate = max(p.output, p.long_output or 0)
    return max_tokens * rate / 1_000_000
