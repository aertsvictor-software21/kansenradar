import pytest
from pydantic import BaseModel

from kansenradar.llm import (
    BudgetExceededError,
    FakeBackend,
    LLMClient,
    LLMRefusalError,
    LLMTruncatedError,
    RunBudget,
    Tier,
    TokenUsage,
    cost_usd,
)
from kansenradar.llm.prompts import BASE_SYSTEM_PROMPT


class Verdict(BaseModel):
    keep: bool
    reason: str


def make_client(backend, max_usd=10.0):
    return LLMClient(backend, RunBudget(max_usd))


def test_cost_calculation_opus():
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=100_000, web_search_requests=10)
    # 4.00 + 2.00 + 0.10
    assert cost_usd("claude-opus-5-5", usage) == pytest.approx(6.10)


def test_cost_calculation_cache_and_haiku_long_prompt():
    cached = TokenUsage(cache_read_input_tokens=1_000_000, cache_creation_input_tokens=1_000_000)
    assert cost_usd("claude-sonnet-5-5", cached) == pytest.approx(0.20 + 2.50)
    long = TokenUsage(input_tokens=200_000, output_tokens=0)
    assert cost_usd("claude-haiku-5-5", long) == pytest.approx(0.10)  # 200k x $0.50/M


def test_structured_call_routes_tier_and_records_cost():
    backend = FakeBackend({"screen": Verdict(keep=False, reason="Generiek idee.")})
    client = make_client(backend)
    out = client.structured(task="screen", tier=Tier.SCREENING, prompt="...", schema=Verdict)
    assert out.keep is False
    assert backend.calls[0]["model"] == "claude-haiku-5-5"
    assert backend.calls[0]["effort"] == "low"
    assert client.budget.by_task()["screen"] > 0


def test_budget_blocks_call_that_could_exceed_ceiling():
    backend = FakeBackend({"deep": Verdict(keep=True, reason="x")})
    client = make_client(backend, max_usd=0.05)
    with pytest.raises(BudgetExceededError):
        # worst case 16k output tokens on Opus = $0.32 > $0.05
        client.structured(task="deep", tier=Tier.ANALYSIS, prompt="...", schema=Verdict)
    assert backend.calls == []


def test_refusal_and_truncation_raise():
    for stop, exc in [("refusal", LLMRefusalError), ("max_tokens", LLMTruncatedError)]:
        backend = FakeBackend({"t": Verdict(keep=True, reason="x")}, stop_reason=stop)
        with pytest.raises(exc):
            make_client(backend).structured(task="t", tier=Tier.EXTRACTION, prompt="p", schema=Verdict)


def test_system_prompt_is_cache_stable():
    # The cached prefix must not contain per-request data.
    for marker in ["{", "2026", "run_", "idea_"]:
        assert marker not in BASE_SYSTEM_PROMPT
