import json

import httpx
import pytest
from pydantic import BaseModel

from kansenradar.config import Settings
from kansenradar.llm import (
    PRESETS,
    BudgetExceededError,
    LLMClient,
    LLMOutputError,
    LLMTruncatedError,
    RunBudget,
    Tier,
    build_llm_client,
)
from kansenradar.llm.openai_compat import OpenAICompatBackend, ProviderError, extract_json


class Out(BaseModel):
    keep: bool
    reason: str


def reply(content: str, finish="stop", status=200, headers=None):
    body = {"choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
    return httpx.Response(status, json=body, headers=headers or {})


def backend_with(responses: list[httpx.Response], seen: list | None = None):
    queue = list(responses)

    def handler(request: httpx.Request):
        if seen is not None:
            seen.append(json.loads(request.content))
        return queue.pop(0)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    clock = FakeClock()
    return OpenAICompatBackend(PRESETS["groq"], api_key="test", http=http, sleep=clock.sleep, clock=clock)


class FakeClock:
    """Time that only moves when the code 'sleeps', so waiting is instant but correct."""

    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def client_for(backend):
    return LLMClient(backend, RunBudget(max_calls=10), models=PRESETS["groq"].models, free=True)


def call(client):
    return client.structured(task="t", tier=Tier.SCREENING, prompt="Beoordeel", schema=Out)


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Hier is het: {"a": 2} groetjes') == {"a": 2}


def test_successful_call_is_free_and_sends_schema():
    seen = []
    client = client_for(backend_with([reply('{"keep": true, "reason": "ok"}')], seen))
    assert call(client).keep is True
    assert client.budget.spent_usd == 0.0
    body = seen[0]
    assert body["model"] == "openai/gpt-oss-20b"
    assert body["reasoning_effort"] == "low" and body["max_tokens"] == 2_500
    assert body["response_format"] == {"type": "json_object"}
    assert '"reason"' in body["messages"][0]["content"]  # schema in system message


def test_invalid_json_gets_one_repair_attempt():
    seen = []
    client = client_for(backend_with(
        [reply('{"keep": "misschien"}'), reply('{"keep": false, "reason": "hersteld"}')], seen))
    assert call(client).reason == "hersteld"
    assert "voldeed niet aan het schema" in seen[1]["messages"][-1]["content"]


def test_still_invalid_after_repair_raises():
    client = client_for(backend_with([reply("geen json"), reply("nog steeds niet")]))
    with pytest.raises(LLMOutputError):
        call(client)


def test_rate_limit_is_retried():
    client = client_for(backend_with([
        httpx.Response(429, headers={"retry-after": "1"}),
        reply('{"keep": true, "reason": "na wachten"}'),
    ]))
    assert call(client).reason == "na wachten"


def test_provider_down_raises_provider_error():
    client = client_for(backend_with([httpx.Response(503)] * 7))
    with pytest.raises(ProviderError):
        call(client)


def test_truncated_output_raises():
    client = client_for(backend_with([reply('{"keep": tr', finish="length")]))
    with pytest.raises(LLMTruncatedError):
        call(client)


def test_call_limit_for_free_tiers():
    client = LLMClient(backend_with([reply('{"keep": true, "reason": "x"}')]),
                       RunBudget(max_calls=1), models=PRESETS["groq"].models, free=True)
    call(client)
    with pytest.raises(BudgetExceededError):
        call(client)


def test_factory_without_provider_or_key_returns_none(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert build_llm_client(Settings(llm_provider="none"), RunBudget()) is None
    assert build_llm_client(Settings(llm_provider="groq"), RunBudget()) is None
    assert build_llm_client(Settings(llm_provider="bestaat-niet"), RunBudget()) is None


def test_factory_builds_free_client_with_overrides(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    client = build_llm_client(Settings(llm_provider="groq", llm_model_analysis="ander-model"),
                              RunBudget())
    assert client.free is True
    assert client.model_for(Tier.ANALYSIS) == "ander-model"
    assert client.model_for(Tier.SCREENING) == "openai/gpt-oss-20b"


def test_ollama_needs_no_key():
    client = build_llm_client(Settings(llm_provider="ollama"), RunBudget())
    assert client is not None and client.free


def test_env_file_keys_become_environment_variables(tmp_path, monkeypatch):
    import os

    from kansenradar.config import load_env_file
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "bestaand")
    env = tmp_path / ".env"
    env.write_text('# commentaar\nGROQ_API_KEY="gsk_test"\nTAVILY_API_KEY=nieuw\n', encoding="utf-8")
    load_env_file(str(env))
    assert os.environ["GROQ_API_KEY"] == "gsk_test"
    assert os.environ["TAVILY_API_KEY"] == "bestaand"  # existing values win
    monkeypatch.delenv("GROQ_API_KEY")



def test_requests_fit_the_free_tier_token_budget():
    seen = []
    big_prompt = "woord " * 1_500  # ~3.000 tokens
    heavy = httpx.Response(200, json={"choices": [{"message": {"content": '{"keep": true, "reason": "ok"}'},
                                                    "finish_reason": "stop"}],
                                       "usage": {"prompt_tokens": 3_500, "completion_tokens": 1_500,
                                                 "total_tokens": 5_000}})
    backend = backend_with([heavy] * 4, seen)
    client = client_for(backend)
    for _ in range(4):
        client.structured(task="t", tier=Tier.SCREENING, prompt=big_prompt, schema=Out)
    # each request: prompt + max_tokens stays below the 8.000 tokens/minute limit
    assert all(b["max_tokens"] <= 2_500 for b in seen)
    # four requests of 5.000 real tokens do not fit in one minute: the backend waited instead of being refused
    assert sum(backend._sleep.__self__.slept) >= 50


def test_models_spread_over_two_budgets():
    assert PRESETS["groq"].models["analysis"] != PRESETS["groq"].models["extraction"]


def test_invalid_json_error_from_provider_is_retried_once():
    bad = httpx.Response(400, json={"error": {"code": "json_validate_failed", "message": "Failed to validate JSON"}})
    client = client_for(backend_with([bad, reply('{"keep": true, "reason": "tweede keer"}')]))
    assert call(client).reason == "tweede keer"


def test_rate_limit_backoff_waits_long_enough():
    backend = backend_with([httpx.Response(429), httpx.Response(429), reply('{"keep": true, "reason": "x"}')])
    call(client_for(backend))
    assert sum(backend._sleep.__self__.slept) >= 30  # 10 s + 20 s, not 2 s + 4 s


def test_oversized_prompt_is_refused_locally():
    client = client_for(backend_with([]))
    with pytest.raises(ProviderError, match="te groot"):
        client.structured(task="t", tier=Tier.SCREENING, prompt="woord " * 30_000, schema=Out)



def test_real_usage_replaces_pessimistic_estimate():
    # responses report 120 tokens used; the estimate per call is far higher
    backend = backend_with([reply('{"keep": true, "reason": "ok"}')] * 6)
    client = client_for(backend)
    for _ in range(6):
        client.structured(task="t", tier=Tier.SCREENING, prompt="woord " * 1_500, schema=Out)
    assert sum(backend._sleep.__self__.slept) < 20  # no minute-long waits needed



def test_daily_limit_switches_to_smaller_model_without_waiting():
    seen = []
    daily = httpx.Response(429, json={"error": {"message": "Rate limit reached for model `openai/gpt-oss-120b` "
                                                "on tokens per day (TPD): Limit 200000, Used 199500"}})
    backend = backend_with([daily, reply('{"keep": true, "reason": "klein model"}'),
                            reply('{"keep": true, "reason": "weer klein"}')], seen)
    client = client_for(backend)
    out = client.structured(task="t", tier=Tier.ANALYSIS, prompt="Beoordeel", schema=Out)
    assert out.reason == "klein model"
    assert [b["model"] for b in seen] == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    assert sum(backend._sleep.__self__.slept) < 10  # no minute-long waiting on a daily limit
    # later calls go straight to the smaller model
    client.structured(task="t", tier=Tier.ANALYSIS, prompt="Nog eens", schema=Out)
    assert seen[-1]["model"] == "openai/gpt-oss-20b"


def test_daily_limit_on_last_model_falls_back_to_rules():
    daily = httpx.Response(429, json={"error": {"message": "tokens per day (TPD) reached"}})
    client = client_for(backend_with([daily]))
    with pytest.raises(ProviderError):
        call(client)  # screening already uses the small model; caller falls back to rules
