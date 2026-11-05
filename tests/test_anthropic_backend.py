from types import SimpleNamespace

from pydantic import BaseModel

from kansenradar.llm import AnthropicBackend


class Out(BaseModel):
    ok: bool


class FakeSDK:
    def __init__(self):
        self.kwargs = None
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    def _parse(self, **kwargs):
        self.kwargs = kwargs
        usage = SimpleNamespace(
            input_tokens=10, output_tokens=5, cache_creation_input_tokens=0,
            cache_read_input_tokens=100, server_tool_use=SimpleNamespace(web_search_requests=3),
        )
        return SimpleNamespace(parsed_output=Out(ok=True), usage=usage,
                               stop_reason="end_turn", model=kwargs["model"])


def call(model):
    sdk = FakeSDK()
    resp = AnthropicBackend(sdk).parse(task="t", model=model, system="S", user="U", schema=Out,
                                       max_tokens=100, effort="high", tools=None)
    return sdk.kwargs, resp


def test_request_shape_with_fallbacks_and_caching():
    kwargs, resp = call("claude-opus-5-5")
    assert kwargs["fallbacks"] == "default"
    assert kwargs["betas"] == ["server-side-fallback-2026-07-01"]
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kwargs["output_config"] == {"effort": "high"}
    assert "temperature" not in kwargs and "thinking" not in kwargs
    assert resp.usage.web_search_requests == 3 and resp.usage.cache_read_input_tokens == 100


def test_haiku_has_no_server_side_fallback():
    kwargs, _ = call("claude-haiku-5-5")
    assert "fallbacks" not in kwargs and "betas" not in kwargs
