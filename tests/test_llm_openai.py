import builtins
import json
from types import SimpleNamespace

import httpx2
import openai
import pytest
from pydantic import BaseModel, ValidationError

from prep_agent.llm.base import (
    LLMAPIError,
    LLMClient,
    LLMConfigError,
    LLMRefusalError,
    LLMResult,
    LLMSchemaError,
    TokenUsage,
)
from prep_agent.llm.openai import DEFAULT_MODEL, OpenAIClient

SECRET = "sk-test-SECRET-0123456789"


class Answer(BaseModel):
    value: int


def _request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.openai.test/v1/responses")


def _response(status: int) -> httpx2.Response:
    return httpx2.Response(status, request=_request())


def _message(*content):
    return SimpleNamespace(type="message", content=list(content))


def _text_item(parsed):
    return SimpleNamespace(type="output_text", parsed=parsed)


def _reply(
    parsed=None,
    *,
    content=None,
    model="gpt-test-2026",
    usage=(11, 7),
    status="completed",
    reason=None,
):
    """Shape of the parts of a parsed Responses API result that the adapter reads."""
    if content is None:
        content = [_text_item(parsed)]
    return SimpleNamespace(
        status=status,
        incomplete_details=None if reason is None else SimpleNamespace(reason=reason),
        output=[_message(*content)],
        output_parsed=parsed,
        model=model,
        usage=None
        if usage is None
        else SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1]),
    )


class FakeSDK:
    """Stands in for `openai.OpenAI`: records `responses.parse` kwargs, replays one outcome."""

    def __init__(self, outcome):
        self.api_key = SECRET
        self.calls: list[dict] = []
        self._outcome = outcome
        self.responses = SimpleNamespace(parse=self._parse)

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


def make_client(outcome, **kwargs) -> tuple[OpenAIClient, FakeSDK]:
    sdk = FakeSDK(outcome)
    return OpenAIClient(client=sdk, **kwargs), sdk


def validation_error(value) -> ValidationError:
    with pytest.raises(ValidationError) as caught:
        Answer.model_validate({"value": value})
    return caught.value


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("OPENAI_API_KEY", "PREP_AGENT_MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_satisfies_the_protocol():
    client, _ = make_client(_reply(Answer(value=1)))

    assert isinstance(client, LLMClient)


def test_sends_messages_schema_and_model_to_the_responses_api():
    client, sdk = make_client(_reply(Answer(value=1)), model="gpt-explicit")

    client.generate("the system", "the user", Answer)

    [call] = sdk.calls
    assert call["model"] == "gpt-explicit"
    assert call["instructions"] == "the system"
    assert call["input"] == "the user"
    assert call["text_format"] is Answer


def test_does_not_ask_the_provider_to_store_the_exchange():
    client, sdk = make_client(_reply(Answer(value=1)))

    client.generate("s", "u", Answer)

    assert sdk.calls[0]["store"] is False


def test_successful_response_is_parsed_into_a_result():
    client, _ = make_client(_reply(Answer(value=5), model="gpt-served", usage=(100, 20)))

    result = client.generate("s", "u", Answer)

    assert isinstance(result, LLMResult)
    assert result.parsed == Answer(value=5)
    assert result.model == "gpt-served"
    assert result.usage == TokenUsage(input_tokens=100, output_tokens=20)


def test_missing_usage_and_model_fall_back_gracefully():
    client, _ = make_client(_reply(Answer(value=1), usage=None, model=None), model="gpt-configured")

    result = client.generate("s", "u", Answer)

    assert result.usage is None
    assert result.model == "gpt-configured"


def test_model_defaults_to_the_documented_default():
    client, sdk = make_client(_reply(Answer(value=1)))

    client.generate("s", "u", Answer)

    assert sdk.calls[0]["model"] == DEFAULT_MODEL


def test_model_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("PREP_AGENT_MODEL", "gpt-from-env")
    client, sdk = make_client(_reply(Answer(value=1)))

    client.generate("s", "u", Answer)

    assert sdk.calls[0]["model"] == "gpt-from-env"


def test_explicit_model_beats_the_environment(monkeypatch):
    monkeypatch.setenv("PREP_AGENT_MODEL", "gpt-from-env")
    client, sdk = make_client(_reply(Answer(value=1)), model="gpt-explicit")

    client.generate("s", "u", Answer)

    assert sdk.calls[0]["model"] == "gpt-explicit"


def test_blank_environment_model_is_ignored(monkeypatch):
    monkeypatch.setenv("PREP_AGENT_MODEL", "  ")
    client, sdk = make_client(_reply(Answer(value=1)))

    client.generate("s", "u", Answer)

    assert sdk.calls[0]["model"] == DEFAULT_MODEL


def test_refusal_raises_refusal_error_with_the_refusal_text():
    refusal = SimpleNamespace(type="refusal", refusal="I cannot help with that.")
    client, _ = make_client(_reply(content=[refusal]))

    with pytest.raises(LLMRefusalError, match="I cannot help with that"):
        client.generate("s", "u", Answer)


def test_content_filter_stop_raises_refusal_error():
    client, _ = make_client(_reply(None, status="incomplete", reason="content_filter"))

    with pytest.raises(LLMRefusalError, match="content filter"):
        client.generate("s", "u", Answer)


def test_validation_error_from_the_sdk_raises_schema_error():
    client, _ = make_client(validation_error("x"))

    with pytest.raises(LLMSchemaError):
        client.generate("s", "u", Answer)


def test_response_without_parsed_output_raises_schema_error():
    client, _ = make_client(_reply(None))

    with pytest.raises(LLMSchemaError):
        client.generate("s", "u", Answer)


def test_truncated_response_raises_schema_error_naming_the_reason():
    client, _ = make_client(_reply(None, status="incomplete", reason="max_output_tokens"))

    with pytest.raises(LLMSchemaError, match="max_output_tokens"):
        client.generate("s", "u", Answer)


@pytest.mark.parametrize(
    "error",
    [
        openai.APIConnectionError(request=_request()),
        openai.APITimeoutError(request=_request()),
        openai.RateLimitError("slow down", response=_response(429), body=None),
        openai.InternalServerError("boom", response=_response(500), body=None),
        openai.AuthenticationError("bad key", response=_response(401), body=None),
        openai.BadRequestError("bad schema", response=_response(400), body=None),
        openai.OpenAIError("something else"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_sdk_errors_become_api_errors(error):
    client, _ = make_client(error)

    with pytest.raises(LLMAPIError):
        client.generate("s", "u", Answer)


def test_api_error_message_names_the_status_code():
    error = openai.RateLimitError("slow down", response=_response(429), body=None)
    client, _ = make_client(error)

    with pytest.raises(LLMAPIError, match="429"):
        client.generate("s", "u", Answer)


def test_real_sdk_client_sends_a_strict_json_schema_request_and_parses_the_reply():
    seen = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        body = {
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "gpt-served",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": '{"value": 3}', "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": 12,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 4,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 16,
            },
        }
        return httpx2.Response(200, json=body)

    sdk = openai.OpenAI(
        api_key=SECRET,
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )

    result = OpenAIClient(model="gpt-x", client=sdk).generate("the system", "the user", Answer)

    assert result.parsed == Answer(value=3)
    assert result.model == "gpt-served"
    assert result.usage == TokenUsage(input_tokens=12, output_tokens=4)
    [request] = seen
    assert request.url.path.endswith("/responses")
    body = json.loads(request.content)
    assert body["model"] == "gpt-x"
    assert body["instructions"] == "the system"
    assert body["input"] == "the user"
    assert body["store"] is False
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["strict"] is True
    assert body["text"]["format"]["schema"]["required"] == ["value"]
    assert body["text"]["format"]["schema"]["additionalProperties"] is False


def test_missing_api_key_raises_config_error_that_names_the_variable():
    with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
        OpenAIClient()


def test_blank_api_key_raises_config_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "   ")

    with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
        OpenAIClient()


def test_api_key_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)

    assert isinstance(OpenAIClient(), LLMClient)


def test_missing_openai_package_raises_config_error(monkeypatch):
    real_import = builtins.__import__

    def import_without_openai(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setattr(builtins, "__import__", import_without_openai)

    with pytest.raises(LLMConfigError, match="prep-agent\\[openai\\]|--extra openai"):
        OpenAIClient()


def test_repr_does_not_contain_the_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)

    client = OpenAIClient(model="gpt-x")

    assert SECRET not in repr(client)
    assert SECRET not in str(client)
    assert "gpt-x" in repr(client)


def test_key_echoed_by_the_provider_is_redacted_from_api_errors():
    error = openai.AuthenticationError(
        f"Incorrect API key provided: {SECRET}", response=_response(401), body=None
    )
    client, _ = make_client(error)

    with pytest.raises(LLMAPIError) as caught:
        client.generate("s", "u", Answer)

    assert SECRET not in str(caught.value)
    assert SECRET not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__


def test_key_is_redacted_from_schema_errors():
    client, _ = make_client(validation_error(SECRET))

    with pytest.raises(LLMSchemaError) as raised:
        client.generate("s", "u", Answer)

    assert SECRET not in str(raised.value)
    assert raised.value.__cause__ is None


def test_key_is_redacted_from_refusal_errors():
    refusal = SimpleNamespace(type="refusal", refusal=f"no, {SECRET}")
    client, _ = make_client(_reply(content=[refusal]))

    with pytest.raises(LLMRefusalError) as raised:
        client.generate("s", "u", Answer)

    assert SECRET not in str(raised.value)
