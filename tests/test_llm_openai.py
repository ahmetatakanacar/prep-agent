import builtins
import json
from types import SimpleNamespace

import httpx2
import openai
import pytest
from pydantic import BaseModel

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
from prep_agent.llm.strict_schema import to_strict_schema
from prep_agent.plan import Plan

SECRET = "sk-test-SECRET-0123456789"


class Answer(BaseModel):
    value: int


class WithDefault(BaseModel):
    value: int
    size: int = 3


def _request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.openai.test/v1/responses")


def _response(status: int) -> httpx2.Response:
    return httpx2.Response(status, request=_request())


def _message(*content):
    return SimpleNamespace(type="message", content=list(content))


def _text_item(text):
    return SimpleNamespace(type="output_text", text=text)


def _reply(
    parsed=None,
    *,
    text=None,
    content=None,
    model="gpt-test-2026",
    usage=(11, 7),
    status="completed",
    reason=None,
):
    """Shape of the parts of a Responses API result that the adapter reads."""
    if text is None:
        text = "" if parsed is None else parsed.model_dump_json()
    if content is None:
        content = [_text_item(text)]
    return SimpleNamespace(
        status=status,
        incomplete_details=None if reason is None else SimpleNamespace(reason=reason),
        output=[_message(*content)],
        output_text="" if content and content[0].type == "refusal" else text,
        model=model,
        usage=None
        if usage is None
        else SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1]),
    )


class FakeSDK:
    """Stands in for `openai.OpenAI`: records `responses.create` kwargs, replays one outcome."""

    def __init__(self, outcome):
        self.api_key = SECRET
        self.calls: list[dict] = []
        self._outcome = outcome
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


def make_client(outcome, **kwargs) -> tuple[OpenAIClient, FakeSDK]:
    sdk = FakeSDK(outcome)
    return OpenAIClient(client=sdk, **kwargs), sdk


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("OPENAI_API_KEY", "PREP_AGENT_MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_satisfies_the_protocol():
    client, _ = make_client(_reply(Answer(value=1)))

    assert isinstance(client, LLMClient)


def test_sends_messages_and_the_strict_schema_to_the_responses_api():
    client, sdk = make_client(_reply(Answer(value=1)), model="gpt-explicit")

    client.generate("the system", "the user", Answer)

    [call] = sdk.calls
    assert call["model"] == "gpt-explicit"
    assert call["instructions"] == "the system"
    assert call["input"] == "the user"
    assert call["text"] == {
        "format": {
            "type": "json_schema",
            "name": "Answer",
            "strict": True,
            "schema": to_strict_schema(Answer),
        }
    }


def test_the_schema_sent_for_plan_is_the_converted_one():
    plan = Plan.model_validate(
        {"target": "t", "task_type": "regression", "model_family": "any", "steps": []}
    )
    client, sdk = make_client(_reply(plan))

    client.generate("s", "u", Plan)

    sent = sdk.calls[0]["text"]["format"]["schema"]
    assert sent == to_strict_schema(Plan)
    assert "oneOf" not in json.dumps(sent)


def test_schema_name_is_safe_for_the_api():
    class Odd(BaseModel):
        value: int

    Odd.__name__ = "Odd Name.v2"
    client, sdk = make_client(_reply(Odd(value=1)))

    client.generate("s", "u", Odd)

    assert sdk.calls[0]["text"]["format"]["name"] == "Odd_Name_v2"


def test_null_for_an_optional_field_means_use_the_default():
    client, sdk = make_client(_reply(text='{"value": 1, "size": null}'))

    result = client.generate("s", "u", WithDefault)

    assert result.parsed == WithDefault(value=1, size=3)
    assert "default" not in json.dumps(sdk.calls[0]["text"]["format"]["schema"])


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


def test_default_model_is_gpt_6_1_sol():
    assert DEFAULT_MODEL == "gpt-6.1-sol"


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


@pytest.mark.parametrize(
    "text",
    ['{"value": "not a number"}', '{"other": 1}', '{"value": ', "not json", "[1, 2]", "null"],
)
def test_output_that_does_not_fit_the_model_raises_schema_error(text):
    client, _ = make_client(_reply(text=text))

    with pytest.raises(LLMSchemaError, match="Answer"):
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


def real_sdk_replying(text: str, seen: list) -> openai.OpenAI:
    """A real SDK client whose HTTP transport records requests and answers with `text`."""

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
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
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

    return openai.OpenAI(
        api_key=SECRET,
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )


def test_real_sdk_client_sends_a_strict_json_schema_request_and_parses_the_reply():
    seen = []
    sdk = real_sdk_replying('{"value": 3}', seen)

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
    assert body["text"]["format"] == {
        "type": "json_schema",
        "name": "Answer",
        "strict": True,
        "schema": to_strict_schema(Answer),
    }


def test_real_sdk_client_sends_the_converted_plan_schema_and_parses_a_plan():
    seen = []
    answer = {
        "target": "churned",
        "task_type": "classification",
        "model_family": "any",
        "steps": [
            {
                "op": "split",
                "reason": "hold out a test set",
                "test_size": None,
                "stratify": True,
                "random_state": None,
            }
        ],
    }
    sdk = real_sdk_replying(json.dumps(answer), seen)

    result = OpenAIClient(model="gpt-x", client=sdk).generate("s", "u", Plan)

    assert result.parsed.steps[0].test_size == 0.2
    sent = json.loads(seen[0].content)["text"]["format"]["schema"]
    assert sent == to_strict_schema(Plan)
    for keyword in ("oneOf", "discriminator", '"default"'):
        assert keyword not in json.dumps(sent)


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
    client, _ = make_client(_reply(text=f'{{"value": "{SECRET}"}}'))

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
