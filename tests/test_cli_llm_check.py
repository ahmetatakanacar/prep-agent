import pytest

from prep_agent.llm.base import (
    LLMAPIError,
    LLMConfigError,
    LLMRefusalError,
    LLMResult,
    TokenUsage,
)
from prep_agent.llm.fake import FakeLLM
from prep_agent.llm.ping import PingReply


class UsageLLM:
    def generate(self, system, user, schema):
        return LLMResult(
            parsed=schema(reply="pong"),
            model="gpt-served",
            usage=TokenUsage(input_tokens=12, output_tokens=4),
        )


def use_client(monkeypatch, client):
    monkeypatch.setattr("prep_agent.cli.create_llm_client", lambda: client)


def test_loads_dotenv_before_running_a_command(invoke_cli, monkeypatch):
    loaded = []
    monkeypatch.setattr("prep_agent.cli.load_dotenv_file", lambda: loaded.append(True))
    use_client(monkeypatch, FakeLLM([{"reply": "pong"}]))

    invoke_cli(["llm-check"])

    assert loaded == [True]


def test_success_prints_model_and_token_usage(invoke_cli, monkeypatch):
    use_client(monkeypatch, UsageLLM())

    result = invoke_cli(["llm-check"])

    assert result.exit_code == 0, result.output
    assert "gpt-served" in result.output
    assert "12" in result.output and "4" in result.output


def test_success_without_usage_says_so(invoke_cli, monkeypatch):
    use_client(monkeypatch, FakeLLM([{"reply": "pong"}]))

    result = invoke_cli(["llm-check"])

    assert result.exit_code == 0, result.output
    assert "fake" in result.output
    assert "not reported" in result.output


def test_sends_one_request_with_the_ping_schema(invoke_cli, monkeypatch):
    llm = FakeLLM([{"reply": "pong"}])
    use_client(monkeypatch, llm)

    invoke_cli(["llm-check"])

    [call] = llm.calls
    assert call.schema is PingReply
    assert call.system and call.user


@pytest.mark.parametrize(
    "error",
    [
        LLMConfigError("OPENAI_API_KEY is not set"),
        LLMAPIError("OpenAI request failed (RateLimitError, HTTP 429): slow down"),
        LLMRefusalError("model refused: no"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_llm_errors_print_the_message_and_exit_1(invoke_cli, monkeypatch, error):
    use_client(monkeypatch, FakeLLM([error]))

    result = invoke_cli(["llm-check"])

    assert result.exit_code == 1
    assert str(error) in result.output.replace("\n", " ")


def test_client_construction_errors_exit_1(invoke_cli, monkeypatch):
    def fail():
        raise LLMConfigError("unknown provider 'nope'")

    monkeypatch.setattr("prep_agent.cli.create_llm_client", fail)

    result = invoke_cli(["llm-check"])

    assert result.exit_code == 1
    assert "unknown provider" in result.output


def test_without_a_key_the_real_factory_gives_a_clear_error(invoke_cli, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PREP_AGENT_PROVIDER", raising=False)

    result = invoke_cli(["llm-check"])

    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output
    assert "Traceback" not in result.output


def test_the_api_key_is_never_printed(invoke_cli, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-SECRET-0123456789")
    use_client(monkeypatch, UsageLLM())

    result = invoke_cli(["llm-check"])

    assert "SECRET" not in result.output
