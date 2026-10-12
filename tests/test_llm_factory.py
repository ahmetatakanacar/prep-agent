import pytest

from prep_agent.llm.base import LLMClient, LLMConfigError
from prep_agent.llm.factory import create_llm_client
from prep_agent.llm.openai import OpenAIClient


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("PREP_AGENT_PROVIDER", "PREP_AGENT_MODEL", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_default_provider_is_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    client = create_llm_client()

    assert isinstance(client, OpenAIClient)
    assert isinstance(client, LLMClient)


def test_provider_name_is_case_and_space_insensitive(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("PREP_AGENT_PROVIDER", "  OpenAI ")

    assert isinstance(create_llm_client(), OpenAIClient)


def test_blank_provider_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("PREP_AGENT_PROVIDER", "")

    assert isinstance(create_llm_client(), OpenAIClient)


def test_unknown_provider_raises_config_error_listing_the_options(monkeypatch):
    monkeypatch.setenv("PREP_AGENT_PROVIDER", "nope")

    with pytest.raises(LLMConfigError, match="nope.*openai"):
        create_llm_client()


def test_fake_is_not_available_through_the_factory(monkeypatch):
    monkeypatch.setenv("PREP_AGENT_PROVIDER", "fake")

    with pytest.raises(LLMConfigError):
        create_llm_client()


def test_missing_key_surfaces_as_config_error():
    with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
        create_llm_client()
