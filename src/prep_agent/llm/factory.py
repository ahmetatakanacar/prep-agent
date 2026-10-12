"""Builds the LLM client for the configured provider."""

import os

from prep_agent.llm.base import LLMClient, LLMConfigError

PROVIDER_VARIABLE = "PREP_AGENT_PROVIDER"
DEFAULT_PROVIDER = "openai"


def create_llm_client() -> LLMClient:
    """Return a client for the provider named by PREP_AGENT_PROVIDER (default: openai).

    FakeLLM is not a provider: tests construct it directly.
    """
    provider = os.environ.get(PROVIDER_VARIABLE, "").strip().lower() or DEFAULT_PROVIDER
    if provider == "openai":
        from prep_agent.llm.openai import OpenAIClient

        return OpenAIClient()
    raise LLMConfigError(f"unknown provider {provider!r} in {PROVIDER_VARIABLE}; supported: openai")
