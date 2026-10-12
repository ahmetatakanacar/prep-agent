"""OpenAI adapter: structured outputs through the Responses API.

The `openai` package is imported lazily so that prep_agent imports fine without the optional
extra; building a client without it raises `LLMConfigError`. Retries on rate limits and server
errors are the SDK's own (`max_retries`, 2 by default); this module adds no retry loop.
"""

import os
from types import ModuleType
from typing import Any

from pydantic import BaseModel, ValidationError

from prep_agent.llm.base import (
    LLMAPIError,
    LLMConfigError,
    LLMRefusalError,
    LLMResult,
    LLMSchemaError,
    TokenUsage,
)

# The model the OpenAI docs recommend as the starting point (developers.openai.com/api/docs/models,
# checked 2026-10-13). Models change often, so PREP_AGENT_MODEL overrides it without a code change.
DEFAULT_MODEL = "gpt-6-astra"

API_KEY_VARIABLE = "OPENAI_API_KEY"
MODEL_VARIABLE = "PREP_AGENT_MODEL"
_REDACTED = "[redacted]"


class OpenAIClient:
    """`LLMClient` backed by the OpenAI Responses API.

    `client` is an already built `openai.OpenAI` instance. It exists so tests can inject a fake
    SDK client; when it is omitted the client is built from OPENAI_API_KEY.
    """

    def __init__(self, *, model: str | None = None, client: Any = None) -> None:
        self._model = model or os.environ.get(MODEL_VARIABLE, "").strip() or DEFAULT_MODEL
        self._client = client if client is not None else _build_sdk_client()

    def __repr__(self) -> str:
        # The API key lives only inside the SDK client and must never reach a repr.
        return f"OpenAIClient(model={self._model!r})"

    def generate[T: BaseModel](self, system: str, user: str, schema: type[T]) -> LLMResult[T]:
        openai = _import_openai()
        try:
            response = self._client.responses.parse(
                model=self._model,
                instructions=system,
                input=user,
                text_format=schema,
                # The exchange holds dataset summaries; there is no reason for the provider to
                # keep it.
                store=False,
            )
        except ValidationError as exc:
            raise LLMSchemaError(
                f"model output does not match {schema.__name__}: {self._redact(str(exc))}"
            ) from None
        except openai.OpenAIError as exc:
            status = getattr(exc, "status_code", None)
            where = type(exc).__name__ + (f", HTTP {status}" if status else "")
            raise LLMAPIError(
                f"OpenAI request failed ({where}): {self._redact(str(exc))}"
            ) from None

        for item in response.output or []:
            if item.type != "message":
                continue
            for part in item.content:
                if part.type == "refusal":
                    raise LLMRefusalError(f"model refused: {self._redact(part.refusal)}")

        details = response.incomplete_details
        if response.status == "incomplete":
            reason = details.reason if details is not None else None
            if reason == "content_filter":
                raise LLMRefusalError("response was blocked by the content filter")
            raise LLMSchemaError(f"response was cut off before it was complete ({reason})")

        if response.output_parsed is None:
            raise LLMSchemaError(f"response contained no output for {schema.__name__}")

        usage = response.usage
        return LLMResult(
            parsed=response.output_parsed,
            model=response.model or self._model,
            usage=None
            if usage is None
            else TokenUsage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens),
        )

    def _redact(self, text: str) -> str:
        key = getattr(self._client, "api_key", None)
        if isinstance(key, str) and key:
            return text.replace(key, _REDACTED)
        return text


def _import_openai() -> ModuleType:
    try:
        import openai
    except ImportError:
        raise LLMConfigError(
            "the openai package is not installed; install it with `uv sync --extra openai`"
        ) from None
    return openai


def _build_sdk_client() -> Any:
    openai = _import_openai()
    api_key = os.environ.get(API_KEY_VARIABLE, "").strip()
    if not api_key:
        raise LLMConfigError(
            f"{API_KEY_VARIABLE} is not set; export it or put it in a .env file in the "
            "current directory"
        )
    return openai.OpenAI(api_key=api_key)
