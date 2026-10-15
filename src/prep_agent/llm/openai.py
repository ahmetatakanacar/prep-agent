"""OpenAI adapter: strict structured outputs through the Responses API.

The request carries an explicit JSON schema (`text.format` of type `json_schema`, strict) that
is generated from the pydantic model by `strict_schema.to_strict_schema`, because the SDK's own
`responses.parse` helper sends pydantic's schema as it is and OpenAI rejects its `oneOf`. The
answer text is validated by the original model, never by the converted schema.

The `openai` package is imported lazily so that prep_agent imports fine without the optional
extra; building a client without it raises `LLMConfigError`. Retries on rate limits and server
errors are the SDK's own (`max_retries`, 2 by default); this module adds no retry loop.
"""

import json
import os
import re
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
    describe_validation_error,
)
from prep_agent.llm.strict_schema import drop_added_nulls, to_strict_schema

# The middle tier: close to the flagship's quality at a lower price, which suits a planner that
# runs many times in benchmarks. PREP_AGENT_MODEL overrides it without a code change.
DEFAULT_MODEL = "gpt-6.1-sol"

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
            response = self._client.responses.create(
                model=self._model,
                instructions=system,
                input=user,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": _format_name(schema),
                        "strict": True,
                        "schema": to_strict_schema(schema),
                    }
                },
                # The exchange holds dataset summaries; there is no reason for the provider to
                # keep it.
                store=False,
            )
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

        text = response.output_text
        if not text:
            raise LLMSchemaError(f"response contained no output for {schema.__name__}")
        try:
            parsed = schema.model_validate_json(
                json.dumps(drop_added_nulls(json.loads(text), schema))
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            problem = (
                describe_validation_error(exc) if isinstance(exc, ValidationError) else str(exc)
            )
            raise LLMSchemaError(
                f"model output does not match {schema.__name__}: {self._redact(problem)}"
            ) from None

        usage = response.usage
        return LLMResult(
            parsed=parsed,
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


def _format_name(schema: type[BaseModel]) -> str:
    # The API only accepts letters, digits, underscores and dashes, at most 64 characters.
    return re.sub(r"[^A-Za-z0-9_-]", "_", schema.__name__)[:64]


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
