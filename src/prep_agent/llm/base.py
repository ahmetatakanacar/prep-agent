"""The LLM interface every provider adapter implements, plus its result and error types.

The interface is deliberately narrow: given a system message, a user message and a pydantic
model, return a validated instance of that model. Everything provider specific (SDKs, retries,
how structured output is requested) stays in the adapters.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int


class LLMError(Exception):
    """Base class for every error the LLM layer raises."""


class LLMConfigError(LLMError):
    """The client cannot be built: a missing API key, a missing package, an unknown provider."""


class LLMAPIError(LLMError):
    """The provider call failed: network, rate limit, server error or rejected request."""


class LLMRefusalError(LLMError):
    """The model declined to answer."""


class LLMSchemaError(LLMError):
    """The model answered, but the answer does not match the requested schema.

    The answer was still paid for, so the error carries what the provider reported about it
    (None when it reported nothing).
    """

    def __init__(
        self, message: str, *, usage: TokenUsage | None = None, model: str | None = None
    ) -> None:
        super().__init__(message)
        self.usage = usage
        self.model = model


def describe_validation_error(error: ValidationError) -> str:
    """Say where and why validation failed without echoing the rejected values.

    Pydantic's own text quotes the input, which here is the model's answer and may repeat
    profile values; this one can be shown to the model again or logged.
    """
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['msg']}"
        for item in error.errors()
    )


@dataclass(frozen=True)
class LLMResult[T: BaseModel]:
    parsed: T
    model: str
    # None when the provider does not report usage; the benchmark treats that as unknown cost.
    usage: TokenUsage | None = None


@runtime_checkable
class LLMClient(Protocol):
    def generate[T: BaseModel](self, system: str, user: str, schema: type[T]) -> LLMResult[T]:
        """Answer the messages with a validated instance of `schema`.

        Raises an `LLMError` subclass on any failure; adapters never leak provider exceptions.
        """
        ...
