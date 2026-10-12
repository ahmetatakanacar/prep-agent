"""A scripted LLM client for tests. It never touches the network."""

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from prep_agent.llm.base import LLMResult, LLMSchemaError

FakeResponse = BaseModel | dict | str | BaseException


class FakeLLMExhaustedError(RuntimeError):
    """The fake was called with no responses left.

    Deliberately not an `LLMError`: code that handles provider failures must not swallow what
    is really a mistake in the test script.
    """


@dataclass(frozen=True)
class FakeCall:
    system: str
    user: str
    schema: type[BaseModel]


class FakeLLM:
    """Answers calls from a fixed queue, one response per call.

    A response is a model instance, a dict, a JSON string or an exception to raise. Dicts and
    JSON strings are validated against the requested schema exactly like a real adapter would,
    so a wrong fake fails with `LLMSchemaError` instead of passing silently.
    """

    def __init__(self, responses: Iterable[FakeResponse]) -> None:
        self._queue: deque[FakeResponse] = deque(responses)
        self.calls: list[FakeCall] = []

    @property
    def remaining(self) -> int:
        return len(self._queue)

    def generate[T: BaseModel](self, system: str, user: str, schema: type[T]) -> LLMResult[T]:
        self.calls.append(FakeCall(system=system, user=user, schema=schema))
        if not self._queue:
            raise FakeLLMExhaustedError(
                f"FakeLLM has no responses left after {len(self.calls) - 1} call(s)"
            )
        response = self._queue.popleft()
        if isinstance(response, BaseException):
            raise response
        return LLMResult(parsed=_validate(response, schema), model="fake")


def _validate[T: BaseModel](response: BaseModel | dict | str, schema: type[T]) -> T:
    if isinstance(response, BaseModel):
        if not isinstance(response, schema):
            raise LLMSchemaError(
                f"fake response is a {type(response).__name__}, expected {schema.__name__}"
            )
        return response
    try:
        if isinstance(response, str):
            return schema.model_validate_json(response)
        return schema.model_validate(response)
    except ValidationError as exc:
        raise LLMSchemaError(f"fake response does not match {schema.__name__}: {exc}") from exc
