import dataclasses

import pytest
from pydantic import BaseModel, ValidationError

from prep_agent.llm.base import (
    LLMAPIError,
    LLMClient,
    LLMConfigError,
    LLMError,
    LLMRefusalError,
    LLMResult,
    LLMSchemaError,
    TokenUsage,
    describe_validation_error,
)


class Answer(BaseModel):
    value: int


class Pair(BaseModel):
    a: int
    b: int


@pytest.mark.parametrize("error", [LLMConfigError, LLMAPIError, LLMRefusalError, LLMSchemaError])
def test_every_error_is_an_llm_error(error):
    assert issubclass(error, LLMError)


def test_error_kinds_are_distinct():
    errors = [LLMConfigError, LLMAPIError, LLMRefusalError, LLMSchemaError]
    for error in errors:
        assert [other for other in errors if other is not error and issubclass(error, other)] == []


def test_result_carries_parsed_model_name_and_usage():
    result = LLMResult(
        parsed=Answer(value=1), model="m", usage=TokenUsage(input_tokens=10, output_tokens=5)
    )

    assert result.parsed == Answer(value=1)
    assert result.model == "m"
    assert result.usage == TokenUsage(input_tokens=10, output_tokens=5)


def test_usage_is_optional():
    assert LLMResult(parsed=Answer(value=1), model="m").usage is None


def test_result_and_usage_are_immutable():
    result = LLMResult(parsed=Answer(value=1), model="m")

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.model = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        TokenUsage(input_tokens=1, output_tokens=1).input_tokens = 2  # type: ignore[misc]


def test_protocol_is_runtime_checkable():
    class Good:
        def generate(self, system, user, schema):
            raise NotImplementedError

    class Bad:
        pass

    assert isinstance(Good(), LLMClient)
    assert not isinstance(Bad(), LLMClient)


def test_validation_errors_are_described_without_the_rejected_values():
    with pytest.raises(ValidationError) as raised:
        Answer.model_validate({"value": "SECRET-VALUE-123"})

    text = describe_validation_error(raised.value)

    assert text.startswith("value: ")
    assert "SECRET-VALUE-123" not in text
    assert "input_value" not in text


def test_every_validation_error_is_listed():
    with pytest.raises(ValidationError) as raised:
        Pair.model_validate({"a": "x", "b": "y"})

    assert describe_validation_error(raised.value).count(";") == 1
