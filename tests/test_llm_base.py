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
    format_usage,
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


def test_schema_error_carries_the_usage_and_model_of_the_rejected_answer():
    error = LLMSchemaError("bad", usage=TokenUsage(10, 5), model="m")

    assert str(error) == "bad"
    assert error.usage == TokenUsage(10, 5)
    assert error.model == "m"


def test_schema_error_usage_and_model_default_to_none():
    error = LLMSchemaError("bad")

    assert error.usage is None
    assert error.model is None


def test_result_carries_parsed_model_name_and_usage():
    result = LLMResult(
        parsed=Answer(value=1), model="m", usage=TokenUsage(input_tokens=10, output_tokens=5)
    )

    assert result.parsed == Answer(value=1)
    assert result.model == "m"
    assert result.usage == TokenUsage(input_tokens=10, output_tokens=5)


def test_cache_counts_are_optional_and_unknown_by_default():
    usage = TokenUsage(10, 5)

    assert usage.cached_input_tokens is None
    assert usage.cache_write_tokens is None


def test_usage_without_cache_counts_reads_input_and_output_only():
    assert format_usage(TokenUsage(1200, 340)) == "1200 input / 340 output"


def test_usage_with_cache_counts_names_them_inside_the_input():
    usage = TokenUsage(1200, 340, cached_input_tokens=900, cache_write_tokens=100)

    assert format_usage(usage) == "1200 input (900 cached, 100 cache writes) / 340 output"


def test_usage_with_one_cache_count_names_only_that_one():
    assert format_usage(TokenUsage(1200, 340, cached_input_tokens=0)) == (
        "1200 input (0 cached) / 340 output"
    )


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
