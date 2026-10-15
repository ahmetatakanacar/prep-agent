import pytest
from pydantic import BaseModel

from prep_agent.llm.base import (
    LLMAPIError,
    LLMClient,
    LLMResult,
    LLMSchemaError,
    TokenUsage,
)
from prep_agent.llm.fake import FakeLLM, FakeLLMExhaustedError


class Answer(BaseModel):
    value: int


class Other(BaseModel):
    name: str


def test_fake_satisfies_the_protocol():
    assert isinstance(FakeLLM([]), LLMClient)


def test_returns_responses_in_order():
    llm = FakeLLM([Answer(value=1), Answer(value=2)])

    first = llm.generate("s", "u", Answer)
    second = llm.generate("s", "u", Answer)

    assert [first.parsed, second.parsed] == [Answer(value=1), Answer(value=2)]


def test_result_is_a_result_object_without_usage():
    result = FakeLLM([Answer(value=1)]).generate("s", "u", Answer)

    assert isinstance(result, LLMResult)
    assert result.model == "fake"
    assert result.usage is None


def test_dict_response_is_validated_into_the_schema():
    result = FakeLLM([{"value": 3}]).generate("s", "u", Answer)

    assert result.parsed == Answer(value=3)


def test_json_response_is_validated_into_the_schema():
    result = FakeLLM(['{"value": 4}']).generate("s", "u", Answer)

    assert result.parsed == Answer(value=4)


@pytest.mark.parametrize(
    "response",
    [{"value": "not a number"}, {"other": 1}, '{"value": ', "not json", "[1, 2]"],
)
def test_response_that_does_not_fit_the_schema_raises_schema_error(response):
    with pytest.raises(LLMSchemaError):
        FakeLLM([response]).generate("s", "u", Answer)


def test_schema_error_reports_the_usage_of_the_fake():
    llm = FakeLLM([{"value": "not a number"}], usage=TokenUsage(30, 4))

    with pytest.raises(LLMSchemaError) as raised:
        llm.generate("s", "u", Answer)

    assert raised.value.usage == TokenUsage(30, 4)
    assert raised.value.model == "fake"


def test_schema_error_has_no_usage_when_the_fake_reports_none():
    with pytest.raises(LLMSchemaError) as raised:
        FakeLLM([{"value": "not a number"}]).generate("s", "u", Answer)

    assert raised.value.usage is None


def test_model_instance_of_the_wrong_type_raises_schema_error():
    with pytest.raises(LLMSchemaError):
        FakeLLM([Other(name="x")]).generate("s", "u", Answer)


def test_exception_response_is_raised():
    llm = FakeLLM([LLMAPIError("rate limited"), Answer(value=1)])

    with pytest.raises(LLMAPIError, match="rate limited"):
        llm.generate("s", "u", Answer)
    assert llm.generate("s", "u", Answer).parsed == Answer(value=1)


def test_every_call_is_recorded_including_failed_ones():
    llm = FakeLLM([LLMAPIError("boom"), Answer(value=1)])

    with pytest.raises(LLMAPIError):
        llm.generate("sys 1", "user 1", Answer)
    llm.generate("sys 2", "user 2", Answer)

    assert [(c.system, c.user, c.schema) for c in llm.calls] == [
        ("sys 1", "user 1", Answer),
        ("sys 2", "user 2", Answer),
    ]


def test_empty_queue_raises_a_clear_error():
    llm = FakeLLM([Answer(value=1)])
    llm.generate("s", "u", Answer)

    with pytest.raises(FakeLLMExhaustedError, match="1 call"):
        llm.generate("s", "u", Answer)


def test_exhausted_error_is_not_an_llm_error():
    # Code under test catches LLMError to handle provider failures; a missing fake response is
    # a bug in the test and must not be swallowed there.
    from prep_agent.llm.base import LLMError

    assert not issubclass(FakeLLMExhaustedError, LLMError)


def test_calls_are_recorded_even_when_the_queue_is_empty():
    llm = FakeLLM([])

    with pytest.raises(FakeLLMExhaustedError):
        llm.generate("s", "u", Answer)

    assert len(llm.calls) == 1


def test_remaining_counts_unused_responses():
    llm = FakeLLM([Answer(value=1), Answer(value=2)])
    llm.generate("s", "u", Answer)

    assert llm.remaining == 1


def test_usage_given_to_the_fake_is_reported_on_every_call():
    llm = FakeLLM([Answer(value=1), Answer(value=2)], usage=TokenUsage(100, 10))

    results = [llm.generate("s", "u", Answer) for _ in range(2)]

    assert [r.usage for r in results] == [TokenUsage(100, 10)] * 2
