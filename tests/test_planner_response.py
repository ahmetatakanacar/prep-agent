import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from prep_agent.llm.strict_schema import drop_added_nulls, to_strict_schema
from prep_agent.plan import Plan
from prep_agent.planner import PlannerResponse, Question

PLAN = {
    "target": "t",
    "task_type": "classification",
    "model_family": "any",
    "steps": [{"op": "split", "stratify": True, "reason": "hold out a test set"}],
}


def question(**overrides) -> dict:
    base = {
        "id": "q1",
        "question": "Which column is the target?",
        "options": ["churned", "plan"],
        "recommended": "churned",
        "reason": "the goal does not name a target",
    }
    return {**base, **overrides}


def test_plan_response_holds_a_plan():
    response = PlannerResponse.model_validate({"kind": "plan", "plan": PLAN, "questions": None})

    assert isinstance(response.plan, Plan)
    assert response.questions is None


def test_questions_response_holds_questions():
    response = PlannerResponse.model_validate(
        {"kind": "questions", "plan": None, "questions": [question()]}
    )

    assert response.plan is None
    assert response.questions is not None and response.questions[0].id == "q1"


def test_unused_fields_may_be_left_out():
    assert PlannerResponse.model_validate({"kind": "plan", "plan": PLAN}).questions is None
    assert (
        PlannerResponse.model_validate({"kind": "questions", "questions": [question()]}).plan
        is None
    )


def test_an_empty_questions_list_next_to_a_plan_is_the_same_as_none():
    response = PlannerResponse.model_validate({"kind": "plan", "plan": PLAN, "questions": []})

    assert response.questions is None


@pytest.mark.parametrize(
    "data",
    [
        {"kind": "plan", "plan": None, "questions": None},
        {"kind": "plan", "plan": PLAN, "questions": [question()]},
        {"kind": "questions", "plan": None, "questions": None},
        {"kind": "questions", "plan": None, "questions": []},
        {"kind": "questions", "plan": PLAN, "questions": [question()]},
        {"kind": "answer", "plan": PLAN, "questions": None},
    ],
    ids=[
        "plan-kind-without-plan",
        "plan-kind-with-questions",
        "questions-kind-without-questions",
        "questions-kind-with-empty-list",
        "questions-kind-with-plan",
        "unknown-kind",
    ],
)
def test_inconsistent_kind_and_fields_are_rejected(data):
    with pytest.raises(ValidationError):
        PlannerResponse.model_validate(data)


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        PlannerResponse.model_validate({"kind": "plan", "plan": PLAN, "note": "hi"})


def test_question_ids_must_be_distinct_within_a_response():
    with pytest.raises(ValidationError, match="distinct"):
        PlannerResponse.model_validate(
            {"kind": "questions", "questions": [question(), question(question="Again?")]}
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"options": ["only one"], "recommended": "only one"},
        {"options": ["a", "b", "c", "d", "e", "f"], "recommended": "a"},
        {"options": ["a", "a"], "recommended": "a"},
        {"options": ["a", " "], "recommended": "a"},
        {"recommended": "somewhere else"},
        {"id": " "},
        {"question": ""},
        {"reason": " "},
    ],
    ids=[
        "one-option",
        "six-options",
        "duplicate-options",
        "blank-option",
        "recommended-not-an-option",
        "blank-id",
        "empty-question",
        "blank-reason",
    ],
)
def test_invalid_questions_are_rejected(overrides):
    with pytest.raises(ValidationError):
        Question.model_validate(question(**overrides))


@pytest.mark.parametrize("count", [2, 5])
def test_two_to_five_options_are_accepted(count):
    options = [f"option {i}" for i in range(count)]

    parsed = Question.model_validate(question(options=options, recommended=options[0]))

    assert parsed.options == options


def test_schema_converts_to_strict_mode_and_accepts_a_typical_answer():
    schema = to_strict_schema(PlannerResponse)

    Draft202012Validator.check_schema(schema)
    answer = {"kind": "questions", "plan": None, "questions": [question()]}
    assert not list(Draft202012Validator(schema).iter_errors(answer))


def test_strict_schema_answer_with_added_nulls_validates_after_cleanup():
    step = {"op": "split", "reason": "r", "test_size": None, "stratify": True, "random_state": None}
    plan = {**PLAN, "steps": [step]}
    answer = {"kind": "plan", "plan": plan, "questions": None}

    cleaned = drop_added_nulls(answer, PlannerResponse)

    parsed = PlannerResponse.model_validate(cleaned)
    assert parsed.plan is not None and parsed.plan.steps[0].test_size == 0.2
