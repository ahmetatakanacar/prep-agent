import json

import pytest
from pydantic import ValidationError

from prep_agent.plan import (
    CastToCategoryStep,
    DedupeStep,
    DropColumnsStep,
    DropMissingTargetStep,
    EncodeStep,
    ImputeStep,
    Plan,
    ScaleStep,
    SplitStep,
)

REASON = "because the data needs it"


def make_plan(steps) -> dict:
    return {
        "target": "churned",
        "task_type": "classification",
        "model_family": "tree",
        "steps": steps,
    }


ALL_STEPS = [
    {"op": "dedupe", "subset": None, "keep": "first", "reason": REASON},
    {"op": "drop_columns", "columns": ["id"], "reason": REASON},
    {"op": "drop_missing_target", "reason": REASON},
    {"op": "cast_to_category", "columns": ["hour"], "reason": REASON},
    {"op": "split", "stratify": True, "reason": REASON},
    {"op": "impute", "columns": ["age"], "strategy": "median", "reason": REASON},
    {"op": "encode", "columns": ["plan"], "method": "onehot", "reason": REASON},
    {"op": "scale", "columns": ["age"], "method": "standard", "reason": REASON},
]


# --- valid steps -----------------------------------------------------------------------------


def test_dedupe_accepts_subset_or_null():
    assert DedupeStep(op="dedupe", subset=None, keep="last", reason=REASON).subset is None
    step = DedupeStep(op="dedupe", subset=["a", "b"], keep="first", reason=REASON)
    assert step.subset == ["a", "b"]


def test_dedupe_subset_defaults_to_all_columns():
    step = DedupeStep.model_validate({"op": "dedupe", "keep": "first", "reason": REASON})

    assert step.subset is None


def test_split_defaults():
    step = SplitStep(op="split", stratify=False, reason=REASON)

    assert step.test_size == 0.2
    assert step.random_state == 42


def test_impute_defaults_and_constant_fill_values():
    step = ImputeStep(op="impute", columns=["a"], strategy="mean", reason=REASON)
    assert step.fill_value is None
    assert step.add_indicator is False

    assert (
        ImputeStep(
            op="impute", columns=["a"], strategy="constant", fill_value=0, reason=REASON
        ).fill_value
        == 0
    )
    assert (
        ImputeStep(
            op="impute", columns=["a"], strategy="constant", fill_value="missing", reason=REASON
        ).fill_value
        == "missing"
    )
    assert (
        ImputeStep(
            op="impute", columns=["a"], strategy="constant", fill_value=1.5, reason=REASON
        ).fill_value
        == 1.5
    )


def test_impute_explicit_null_fill_value_is_allowed_for_non_constant_strategies():
    step = ImputeStep.model_validate(
        {
            "op": "impute",
            "columns": ["a"],
            "strategy": "median",
            "fill_value": None,
            "reason": REASON,
        }
    )

    assert step.fill_value is None


def test_encode_max_categories_for_onehot():
    step = EncodeStep(op="encode", columns=["a"], method="onehot", max_categories=5, reason=REASON)

    assert step.max_categories == 5
    assert (
        EncodeStep(op="encode", columns=["a"], method="onehot", reason=REASON).max_categories
        is None
    )


# --- invalid steps ---------------------------------------------------------------------------


@pytest.mark.parametrize("reason", ["", "   ", "\n\t"])
@pytest.mark.parametrize(
    "step",
    ALL_STEPS,
    ids=[s["op"] for s in ALL_STEPS],
)
def test_empty_reason_is_rejected(step, reason):
    plan = make_plan([{**step, "reason": reason}])

    with pytest.raises(ValidationError, match="reason"):
        Plan.model_validate(plan)


@pytest.mark.parametrize("step", ALL_STEPS, ids=[s["op"] for s in ALL_STEPS])
def test_missing_reason_is_rejected(step):
    without_reason = {k: v for k, v in step.items() if k != "reason"}

    with pytest.raises(ValidationError, match="reason"):
        Plan.model_validate(make_plan([without_reason]))


@pytest.mark.parametrize("step", ALL_STEPS, ids=[s["op"] for s in ALL_STEPS])
def test_unknown_fields_are_rejected(step):
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        Plan.model_validate(make_plan([{**step, "surprise": 1}]))


@pytest.mark.parametrize("test_size", [0, 0.0, 1, 1.0, 1.5, -0.2])
def test_split_test_size_must_be_between_zero_and_one(test_size):
    with pytest.raises(ValidationError, match="test_size"):
        SplitStep(op="split", test_size=test_size, stratify=False, reason=REASON)


def test_split_random_state_must_be_a_non_negative_integer():
    with pytest.raises(ValidationError, match="random_state"):
        SplitStep(op="split", random_state=-1, stratify=False, reason=REASON)
    with pytest.raises(ValidationError, match="random_state"):
        SplitStep(op="split", random_state=1.5, stratify=False, reason=REASON)


def test_split_requires_an_explicit_stratify_choice():
    with pytest.raises(ValidationError, match="stratify"):
        SplitStep.model_validate({"op": "split", "reason": REASON})


def test_dedupe_requires_keep_and_rejects_unknown_keep():
    with pytest.raises(ValidationError, match="keep"):
        DedupeStep.model_validate({"op": "dedupe", "reason": REASON})
    with pytest.raises(ValidationError, match="keep"):
        DedupeStep(op="dedupe", keep="middle", reason=REASON)


def test_dedupe_subset_must_not_be_an_empty_list():
    with pytest.raises(ValidationError, match="subset"):
        DedupeStep(op="dedupe", subset=[], keep="first", reason=REASON)


@pytest.mark.parametrize(
    "model, kwargs",
    [
        (DropColumnsStep, {"op": "drop_columns"}),
        (CastToCategoryStep, {"op": "cast_to_category"}),
        (ImputeStep, {"op": "impute", "strategy": "mean"}),
        (EncodeStep, {"op": "encode", "method": "ordinal"}),
        (ScaleStep, {"op": "scale", "method": "robust"}),
    ],
)
@pytest.mark.parametrize("columns", [[], [""], ["a", "a"], "a"])
def test_columns_must_be_a_non_empty_list_of_distinct_names(model, kwargs, columns):
    with pytest.raises(ValidationError, match="columns"):
        model(columns=columns, reason=REASON, **kwargs)


def test_impute_constant_requires_fill_value():
    with pytest.raises(ValidationError, match="fill_value"):
        ImputeStep(op="impute", columns=["a"], strategy="constant", reason=REASON)


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_impute_rejects_fill_value_for_other_strategies(strategy):
    with pytest.raises(ValidationError, match="fill_value"):
        ImputeStep(op="impute", columns=["a"], strategy=strategy, fill_value=0, reason=REASON)


def test_impute_fill_value_must_not_be_a_boolean():
    with pytest.raises(ValidationError, match="fill_value"):
        ImputeStep(op="impute", columns=["a"], strategy="constant", fill_value=True, reason=REASON)


def test_impute_rejects_unknown_strategy():
    with pytest.raises(ValidationError, match="strategy"):
        ImputeStep(op="impute", columns=["a"], strategy="knn", reason=REASON)


def test_encode_rejects_unknown_method():
    with pytest.raises(ValidationError, match="method"):
        EncodeStep(op="encode", columns=["a"], method="target", reason=REASON)


def test_encode_max_categories_requires_onehot():
    with pytest.raises(ValidationError, match="max_categories"):
        EncodeStep(op="encode", columns=["a"], method="ordinal", max_categories=5, reason=REASON)


@pytest.mark.parametrize("max_categories", [0, 1, -3])
def test_encode_max_categories_must_be_at_least_two(max_categories):
    with pytest.raises(ValidationError, match="max_categories"):
        EncodeStep(
            op="encode",
            columns=["a"],
            method="onehot",
            max_categories=max_categories,
            reason=REASON,
        )


def test_scale_rejects_unknown_method():
    with pytest.raises(ValidationError, match="method"):
        ScaleStep(op="scale", columns=["a"], method="log", reason=REASON)


# --- plan ------------------------------------------------------------------------------------


def test_plan_parses_every_step_type_in_order():
    plan = Plan.model_validate(make_plan(ALL_STEPS))

    assert [type(s) for s in plan.steps] == [
        DedupeStep,
        DropColumnsStep,
        DropMissingTargetStep,
        CastToCategoryStep,
        SplitStep,
        ImputeStep,
        EncodeStep,
        ScaleStep,
    ]


def test_unknown_op_name_is_a_validation_error():
    plan = make_plan([{"op": "one_hot_everything", "columns": ["a"], "reason": REASON}])

    with pytest.raises(ValidationError, match="one_hot_everything|op"):
        Plan.model_validate(plan)


def test_missing_op_name_is_a_validation_error():
    with pytest.raises(ValidationError, match="op"):
        Plan.model_validate(make_plan([{"columns": ["a"], "reason": REASON}]))


def test_unknown_op_in_json_text_is_a_validation_error():
    text = json.dumps(make_plan([{"op": "run_python", "code": "import os", "reason": REASON}]))

    with pytest.raises(ValidationError):
        Plan.model_validate_json(text)


def test_plan_round_trips_through_json():
    plan = Plan.model_validate(make_plan(ALL_STEPS))

    restored = Plan.model_validate_json(plan.model_dump_json())

    assert restored == plan
    assert [type(s) for s in restored.steps] == [type(s) for s in plan.steps]


@pytest.mark.parametrize(
    "field, value",
    [
        ("task_type", "clustering"),
        ("model_family", "neural"),
        ("target", ""),
        ("target", "  "),
    ],
)
def test_plan_rejects_invalid_header_fields(field, value):
    data = {**make_plan([]), field: value}

    with pytest.raises(ValidationError, match=field):
        Plan.model_validate(data)


@pytest.mark.parametrize("field", ["target", "task_type", "model_family", "steps"])
def test_plan_requires_header_fields(field):
    data = make_plan([])
    del data[field]

    with pytest.raises(ValidationError, match=field):
        Plan.model_validate(data)


def test_plan_json_schema_can_be_generated():
    schema = Plan.model_json_schema()

    json.dumps(schema)
    assert set(schema["required"]) >= {"target", "task_type", "model_family", "steps"}
    assert schema["additionalProperties"] is False
    step_schemas = schema["$defs"]
    ops = {
        definition["properties"]["op"]["const"]
        for name, definition in step_schemas.items()
        if name.endswith("Step")
    }
    assert ops == {s["op"] for s in ALL_STEPS}
    for name, definition in step_schemas.items():
        if name.endswith("Step"):
            assert definition["additionalProperties"] is False
            assert {"op", "reason"} <= set(definition["required"])
